---
title: "Phase 01 — MongoDB + JWT + Bcrypt Core (Server-Side)"
phase: "01"
plan: "260326-0157-web-bridge-auth-improvements"
status: done
priority: P1
effort: ~4h
depends_on: []
blocks: ["phase-03-client-jwt-integration"]
---

# Phase 01 — MongoDB + JWT + Bcrypt Core (Server-Side)

## Context Links

- Plan: [plan.md](plan.md)
- Next phase: [phase-02-rate-limit-hardening.md](phase-02-rate-limit-hardening.md)
- Client integration: [phase-03-client-jwt-integration.md](phase-03-client-jwt-integration.md)

## Overview

| Field | Value |
|-------|-------|
| Date | 2026-03-26 |
| Description | Replace env-var credentials with MongoDB user store; bcrypt password hashing; JWT session tokens; per-command validation; security headers |
| Priority | P1 |
| Implementation status | done |
| Review status | done |

## Key Insights

- `bcrypt = "0.15"` and `jsonwebtoken = "9.2"` are already in `Cargo.toml` — zero dependency additions for those.
- MongoDB is a **new dependency** — add `mongodb` crate to `Cargo.toml`. The official `mongodb` crate is async-native (tokio).
- Auth happens in the Socket.IO namespace closure starting at line 253 of `main.rs`. The comparison at line 267 is a single expression — replaced with MongoDB lookup + bcrypt verify.
- User schema: `{ username, password_hash, role, enabled, created_at, updated_at }` in `qm_hub.robo_control_user`. Role stored for future RBAC but not enforced in this phase.
- On first startup, seed a default admin user if collection is empty. Guard: refuse to run with the seeded default password unless `ALLOW_DEFAULT_CREDENTIALS=true`.
- JWT TTL: 1 hour (validated decision). Proactive client refresh at ~55 min.
- JWT validation: on **command events only** (rover_command, arm_command, tracking_command, tts_command, fleet_select, audio_stream). Background sweep handles expiry for idle sessions.

## Requirements

1. **MongoDB connection**
   - Add `mongodb` crate to `Cargo.toml`.
   - Read `MONGODB_URI` from environment (e.g., `mongodb://localhost:27017`).
   - Create `.env` file at project root with `MONGODB_URI` template. Add `.env` to `.gitignore`.
   - Connect on startup, get `qm_hub` database, `robo_control_user` collection.
   - Fail startup with clear error if MongoDB unreachable.

2. **User schema & seed**
   - Define `User` struct: `{ username: String, password_hash: String, role: String, enabled: bool, created_at: DateTime, updated_at: DateTime }`.
   - On startup: if `robo_control_user` collection is empty, insert default admin user with bcrypt-hashed "password".
   - Log warning: "Default admin user created — change password immediately."
   - Create unique index on `username` field.

3. **Bcrypt password verification**
   - On auth: look up user by `username` in MongoDB.
   - If user not found or `enabled == false`: return auth error.
   - Verify password with `bcrypt::verify(candidate, user.password_hash)` — constant-time.
   - Use `tokio::task::spawn_blocking` for bcrypt verify (CPU-bound, ~250ms).

4. **JWT session tokens**
   - On successful auth, emit `auth_token` event with a signed JWT string.
   - JWT claims: `sub` (username), `role`, `iat`, `exp` (`iat + SESSION_TTL_SECONDS`), `jti` (UUID v4).
   - Sign with HS256 using `JWT_SECRET` env var. If not set, auto-generate 32 random bytes as hex, log `WARN` loudly.
   - `SESSION_TTL_SECONDS` env var, default `3600` (1 hour).

5. **Token validation on reconnect**
   - Extend `AuthCredentials` struct: `token: Option<String>`.
   - In namespace handler: if `token` present AND valid, skip password check and authenticate directly.
   - If `token` present but invalid/expired, fall through to password auth (graceful credential re-entry).
   - `auth_refresh` event handler: client sends current JWT, server validates and returns new one with refreshed `exp`.

6. **`auth_error` event**
   - On auth failure emit `auth_error` with `{ reason: "invalid_credentials" | "token_expired" | "rate_limited" | "account_disabled" }` before disconnecting.

7. **Per-command JWT validation (SEC-005)**
   - Store `(socket_id → JWT claims)` in `SessionRegistry` struct in `security.rs`.
   - Reusable guard function `check_session_valid(socket_id, registry)` called at top of **command event handlers**: `rover_command`, `arm_command`, `tracking_command`, `tts_command`, `fleet_select`, `audio_stream`.
   - NOT called on server→client events or read-only events (`camera_control`, `audio_control`, `performance_control`).
   - On validation failure: emit `auth_error { reason: "token_expired" }` and disconnect.
   - Background task every 60s to sweep and disconnect expired sessions.

8. **Default credential guard**
   - If seeded admin user still has default password AND `ALLOW_DEFAULT_CREDENTIALS != "true"`, exit with `process::exit(1)`.
   - Detection: on startup, load admin user from MongoDB, attempt `bcrypt::verify("password", hash)`. If true → default password still in use.

9. **Dataflow YAML cleanup (SEC-002)**
   - Remove hardcoded `AUTH_USERNAME: "admin"` and `AUTH_PASSWORD: "password"` from `orchestra/orchestra-dataflow.yml`.
   - Add `MONGODB_URI` env var to dataflow YAML.
   - Remove `AUTH_USERNAME`/`AUTH_PASSWORD` env vars entirely (no longer used).

10. **Security headers**
    - Add to Axum router:
      - `X-Frame-Options: DENY`
      - `X-Content-Type-Options: nosniff`
      - `Strict-Transport-Security: max-age=31536000; includeSubDomains`
      - `Referrer-Policy: no-referrer`

## Architecture

```
startup
  ├── load MONGODB_URI from env / .env file
  ├── connect to MongoDB → qm_hub.robo_control_user
  ├── ensure unique index on username
  ├── if collection empty → seed admin user (bcrypt hash "password")
  ├── if admin has default password && !ALLOW_DEFAULT_CREDENTIALS → exit(1)
  ├── load JWT_SECRET (or auto-generate + WARN)
  └── load SESSION_TTL_SECONDS (default 3600)

socket.io ns "/" handler
  ├── check auth rate limit (existing)
  ├── parse AuthCredentials { username, password, token? }
  ├── if token present → validate_jwt(token)
  │     ├── valid + not expired → authenticated, skip MongoDB lookup
  │     └── invalid/expired    → fall through to password path
  ├── password path → MongoDB find_one(username) → bcrypt::verify (spawn_blocking)
  │     ├── user not found / disabled → auth_error { reason }
  │     └── password mismatch → auth_error { reason }
  ├── on failure → emit auth_error, disconnect
  └── on success
        ├── emit auth_token (signed JWT with sub, role, exp, jti)
        ├── register (socket_id, claims) in SessionRegistry
        └── continue with existing event wiring

per-command validation (command events only)
  └── check_session_valid(socket_id) → if expired: disconnect + auth_error

background sweep (every 60s)
  └── SessionRegistry::sweep_expired() → disconnect stale sessions
```

## Related Code Files

**Owned by this phase:**
- `common/web_bridge/src/main.rs` — auth flow (lines 233–290), startup, security headers
- `common/web_bridge/src/security.rs` — JWT module, SessionRegistry, credential validation
- `common/web_bridge/Cargo.toml` — add `mongodb`, `rand`, `dotenv`; enable tower-http `set-response-header` feature

**Also modified:**
- `orchestra/orchestra-dataflow.yml` — remove AUTH_USERNAME/AUTH_PASSWORD, add MONGODB_URI
- `.gitignore` — add `.env`
- `.env` (new) — MONGODB_URI template

**Read-only reference:**
- `common/web_bridge/src/main.rs` lines 1–50 (imports, structs)

## Implementation Steps

1. **Cargo.toml** — add `mongodb = "2"`, `rand = "0.8"`, `dotenv = "0.15"`; change `tower-http` features to `["cors", "set-response-header"]`.

2. **`.env` file** — create at project root:
   ```
   MONGODB_URI=mongodb://localhost:27017
   JWT_SECRET=
   ALLOW_DEFAULT_CREDENTIALS=true
   ```

3. **`.gitignore`** — add `.env` entry (keep `.env.example` tracked).

4. **`security.rs` — User struct** — add `pub struct User { pub username: String, pub password_hash: String, pub role: String, pub enabled: bool, pub created_at: DateTime<Utc>, pub updated_at: DateTime<Utc> }` with serde derives.

5. **`security.rs` — MongoDB helpers** — add:
   - `pub async fn connect_db(uri: &str) -> Result<Database, mongodb::error::Error>`
   - `pub async fn ensure_indexes(collection: &Collection<User>) -> Result<()>`
   - `pub async fn seed_admin_user(collection: &Collection<User>) -> Result<bool>` — returns true if seeded
   - `pub async fn find_user(collection: &Collection<User>, username: &str) -> Result<Option<User>>`
   - `pub fn verify_password_blocking(candidate: &str, hash: &str) -> bool` — wraps bcrypt::verify

6. **`security.rs` — JWT module** — add `pub mod jwt` with:
   - `pub struct Claims { sub, role, iat, exp, jti }` — Serialize, Deserialize
   - `pub fn generate_token(username: &str, role: &str, secret: &str, ttl_secs: u64) -> Result<String>`
   - `pub fn validate_token(token: &str, secret: &str) -> Result<Claims>`

7. **`security.rs` — SessionRegistry** — `pub struct SessionRegistry` wrapping `Arc<Mutex<HashMap<String, Claims>>>` with methods: `register`, `is_valid`, `remove`, `sweep_expired`.

8. **`main.rs` — startup block** — before `setup_socketio`:
   - `dotenv::dotenv().ok()`
   - Connect to MongoDB, get collection
   - Ensure indexes
   - Seed admin if empty
   - Default credential guard (bcrypt verify "password" against admin hash)
   - Load/generate JWT_SECRET
   - Load SESSION_TTL_SECONDS

9. **`main.rs` — extend AuthCredentials** — add `pub token: Option<String>`.

10. **`main.rs` — replace auth comparison** — replace lines 264–271 with:
    - Token-first path: if `token` is `Some(t)` → `jwt::validate_token` → on success skip MongoDB
    - Password path: `find_user` → check `enabled` → `spawn_blocking(verify_password)` → `bcrypt::verify`
    - On failure: `socket.emit("auth_error", ...)` then `socket.disconnect()`
    - On success: `socket.emit("auth_token", token)`, `session_registry.register(socket_id, claims)`

11. **`main.rs` — `auth_refresh` handler** — `socket.on("auth_refresh", ...)` validates incoming JWT, emits fresh one.

12. **`main.rs` — per-command validation guard** — add `check_session_valid` calls to: `rover_command`, `arm_command`, `tracking_command`, `tts_command`, `fleet_select`, `audio_stream` handlers.

13. **`main.rs` — background sweep task** — `tokio::spawn` interval loop every 60s, sweeps expired sessions.

14. **`main.rs` — security headers** — wrap Axum router with `SetResponseHeaderLayer` for the four headers.

15. **`orchestra-dataflow.yml`** — remove `AUTH_USERNAME`, `AUTH_PASSWORD` env vars. Add `MONGODB_URI: ${MONGODB_URI}`.

16. **Tests** — add to `security.rs` `#[cfg(test)]`:
    - `test_bcrypt_verify`: hash a password, verify correct and wrong candidate.
    - `test_jwt_roundtrip`: generate token, validate, check `sub` and `role` match.
    - `test_jwt_expired`: generate token with `ttl_secs = 0`, validate should fail.
    - `test_session_registry_sweep`: register expired entry, sweep, verify removed.

## Todo

- [x] Add `mongodb`, `rand`, `dotenv` to `Cargo.toml`
- [x] Enable `tower-http` `set-response-header` feature
- [x] Create `.env` file with MONGODB_URI template
- [x] Add `.env` to `.gitignore`
- [x] Add `User` struct with serde derives to `security.rs`
- [x] Implement MongoDB connection + index helpers in `security.rs`
- [x] Implement `seed_admin_user` in `security.rs`
- [x] Implement `find_user` in `security.rs`
- [x] Implement `verify_password_blocking` bcrypt wrapper in `security.rs`
- [x] Implement JWT module (`Claims`, `generate_token`, `validate_token`) in `security.rs`
- [x] Implement `SessionRegistry` in `security.rs`
- [x] Add MongoDB connection + seed logic to `main.rs` startup
- [x] Add default credential guard to `main.rs` startup
- [x] Load/generate JWT_SECRET at startup in `main.rs`
- [x] Extend `AuthCredentials` with optional `token` field in `main.rs`
- [x] Replace plaintext comparison with MongoDB lookup + bcrypt in `main.rs`
- [x] Emit `auth_token` on successful auth in `main.rs`
- [x] Emit `auth_error` on auth failure in `main.rs`
- [x] Register session in `SessionRegistry` on auth success in `main.rs`
- [x] Add `auth_refresh` event handler in `main.rs`
- [x] Add per-command JWT validation guard to command handlers in `main.rs`
- [x] Spawn background sweep task in `main.rs`
- [x] Add security headers to Axum router in `main.rs`
- [x] Remove AUTH_USERNAME/AUTH_PASSWORD from `orchestra-dataflow.yml`
- [x] Add MONGODB_URI to `orchestra-dataflow.yml`
- [x] Write unit tests for bcrypt, JWT, session registry
- [x] Run `cargo build -p web_bridge` — zero errors
- [x] Run `cargo test -p web_bridge` — all tests pass

## Success Criteria

- `cargo build -p web_bridge` passes with no warnings about unused imports for bcrypt/jsonwebtoken.
- Server connects to MongoDB on startup and seeds admin user if collection empty.
- Server refuses to start with default admin password unless `ALLOW_DEFAULT_CREDENTIALS=true`.
- Successful socket handshake triggers MongoDB user lookup + bcrypt verify.
- `auth_token` event emitted with valid JWT on success.
- Token-based reconnect works without password re-entry.
- Command events (`rover_command`, `arm_command`, etc.) validate session before processing.
- `auth_error` event emitted before disconnect on failure.
- Response headers include `X-Frame-Options: DENY` on `/health`.
- `.env` is in `.gitignore`, credentials not in version control.
- All unit tests pass.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MongoDB connection failure on startup | Medium | High | Clear error message + retry with backoff; document MONGODB_URI setup |
| bcrypt blocking async runtime | Medium | High | Use `tokio::task::spawn_blocking` for all bcrypt calls |
| JWT secret lost on restart | Low | Medium | Document: sessions invalidated on restart; clients must re-auth |
| Breaking existing web UI | Medium | High | Token path is additive; username/password still works — no client change required for Phase 01 |
| MongoDB dependency increases binary size | Low | Low | MongoDB driver is well-maintained; binary size acceptable |
| `mongodb` crate version compatibility | Low | Medium | Pin to `mongodb = "2"`, test with CI |

## Security Considerations

- bcrypt cost 12 adds ~250ms per login — acceptable for infrequent auth; use `spawn_blocking` to avoid starving the event loop.
- JWT HS256 with 256-bit secret provides adequate security for single-server deployment.
- `jti` (JWT ID) included for future revocation support. Not enforced in this phase.
- Do not log JWT tokens. Log only `sub` and `exp`.
- MongoDB connection string may contain credentials — stored in `.env`, gitignored.
- `enabled` field on User allows disabling accounts without deletion.
- Default admin seed uses bcrypt — the plaintext "password" is never stored.

## Next Steps

- Phase 02 extends rate limiting and adds idle timeout, IP-based limiting, and origin HTTPS enforcement.
- Phase 03 updates the React client to consume `auth_token` / `auth_error` events.

## Completed

- **Date:** 2026-03-26
- **Score:** 9/10
- **Notes:** AccountDisabled mapped to same error string as InvalidCredentials to prevent enumeration. Token renewal now validates user.enabled via DB lookup. spawn_blocking panic logged explicitly.
