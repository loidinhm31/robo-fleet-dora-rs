---
title: "Phase 02 — Rate Limiting & Hardening (Server-Side)"
phase: "02"
plan: "260326-0157-web-bridge-auth-improvements"
status: done
priority: P1
effort: ~2h
depends_on: ["phase-01-jwt-bcrypt-core"]
blocks: []
---

# Phase 02 — Rate Limiting & Hardening (Server-Side)

## Context Links

- Plan: [plan.md](plan.md)
- Previous phase: [phase-01-jwt-bcrypt-core.md](phase-01-jwt-bcrypt-core.md)
- Client integration: [phase-03-client-jwt-integration.md](phase-03-client-jwt-integration.md)

## Overview

| Field | Value |
|-------|-------|
| Date | 2026-03-26 |
| Description | Extend command rate limiting to all events, add idle timeout, per-IP rate limiting, sanitize auth logs, enforce HTTPS origins in production |
| Priority | P1 |
| Implementation status | done |
| Review status | approved 2026-03-26 |

## Key Insights

- `CommandRateLimiter` in `security.rs` (lines 50–85) is keyed by `socket_id` string. Extending per-IP requires extracting a real IP key from the Socket.IO handshake. Socket.IO in `socketioxide` exposes the underlying request via `socket.req_parts()` — headers are accessible there.
- The existing command rate limiter is called inside each event handler individually. Events `tracking_command`, `camera_control`, `audio_control`, and `voice_command_audio` do not call it. The fix is mechanical — add the same guard to each missing handler.
- Idle timeout is straightforward: store `Arc<Mutex<Instant>>` per socket (or in `ClientState` which already exists), update on each event, and sweep in a background task.
- `ClientState` already tracks `last_video_sent` / `last_audio_sent` as `Arc<Mutex<SystemTime>>` — add a single `last_activity: Arc<Mutex<Instant>>` field.
- Auth logging in `log_auth_attempt` (security.rs line 172) logs `username` on failure. The fix is to remove it from the failure branch.
- CORS origin HTTPS validation is a startup check in `parse_allowed_origins()` or at the point of constructing the CORS layer.

## Requirements

1. **Extend command rate limiting to all events**
   - Add `check_command` calls to: `tracking_command`, `camera_control`, `audio_control`, `voice_command_audio` (and any other handlers found without it).
   - Use the existing `shared_state.command_rate_limiter.check_command(&socket_id)` pattern already present in `arm_command` and `rover_command` handlers.

2. **Inactivity timeout**
   - Add `last_activity: Arc<Mutex<Instant>>` to `ClientState`.
   - Update `last_activity` at the top of every event handler.
   - `IDLE_TIMEOUT_SECONDS` env var, default `1800` (30 min).
   - Background task (spawned in `setup_socketio` or in the main tokio spawn block) runs every 60 s, iterates `shared_state.video_clients`, and disconnects sockets idle beyond the threshold.
   - On disconnect due to idle: emit `auth_error { reason: "idle_timeout" }` before disconnect.

3. **Auth log sanitization**
   - In `security::log_auth_attempt`: remove `username` from the failure log line. Log only `client_id`, `timestamp` (implicit via tracing), and a generic "Authentication failed" message.
   - On success, keep logging `username` (confirms which account authenticated).

4. **Default credential enforcement at startup**
   - If `AUTH_USERNAME == "admin"` and `AUTH_PASSWORD == "password"` (raw, before hashing) and `ALLOW_DEFAULT_CREDENTIALS != "true"` → `tracing::error!("FATAL: default credentials in use. Set AUTH_USERNAME and AUTH_PASSWORD or set ALLOW_DEFAULT_CREDENTIALS=true")` + `std::process::exit(1)`.
   - Note: this logic may already be added in Phase 01. If so, this item is a verification step only.

5. **Per-IP rate limiting**
   - Add `IpRateLimiter` struct to `security.rs` with the same `governor`-based pattern as `AuthRateLimiter`, keyed by IP string.
   - Extract client IP in the namespace handler: check `X-Forwarded-For` header first (when `TRUST_PROXY_HEADERS=true`), then `X-Real-IP`, then fall back to socket peer address (socketioxide provides `socket.req_parts().extensions` or headers).
   - Apply `IpRateLimiter` for auth attempts (in addition to the existing socket-ID limiter).
   - `TRUST_PROXY_HEADERS` env var, default `false`.
   - `RATE_LIMIT_AUTH_PER_MINUTE_IP` env var for the per-IP auth limit, default `20`.

6. **Origin HTTPS validation**
   - In `parse_allowed_origins()` (or at CORS construction site in main): if `ALLOW_HTTP_ORIGINS != "true"`, warn on any origin that starts with `http://` but is not `localhost` or `127.0.0.1`.
   - Do not block HTTP origins — only warn. This avoids breaking dev setups while alerting on production misconfiguration.

## Architecture

```
security.rs additions
  ├── IpRateLimiter { limiters: HashMap<String, (limiter, Instant)>, max_attempts: u32 }
  │   └── check_auth_attempt_ip(ip: &str) -> bool
  ├── log_auth_attempt (modified) — remove username from failure log
  └── extract_client_ip(headers: &HeaderMap, trust_proxy: bool) -> String

main.rs modifications
  ├── ClientState
  │   └── add last_activity: Arc<Mutex<Instant>>
  ├── setup_socketio
  │   ├── load IDLE_TIMEOUT_SECONDS
  │   ├── load TRUST_PROXY_HEADERS
  │   ├── construct IpRateLimiter
  │   ├── namespace handler
  │   │   ├── extract IP → check IpRateLimiter
  │   │   └── update last_activity on each event handler (all events)
  │   └── spawn idle-sweep background task (tokio::spawn interval loop)
  └── tracking_command, camera_control, audio_control, voice_command_audio
      └── add check_command rate limit guard (currently missing)

parse_allowed_origins (security.rs)
  └── warn on http:// origins that are not localhost
```

## Related Code Files

**Owned by this phase:**
- `common/web_bridge/src/security.rs` — `IpRateLimiter`, `extract_client_ip`, modify `log_auth_attempt`
- `common/web_bridge/src/main.rs` — `ClientState`, all event handlers, idle-sweep task, CORS origin warn

**Read-only reference:**
- Phase 01 additions to `main.rs` and `security.rs` (already applied before this phase runs)

## Implementation Steps

1. **`security.rs` — sanitize `log_auth_attempt`** — remove `username = username` structured field from the failure branch. Keep it in the success branch.

2. **`security.rs` — `extract_client_ip`** — add `pub fn extract_client_ip(headers: &axum::http::HeaderMap, trust_proxy: bool) -> String`. When `trust_proxy` is true, parse `X-Forwarded-For` (first IP in comma-separated list) then `X-Real-IP`. Fall back to `"unknown"`.

3. **`security.rs` — `IpRateLimiter`** — mirror the `AuthRateLimiter` struct but use `RATE_LIMIT_AUTH_PER_MINUTE_IP` env var (default 20) and expose `check_auth_attempt_ip(ip: &str) -> bool`.

4. **`security.rs` — HTTP origin warning** — add `pub fn warn_http_origins(origins: &[String])` that logs a warning for any non-localhost HTTP origin when `ALLOW_HTTP_ORIGINS != "true"`.

5. **`main.rs` — `ClientState`** — add `last_activity: Arc<Mutex<Instant>>` field; initialise to `Instant::now()` in `ClientState::new`.

6. **`main.rs` — update last_activity helper** — add `fn touch_activity(clients: &Mutex<Vec<ClientState>>, socket_id: &str)` that finds the matching client and updates `last_activity`.

7. **`main.rs` — namespace handler** — after loading `TRUST_PROXY_HEADERS`, construct `IpRateLimiter`; in the connection handler extract IP using `socket.req_parts()` headers and check against `IpRateLimiter` before credential validation.

8. **`main.rs` — add rate limit to missing handlers** — scan all `socket.on(...)` registrations for `tracking_command`, `camera_control`, `audio_control`, `voice_command_audio` and add `check_command` guard at the top of each, matching the existing pattern in `arm_command`.

9. **`main.rs` — touch activity in every handler** — at the top of every `socket.on(...)` closure (after rate limit check), call `touch_activity`.

10. **`main.rs` — idle sweep task** — spawn a `tokio::spawn(async move { loop { tokio::time::sleep(Duration::from_secs(60)).await; ... } })` that locks `video_clients`, checks `last_activity`, collects expired `socket_id`s, emits `auth_error { reason: "idle_timeout" }` and calls `io.get_socket(id).map(|s| s.disconnect())`.

11. **`main.rs` — CORS origin warn** — call `security::warn_http_origins(&allowed_origins)` after `parse_allowed_origins()`.

12. **Tests** — add to `security.rs` `#[cfg(test)]`:
    - `test_extract_ip_forwarded_for`: headers with `X-Forwarded-For: 1.2.3.4, 5.6.7.8` → returns `"1.2.3.4"` when trust_proxy true.
    - `test_extract_ip_no_proxy`: returns `"unknown"` when trust_proxy false and no peer addr.
    - `test_ip_rate_limiter_blocks`: exceed configured limit, verify `false` returned.
    - `test_warn_http_origins_no_panic`: confirm function runs without panic for mixed origin list.

## Todo

- [ ] Remove username from `log_auth_attempt` failure branch in `security.rs`
- [ ] Implement `extract_client_ip` in `security.rs`
- [ ] Implement `IpRateLimiter` in `security.rs`
- [ ] Implement `warn_http_origins` in `security.rs`
- [ ] Add `last_activity` field to `ClientState` in `main.rs`
- [ ] Add `touch_activity` helper in `main.rs`
- [ ] Integrate `IpRateLimiter` in namespace connection handler in `main.rs`
- [ ] Add `check_command` rate limit to `tracking_command` handler in `main.rs`
- [ ] Add `check_command` rate limit to `camera_control` handler in `main.rs`
- [ ] Add `check_command` rate limit to `audio_control` handler in `main.rs`
- [ ] Add `check_command` rate limit to `voice_command_audio` handler in `main.rs`
- [ ] Call `touch_activity` in every event handler in `main.rs`
- [ ] Spawn idle-sweep background task in `main.rs`
- [ ] Call `warn_http_origins` after `parse_allowed_origins` in `main.rs`
- [ ] Verify default-credential guard (from Phase 01) is in place
- [ ] Write unit tests for IP extraction, IP rate limiter
- [ ] Run `cargo build -p web_bridge` — zero errors
- [ ] Run `cargo test -p web_bridge` — all tests pass

## Success Criteria

- `cargo build -p web_bridge` clean.
- `tracking_command`, `camera_control`, `audio_control`, `voice_command_audio` all have `check_command` guard — verifiable by code grep.
- Failed auth log lines do not contain the attempted username — verifiable by inspecting `log_auth_attempt`.
- Starting with `ALLOWED_ORIGINS=http://example.com` logs a warning about non-HTTPS origin.
- Starting with `TRUST_PROXY_HEADERS=true` and a socket connection with `X-Forwarded-For: 10.0.0.1` uses `10.0.0.1` as the rate-limit key.
- Idle sweep task is spawned (log line confirms).
- All unit tests pass.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| socketioxide `req_parts()` API differs from assumed | Medium | Low | If headers not accessible in namespace handler, skip per-IP limiter; implement as connection-layer middleware instead |
| Idle sweep disconnecting legitimate slow operators | Low | Medium | Default 30 min is generous for robotics ops; document `IDLE_TIMEOUT_SECONDS` env var clearly |
| `touch_activity` lock contention | Low | Low | `Mutex<Instant>` write is nanoseconds; not a hot path issue |
| Forgetting a handler when adding `check_command` | Medium | Medium | Add a `#[cfg(test)]` compile-time list assertion or document every handler name in this file |

## Security Considerations

- Per-IP rate limiting prevents distributed brute-force across multiple browser tabs/reconnects that share one real IP.
- `TRUST_PROXY_HEADERS=false` default prevents IP spoofing via forged headers when the server is directly internet-facing.
- Removing username from failure logs eliminates username enumeration via log exfiltration.
- Idle timeout mitigates session hijacking: if a JWT is stolen but the legitimate user disconnects, the stolen session eventually expires even without token revocation.

## Next Steps

- Phase 03 updates the React client to handle `auth_token`, `auth_error`, and `auth_refresh` events.
