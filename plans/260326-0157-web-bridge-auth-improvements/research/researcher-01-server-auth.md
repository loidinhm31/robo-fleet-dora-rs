# Server-Side Auth Analysis

## Current Auth Flow (`common/web_bridge/src/main.rs`)

1. **Credential Loading** (lines 237-244): `AUTH_USERNAME`/`AUTH_PASSWORD` env vars, defaults `admin/password`
2. **Socket.IO Handshake** (line 253): `TryData::<AuthCredentials>` extractor on connection
3. **Rate Limit Check** (lines 257-262): Before credential validation
4. **Plain Text Comparison** (line 267): `credentials.username == auth_username && credentials.password == auth_password`
5. **Disconnect on Failure** (lines 276-280): Socket disconnected, audit logged
6. **No Session Tokens**: Connection = permanent auth, no re-authentication

## Security Gaps

| Severity | Issue | Detail |
|----------|-------|--------|
| CRITICAL | Plaintext password comparison | Line 267, timing-attack vulnerable |
| CRITICAL | Default credentials | `admin/password`, only warns |
| CRITICAL | No password hashing | `bcrypt` in Cargo.toml but UNUSED |
| HIGH | No JWT sessions | `jsonwebtoken` in Cargo.toml but UNUSED |
| HIGH | No session expiry | Socket stays authenticated forever |
| MEDIUM | Incomplete rate limiting | `tracking_command`, `camera_control` exempt |
| MEDIUM | No security headers | Missing X-Frame-Options, HSTS, CSP |
| MEDIUM | Username in logs | Leaks valid usernames on failed auth |

## Rate Limiting (`security.rs`)

- **Auth**: 5/min per socket ID (configurable)
- **Commands**: 100/sec per socket ID (partial — some events exempt)
- **Gap**: Socket ID ≠ client IP behind reverse proxy

## CORS (lines 630-668)

- Specific origins with credentials enabled
- No HTTPS enforcement
- Permissive fallback if wildcard `*`

## Unused Dependencies (Cargo.toml)

- `bcrypt = "0.15"` — never imported
- `jsonwebtoken = "9.2"` — never imported
