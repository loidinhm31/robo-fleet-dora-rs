---
title: "Phase 03 — Client-Side JWT Integration"
phase: "03"
plan: "260326-0157-web-bridge-auth-improvements"
status: done
priority: P1
effort: ~1h
depends_on: ["phase-01-jwt-bcrypt-core"]
blocks: []
---

# Phase 03 — Client-Side JWT Integration

## Context Links

- Plan: [plan.md](plan.md)
- Depends on: [phase-01-jwt-bcrypt-core.md](phase-01-jwt-bcrypt-core.md)
- Server hardening (parallel): [phase-02-rate-limit-hardening.md](phase-02-rate-limit-hardening.md)

## Overview

| Field | Value |
|-------|-------|
| Date | 2026-03-26 |
| Description | Update React client to store JWT from server, send on reconnect, handle auth_error events, surface session status in UI |
| Priority | P1 |
| Implementation status | pending |
| Review status | pending |

## Key Insights

- `SocketAuth` is defined in two places: `ISocketService.ts` (the canonical interface) and `ServerSettings.tsx` (a local re-declaration). The local one in `ServerSettings.tsx` must be removed and replaced with an import to avoid drift.
- `SocketService` is a pure static facade — it delegates every call to `getSocketService()`. Token logic belongs in the underlying adapter, not the facade, but the facade's `connect()` signature must accept `token`.
- `useConnection.ts` currently hardcodes `{ username: string; password: string }` inline rather than importing `SocketAuth`. This should be fixed to use the shared type.
- Credentials stored in `localStorage` survive tab close, which is undesirable for JWT tokens (tokens should expire with the session). Store the JWT in `sessionStorage` instead. Credentials (username/password) can remain in `localStorage` since they are not secrets in the same sense — but add a comment noting the trade-off.
- The `auth_refresh` event is server-initiated re-issue; client can also proactively refresh by emitting `auth_refresh` with the current token 5 minutes before expiry. This requires decoding the JWT payload (base64 decode the middle segment — no library needed).
- `ServerSettings.tsx` is the auth UI. Adding a "Session active" badge is additive and does not break the existing popover layout.

## Requirements

1. **`SocketAuth` type update**
   - Add `token?: string` to the `SocketAuth` interface in `ISocketService.ts`.
   - Remove the duplicate `SocketAuth` type declaration in `ServerSettings.tsx`; import from the interface instead.
   - Update `useConnection.ts` to import and use `SocketAuth` from the interface.

2. **Token storage — in-memory primary, sessionStorage fallback (SEC-005)**
   - Primary: Store JWT in a module-scoped variable (`let currentToken: string | null = null`) inside the socket adapter. This is the SEC-005 requirement — no persistent browser storage for tokens.
   - Fallback: Also write to `sessionStorage` so page refresh doesn't force re-auth (UX trade-off). Clear sessionStorage on `auth_error` or tab close.
   - On `SocketService.connect()`: prefer in-memory token, fall back to `sessionStorage.getItem("robo_auth_token")`.
   - On `auth_error` with `reason === "token_expired"` or `reason === "invalid_credentials"`: clear both in-memory and sessionStorage.

3. **Auto-reconnect with token**
   - The Socket.IO client library handles reconnect automatically. The reconnect auth payload is set at connection construction time.
   - Implement a `reconnectAuth()` helper that reads the token from `sessionStorage` and returns a `SocketAuth` with `token` set (and without `username`/`password` — server accepts token-only auth).
   - Pass this helper result as the `auth` option when constructing the socket instance.

4. **Proactive token refresh**
   - After storing a new token, decode the `exp` claim from the JWT payload (base64-decode the second `.`-delimited segment, parse JSON, read `exp`).
   - Schedule `setTimeout` to emit `auth_refresh` 5 minutes before `exp` (at ~55 min for 1h TTL). Clear any existing timer on each new token receipt.
   - Listen for `auth_token` response and store the refreshed token, rescheduling the timer.

5. **`auth_error` handling**
   - In the socket adapter, listen for `auth_error` event.
   - Map `reason` to user-facing messages:
     - `"invalid_credentials"` → "Authentication failed. Check username and password."
     - `"token_expired"` → "Session expired. Please reconnect."
     - `"rate_limited"` → "Too many attempts. Please wait."
     - `"idle_timeout"` → "Disconnected due to inactivity."
   - Surface via the existing connection status callback: emit a new `ConnectionStatus` of `"error"` and pass the message as an additional argument, OR add an `authError?: string` field to the status change callback.
   - On `token_expired` / `idle_timeout`: clear token from sessionStorage.

6. **`ServerSettings` UI updates**
   - Accept optional `sessionActive: boolean` and `sessionExpiresAt?: Date` props.
   - When `sessionActive` is true, show a small "Session active" badge (green dot + text) in the connection status section of the popover.
   - Show auth error inline in the popover when present (receive as `authError?: string` prop).
   - Update the comment on line 204 from `// persisted in localStorage` to `// credentials: localStorage | token: sessionStorage`.

## Architecture

```
ISocketService.ts
  └── SocketAuth { username: string; password: string; token?: string }

SocketService.ts (facade)
  └── connect(url, auth?) — unchanged API; implementation reads sessionStorage token

socket adapter (underlying implementation)
  ├── on connect: register auth_token listener → store token + schedule refresh
  ├── on auth_error: → clear token if expired, emit status change with error message
  ├── reconnect auth: read token from sessionStorage, inject into socket auth option
  └── proactive refresh timer: setTimeout(emit auth_refresh, exp - 5min - now)

useConnection.ts
  ├── import SocketAuth from interfaces
  ├── connect(url, auth?: SocketAuth) — type corrected
  └── expose authError?: string from status change callback

ServerSettings.tsx
  ├── remove local SocketAuth re-declaration
  ├── import SocketAuth from interfaces
  ├── accept sessionActive, sessionExpiresAt, authError props
  └── render session badge + error message
```

## Related Code Files

**Owned by this phase** (in `G:/ws/sharing/qm-sync/embed-app/robo-control-app`):
- `packages/ui/src/adapters/factory/interfaces/ISocketService.ts` — add `token?` to `SocketAuth`
- `packages/ui/src/services/SocketService.ts` — no API change; may need internal token-read logic if adapter is not abstracted
- `packages/ui/src/hooks/useConnection.ts` — import `SocketAuth`, expose `authError`
- `packages/ui/src/components/organisms/ServerSettings.tsx` — remove local `SocketAuth`, add session badge + error display
- `packages/shared/src/types/socket.ts` — add `AuthErrorEvent { reason: AuthErrorReason }` and `AuthTokenEvent { token: string }` if not already present

**Read-only reference:**
- `packages/ui/src/adapters/factory/` — to find the concrete socket adapter implementation that `getSocketService()` returns

## Implementation Steps

1. **`ISocketService.ts`** — add `token?: string` to `SocketAuth`. No other changes.

2. **`packages/shared/src/types/socket.ts`** — add:
   ```ts
   export type AuthErrorReason =
     | "invalid_credentials"
     | "token_expired"
     | "rate_limited"
     | "idle_timeout";

   export interface AuthErrorEvent {
     reason: AuthErrorReason;
   }
   ```

3. **Find concrete adapter** — locate the file returned by `getSocketService()` in `packages/ui/src/adapters/factory/`. Read it to understand where `socket.on(...)` listeners are registered and where the socket instance is constructed.

4. **Concrete adapter — `auth_token` listener** — register `socket.on("auth_token", (token: string) => { sessionStorage.setItem("robo_auth_token", token); scheduleRefresh(token); })` after socket construction.

5. **Concrete adapter — `auth_error` listener** — register `socket.on("auth_error", (event: AuthErrorEvent) => { if (event.reason === "token_expired" || event.reason === "idle_timeout") sessionStorage.removeItem("robo_auth_token"); notifyStatusChange("error", errorMessage(event.reason)); })`.

6. **Concrete adapter — token in connect** — in the `connect(url, auth?)` method, before constructing the socket: `const token = sessionStorage.getItem("robo_auth_token"); const fullAuth = token ? { ...auth, token } : auth;`. Use `fullAuth` as the socket `auth` option.

7. **Concrete adapter — proactive refresh** — add `scheduleRefresh(token: string)`: decode JWT payload (second segment, base64url decode, JSON.parse), compute `refreshAt = exp * 1000 - Date.now() - 5 * 60 * 1000`, call `clearTimeout(refreshTimer)` and `refreshTimer = setTimeout(() => socket.emit("auth_refresh", { token }), Math.max(0, refreshAt))`.

8. **`useConnection.ts`** — import `SocketAuth` from interfaces. Change the inline type in `connect` callback to `SocketAuth`. Add `authError: string | null` state; subscribe to error status changes to populate it.

9. **`ServerSettings.tsx`** — remove the local `SocketAuth` interface (lines 6–9). Add import from interfaces. Add `sessionActive?: boolean`, `sessionExpiresAt?: Date`, `authError?: string` to `ServerSettingsProps`. In the JSX: render a session badge when `sessionActive` is true (small green dot + "Session active" text). Render `authError` as a red inline message below the auth inputs when present. Update the localStorage comment.

10. **`packages/shared` types barrel** — ensure `AuthErrorEvent` and `AuthErrorReason` are exported from the package's `index.ts` if other consumers need them.

11. **Type check** — run `pnpm check-types` from `robo-control-app/` — zero errors.

## Todo

- [x] Add `token?: string` to `SocketAuth` in `ISocketService.ts`
- [x] Add `AuthErrorReason`, `AuthErrorEvent` to `packages/shared/src/types/socket.ts`
- [x] Locate concrete socket adapter file
- [x] Add `auth_token` listener with sessionStorage write and refresh scheduling in adapter
- [x] Add `auth_error` listener with token cleanup and status notification in adapter
- [x] Inject sessionStorage token into connect auth payload in adapter
- [x] Implement `scheduleRefresh` for proactive token renewal in adapter
- [x] Update `useConnection.ts` to import `SocketAuth`, expose `authError`
- [x] Remove duplicate `SocketAuth` from `ServerSettings.tsx`
- [x] Add `sessionActive`, `sessionExpiresAt`, `authError` props to `ServerSettings.tsx`
- [x] Render session badge in `ServerSettings.tsx`
- [x] Render auth error inline in `ServerSettings.tsx`
- [x] Update localStorage comment in `ServerSettings.tsx`
- [x] Export new shared types from package barrel
- [x] Run `pnpm check-types` — zero errors
- [x] Manual test: connect with credentials, verify token stored in sessionStorage
- [x] Manual test: refresh page, reconnect — token used automatically
- [x] Manual test: server restart (token invalidated) — client falls back to credential re-entry

## Success Criteria

- `pnpm check-types` passes with zero errors.
- `SocketAuth` has `token?: string` and is not duplicated across files.
- After successful connection, `sessionStorage.getItem("robo_auth_token")` is non-null.
- On reconnect (e.g., network blip), the token is sent without the user re-entering credentials.
- `auth_error` events update the connection status to `"error"` with a human-readable message.
- `ServerSettings` popover shows "Session active" badge when token is valid.
- Expired token clears from sessionStorage and prompts the user to re-enter credentials.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Concrete adapter not found in expected location | Low | Medium | Read `getSocketService()` implementation to trace factory; fall back to modifying `SocketService.ts` facade directly |
| JWT decode fails for malformed token | Low | Low | Wrap in try/catch; on error clear token and skip refresh scheduling |
| SessionStorage unavailable (private browsing edge case) | Low | Low | Wrap sessionStorage access in try/catch; fall back to in-memory variable |
| Phase 01 not complete when Phase 03 runs | Low | High | Phase 03 depends on Phase 01; do not start until `auth_token` / `auth_error` events are confirmed in server |
| `ServerSettings` prop API change breaks parent components | Medium | Medium | Props are additive (`?` optional) — existing callers need no changes; no breaking change |

## Security Considerations

- `sessionStorage` is cleared on tab close, unlike `localStorage`. This is intentional — reduces the window a stolen token is valid.
- JWT payload decode (step 7) is not signature verification — it is only used for scheduling. The server is the authoritative validator; client-side exp reading is an optimistic UX convenience.
- Do not store the raw `password` alongside the token in sessionStorage. Credentials remain in React state only during the popover session.
- `auth_error { reason: "invalid_credentials" }` must not reveal whether the username or password was wrong (server sends the same reason for both).

## Next Steps

All three phases complete the planned improvements. Future work (not in scope):
- Multi-factor authentication
- JWT revocation list (Redis-backed `jti` blocklist)
- Audit log export endpoint
- Role-based authorization (read-only vs. control access)
