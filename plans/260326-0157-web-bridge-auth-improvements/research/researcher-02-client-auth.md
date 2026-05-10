# Client-Side Auth Analysis

## Files (robo-control-app)

- `packages/ui/src/services/SocketService.ts` — static facade wrapping socket factory
- `packages/ui/src/hooks/useConnection.ts` — React hook for connection state
- `packages/ui/src/adapters/factory/interfaces/ISocketService.ts` — `SocketAuth { username, password }` type
- `packages/ui/src/components/organisms/ServerSettings.tsx` — auth UI popover
- `packages/shared/src/types/socket.ts` — event type definitions (no auth types)

## Auth Flow

1. User enters URL + optional username/password in `ServerSettings` popover
2. `buildAuth()` trims and returns `SocketAuth | undefined`
3. `SocketService.connect(url, auth)` passes auth to socket factory
4. Auth sent as Socket.IO handshake `auth` payload (plaintext JSON)
5. Credentials persisted in localStorage (noted in UI comment line 204)

## Key Observations

- **No token storage**: No JWT/session token handling anywhere
- **No refresh logic**: Connection is fire-and-forget auth
- **No error feedback**: Failed auth = disconnect, no specific error message to user
- **Credentials in memory**: Stored in React state during popover session
- **Optional auth**: Client allows connecting without credentials (line 43: returns undefined if empty)

## Changes Needed for JWT Auth

1. `ISocketService` and `SocketAuth` types need `token?: string` field
2. `SocketService.connect()` needs to accept token-based auth
3. `useConnection` hook needs token refresh callback
4. `ServerSettings` needs token expiry indicator
5. New `useAuth` hook or context for token lifecycle management
6. Error handling for auth_error/token_expired server events
