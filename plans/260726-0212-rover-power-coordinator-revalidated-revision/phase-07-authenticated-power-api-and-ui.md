# Phase 07 — Authenticated Power API and UI

## Context links

- Parent: [plan.md](./plan.md); design: [UI and history](../../docs/power-coordinator-architecture.md#ui-and-local-voice-wake).
- External UI: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.
- Evidence: [voice/UI research](./research/researcher-02-voice-ui-history-revision.md).
- Dependencies: Phases 01–05.

## Overview

- Date: 2026-07-26; priority: P1; implementation/review: Pending.
- Expose exact-target authenticated policy/Wake controls, live authority status, and bounded 90-day history across shared web/Tauri UI.

## Key Insights

- Existing lifecycle socket provides the reusable authentication, pending-capacity, exact target, rate, audit actor, reconnect, and authoritative terminal-status patterns.
- Any authenticated user may mutate persistent policy, but client input is never the authority boundary.

## Requirements

- Add versioned power Socket.IO commands/results/status/transition/history; derive actor and selected entity server-side.
- `power_wake` atomically turns Sleep to Auto and acquires/renews a 120-second UI demand. Disconnect, target change, expiry, and server sweep release it.
- Reject stale/disconnected authority, expired/changed duplicate/cross-entity/rate-limited inputs; no optimistic Ready.
- Live `(epoch,sequence)` outranks historical Mongo/current projection. Cold state is visibly Historical/Stale and cannot regress live state.

## Architecture

UI → `common/web_bridge` → local Orchestra/direct Rover coordinator. `power_current_state` and paginated `power_lifecycle_events` serve history only; live coordinator snapshot/status is authoritative.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/{power-socket.rs,power-history-gateway.rs,power-queues.rs}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/{main.rs,security.rs,lifecycle-socket.rs}` and relevant Orchestra/direct dataflows.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/{power.ts,power-fixtures.ts,power-v1.json}`; modify its type index.
- Create UI `power-state` reducer/hook/card/history files; modify `RoboRoverControl.tsx` and tests.

## Implementation Steps

1. Mirror golden contracts in TypeScript and test Rust/TS fixture compatibility.
2. Add authenticated handlers, separate policy limiter, exact entity pinning, bounded queues, request expiry, and audit event results.
3. Implement Wake as one request/transition ID with 120-second activity-renew demand and server cleanup.
4. Broadcast validated monotonic live status; clear stale local state on disconnect/auth/selection change and request snapshot on reconnect.
5. Add server-enforced history filters/cursor/90-day bound and UI reducer with live-over-cold/event-ID dedupe.
6. Add Fleet Resources card and accessible history panel for policy, requested/effective profile, AuthorityUnknown, demand, freshness, transition failure, and occurrence power state.

## Todo list

- [ ] Add backend Socket.IO/history contracts and limiter.
- [ ] Add shared fixture and normalized store.
- [ ] Add controls/history views and auth/reconnect tests.

## Success Criteria

- Any authenticated session can set policy only for its server-pinned active rover, with rate/audit enforcement.
- Wake from Sleep becomes Auto plus a two-minute demand; it never shows Ready before authority confirms it.
- Historical data cannot overwrite live effective state.

## Risk Assessment

- Broad signed-in policy rights increase availability impact; audit and separate limiter are non-optional.
- TTL leaks can hold profiles awake; cleanup/sweep tests are required.

## Security Considerations

- Do not expose session token, raw audio, paths, or native errors in history. Validate authorization at web bridge and coordinator.

## Next steps

Phase 08 validates browser, distributed, and target hardware gates before enabling flags.
