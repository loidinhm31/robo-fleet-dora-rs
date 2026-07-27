# Elapsed time research

## Finding

The UI formats `status.duration_ms` directly. Active status duration stays zero
until finalization and identical statuses are intentionally suppressed by the
web bridge, so an active recording cannot advance on screen.

## Recommended design

Use a UI-local monotonic anchor: on an active status, render its
`duration_ms + (performance.now() - receivedAt)`. Re-anchor when recording ID,
state, or duration changes. Freeze the exact backend value for terminal states.

Do not derive elapsed from `started_at_ms`: it is a server wall clock and can
skew from the browser clock. Do not add periodic Dora or Socket status traffic.

## Evidence

- UI: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-session-control.tsx:31-35`.
- Recorder active update: `orchestra/media_recorder/src/session-manager.rs:402-413`.
- Terminal duration: `orchestra/media_recorder/src/session-manager.rs:584-587`.
- Bridge dedupe: `common/web_bridge/src/main.rs:3473-3488`.
- Reconnect cached status: `common/web_bridge/src/main.rs:779-780`.

## Required tests

- Fake timers advance active elapsed without a second status event.
- New status re-anchors the value.
- Completed/failed status freezes duration and clears the interval.
- Unmount cleans up the timer.

## Unresolved questions

- None. The existing status contract is sufficient.
