# Phase 1 — Fix and verify media controls

Context: [plan](plan.md) · Date: 2026-07-18 · Priority: high · Status: Completed

## Overview

Repair two confirmed UI regressions without changing the recorder or bridge wire protocol.

## Key insights

- `stream_control` and `camera_control` own independent browser media demands; Camera Off must first release its active stream demand.
- Playback tickets deliberately contain a relative bridge route, but the Vite app is a different origin.
- Clearing a failed ticket and incrementing the auto-request state in the same error callback creates the observed retry loop.

## Requirements

- Camera Off stops a live browser stream, then requests camera stop, and keeps detection shutdown behavior.
- Relative playback paths resolve to the configured Socket.IO/bridge origin and malformed/external paths are rejected.
- Playback media errors are stable and recover only through the explicit request button.

## Architecture

Pass the active bridge URL from `RoboRoverControl` to `MediaRecordingPage`, then to `RecordingPlaybackPanel`. Keep the backend ticket relative. Synchronize CameraViewer's stream state and emitted `stream_control: stop` before its existing `camera_control: stop`.

## Related code files

- `robo-control-app/packages/ui/src/components/features/CameraViewer.tsx`
- `robo-control-app/packages/ui/src/components/features/CameraViewer.test.tsx`
- `robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx`
- `robo-control-app/packages/ui/src/components/pages/media-recording-page.tsx`
- `robo-control-app/packages/ui/src/components/pages/media-recording-page.test.tsx`
- `robo-control-app/packages/ui/src/components/features/recording-playback-panel.tsx`
- `robo-control-app/packages/ui/src/components/features/recording-playback-panel.test.tsx`
- `robo-control-app/apps/web/src/recording-e2e-harness.tsx` if required by the prop contract.

## Implementation steps

1. Extract/reuse the stream-stop emission in `CameraViewer`; when Camera Off is clicked in legacy mode while streaming, stop the stream state/demand first, then emit camera stop.
2. Thread the configured bridge URL into the recording player. Resolve only a safe relative ticket URL against its HTTP(S) origin; render the resulting absolute URL.
3. Make playback error handling clear the unusable ticket but block the automatic request effect until selection changes or the user requests playback again.
4. Update focused tests for stop ordering/state, configured-origin URL resolution, rejected URL inputs, and no automatic retry after a video error.

## Todo

- [x] Implement camera-off stream synchronization.
- [x] Implement safe configured-origin playback resolution.
- [x] Stop automatic error retry.
- [x] Add and run regression tests.
- [x] Run typecheck and authenticated live smoke.

Validation: 28 focused tests, workspace typecheck, lint, authenticated playback range `206`, and Rover camera stop/restart smoke.

## Success criteria

- With a started stream, Camera Off emits `stream_control: stop` and `camera_control: stop`; the backend can transition the physical camera off.
- A `/recordings/playback/<ticket>` response becomes an MP4 URL under the configured bridge origin, not the Vite origin.
- A video error yields one stable error state and one user-controlled retry.
- Existing recording/session behavior still passes focused tests.

## Risk assessment

Low. Changes are local UI state/URL derivation. Primary risk is accepting an unsafe configured origin; constrain it to parsed HTTP(S) origins.

## Security considerations

Do not expose ticket tokens in logs. Do not permit protocol-relative or arbitrary ticket origins. Preserve authenticated ticket issuance and backend expiry/range enforcement.

## Next steps

Implement via `/code`, validate tests/typecheck/live smoke, then perform code review and request approval before finalization.
