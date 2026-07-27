# Phase 03 — Dual-Source Voice UI

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phase 01](./phase-01-web-bridge-dual-source-transport.md)
- UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Voice controls: `packages/ui/src/components/features/VoiceControls.tsx`
- Transcript display: `packages/ui/src/components/features/TranscriptionDisplay.tsx`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-03 |
| Description | Activate browser voice transport and separate private browser from fleet rover transcription surfaces. |
| Priority | P1 |
| Implementation status | Complete |
| Review status | Approved |
| Effort | 8h |

## Key Insights

- Shared TypeScript contracts partly match Phase 01, including nullable confidence and new events.
- Voice worklet posts samples internally but does not forward messages to Socket.IO.
- Rover transcription display now tolerates null confidence, but source/status separation remains incomplete.
- `VoiceControls.tsx` is already large; capture and panel behavior should be extracted.

## Requirements

- Browser Voice Commands owns private browser capture and final history.
- Speech Transcription owns rover-origin history with entity labels.
- Global startup profile/status is display-only; no selector.
- Browser lifecycle emits exactly one start and stop per stream.
- All microphone, worklet, AudioContext, animation, and URL resources clean up on every exit.
- Walkie-talkie and manual TTS behavior remain unchanged.

## Architecture

```text
use-browser-voice-capture -> start -> 50 ms F32 frames -> stop
socket status -> both voice surfaces
private browser transcription -> Voice Controls
fleet rover transcription -> Transcription Display
```

## Related Code Files

- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/voice.ts`: finalized contracts/fixtures.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts`: typed event maps.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-browser-voice-capture.ts`: browser audio lifecycle.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/voice-command-panel.tsx`: focused private transcript UI.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx`: compose extracted behavior.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/TranscriptionDisplay.tsx`: rover labels/status/history.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx`: source-specific socket state/listeners.

## Implementation Steps

1. Verify Rust and TypeScript field names, enum values, nullability, and event signatures with shared fixtures.
2. Extract browser permission, AudioContext, media source, worklet, analyser, IDs, frame counter, and cleanup into one hook.
3. On explicit start, create a UUID, read actual AudioContext sample rate, emit start before audio, and capture current selected rover only for display.
4. Attach `workletNode.port.onmessage`; batch about 50 ms per frame and emit F32 audio with monotonically increasing frame IDs.
5. Increment frame ID only after the emit path accepts the frame locally; stop on invalid/disconnected state.
6. Emit one stop and release all browser resources on stop, mode switch, unmount, disconnect, or error.
7. Add page state/listeners for `stt_status`, `voice_command_transcription`, and rover `transcription`.
8. Keep browser and rover histories independent and bounded.
9. Show browser capture state, authoritative backend status/profile, target rover, level, and latest final text in Voice Controls.
10. Show rover entity badge, language/profile, final text, and optional confidence in Transcription Display.
11. Disable browser start unless connected, authenticated, backend ready, and a selected rover exists.
12. Preserve walkie-talkie/manual TTS paths and verify mode transitions clean up before starting another mode.
13. Add Vitest/Testing Library coverage for frame forwarding, start/stop once, teardown, status states, privacy/source separation, history bounds, and null confidence.
14. Run type-check, lint, tests, and production build in the external UI repository.

## Todo List

- [x] Finalize shared TypeScript contracts.
- [x] Extract browser capture hook.
- [x] Forward worklet frames to Socket.IO.
- [x] Implement exact start/stop lifecycle.
- [x] Add private browser transcript panel.
- [x] Add fleet rover labels/history.
- [x] Render authoritative global status/profile.
- [x] Add UI lifecycle/state tests.
- [x] Run full UI validation.

## Success Criteria

- Voice Commands produces bounded frames and receives private final text.
- Rover transcripts appear only in fleet transcription and show correct rover IDs.
- Missing confidence never renders `NaN%` or throws.
- Reconnect restores authoritative status without optimistic profile state.
- Stop/switch/unmount leaves no live media resource or unflushed stream.
- Walkie-talkie and manual TTS pass regression checks.

## Risk Assessment

- Browsers may ignore requested sample rate. Send actual rate and rely on central resampling.
- React listener duplication can leak events. Use stable handlers and explicit effect cleanup.
- UI repository is separate. Coordinate contract commits and deployment order.

## Security Considerations

- Start microphone only from explicit user action.
- Do not send authoritative target/entity fields from the browser.
- Render only sanitized backend status errors.

## Next Steps

Completed 2026-07-03 after UI tests, lint, type-check, and build passed and review approval cleared. Proceed to Phase 04 system validation gate.

## Unresolved Questions

None.
