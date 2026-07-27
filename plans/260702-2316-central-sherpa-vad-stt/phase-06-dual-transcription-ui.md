# Phase 06 — Dual Transcription UI

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phase 01](./phase-01-architecture-contracts-baseline.md), [Phase 04](./phase-04-web-bridge-dual-source-transport.md)
- UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Current Voice Controls: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx`
- Current rover transcript panel: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/TranscriptionDisplay.tsx`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-02 |
| Description | Activate browser voice capture and render browser/rover transcripts in their correct UI surfaces. |
| Priority | P1 |
| Implementation status | Pending |
| Review status | Pending |
| Effort | 8h |

## Key Insights

- Voice Commands worklet posts 4096-sample messages but never installs `port.onmessage`, so no audio leaves the browser.
- Speech Transcription already has history but assumes numeric confidence and lacks rover labels.
- `VoiceControls.tsx` is already over the repository's preferred size; add behavior through extracted hook/component modules.
- Browser status combines local capture state and global STT backend state; these must remain distinct.

## Requirements

- Voice Commands owns browser capture and browser-origin transcript history.
- Speech Transcription owns rover-origin history from all active rovers.
- Show global profile/status in both contexts without a profile selector.
- Omit confidence UI when missing; never render `NaN%`.
- Preserve walkie-talkie and manual TTS behavior.
- Cleanly stop/flush browser stream on unmount, disconnect, or mode change.

## Architecture

```text
useBrowserVoiceCapture
  -> control:start -> 50 ms F32 frames -> control:stop
  -> VoiceCommandPanel local state/audio level

RoboRoverControl socket listeners
  -> stt_status
  -> voice_command_transcription -> VoiceControls
  -> transcription -> TranscriptionDisplay
```

## Related Code Files

- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/{voice,socket}.ts`: contracts.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx`: listeners and state split.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx`: composition.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/TranscriptionDisplay.tsx`: rover labels/status/confidence.
- Create focused browser capture hook and Voice Command panel modules under `packages/ui/src/` following existing UI naming conventions.

## Implementation Steps

1. Update shared contracts exactly as Phase 01 specifies, including typed client/server event maps.
2. Extract microphone permission, AudioContext, worklet creation, analyser, stream/frame IDs, and cleanup into `useBrowserVoiceCapture`.
3. At start, create UUID stream ID, reset frame counter, read actual `audioContext.sampleRate`, require mono, and emit `voice_command_control:start` before frames.
4. Size worklet buffer to approximately 50 ms using actual sample rate rather than fixed 4096 samples.
5. Attach `workletNode.port.onmessage`; emit `voice_command_audio` with stream/frame/rate/channels/sample count/F32 data.
6. Increment frame ID only after emit call. Surface capture errors and stop the stream on invalid state.
7. On stop/unmount/disconnect, disconnect worklet/source, stop tracks, cancel animation frame, close AudioContext, emit one `voice_command_control:stop`, and revoke Blob URL.
8. Keep walkie-talkie state separate so switching modes performs full cleanup before starting the other mode.
9. Add `sttStatus`, `browserTranscription`, and `roverTranscription` page state and Socket.IO listeners.
10. Refactor Voice Commands into a focused panel showing capture state, backend status/profile, target rover, audio level, latest final text, and bounded five-item browser history.
11. Pass only browser transcription to Voice Controls; it is already server-filtered to the owner.
12. Pass only rover transcription to `TranscriptionDisplay`; store a bounded history of all rover results and render an `entity_id` badge per item.
13. Replace confidence helper signatures with optional handling. Render no badge when absent.
14. Display `loading`, `ready`, and sanitized `error` states. Disable browser Start unless socket connected, STT ready, and a selected rover exists.
15. Avoid optimistic STT backend state; Socket.IO `stt_status` is authoritative.
16. Add Vitest/Testing Library coverage for worklet frames, start/stop once semantics, unmount cleanup, status states, browser/rover separation, source labels, five-item history, and null confidence.
17. Update UI architecture and component documentation if implementation file boundaries differ from Phase 01 design.

## Todo List

- [ ] Update shared TS contracts.
- [ ] Extract browser capture hook.
- [ ] Add functional worklet message forwarding.
- [ ] Add explicit start/stop lifecycle.
- [ ] Add browser transcript panel/history.
- [ ] Add rover labels/history.
- [ ] Add STT status/profile display.
- [ ] Remove numeric-confidence assumptions.
- [ ] Add UI tests.

## Success Criteria

- Browser microphone produces frames while Voice Commands is active.
- Browser final transcript appears only in Voice Commands.
- Rover final transcripts appear only in Speech Transcription and show correct rover IDs.
- Status reconnect and error rendering are authoritative and actionable.
- No confidence produces no percentage or exception.
- Switching/stopping/unmounting leaves no live track, worklet, AudioContext, Blob URL, or unflushed stream.
- Walkie-talkie and manual TTS remain functional.

## Risk Assessment

- Risk: Browser ignores requested 16 kHz AudioContext. Mitigation: send actual rate and resample centrally.
- Risk: React socket listener closure duplication. Mitigation: stable handlers and explicit cleanup in effect return.
- Risk: Dynamic worklet URL leak. Mitigation: retain and revoke URL after module load/cleanup.

## Security Considerations

- Browser never supplies authoritative target/entity identity.
- Do not render raw backend/model errors; use sanitized status.
- Microphone starts only from explicit user action and always stops on component teardown.

## Next Steps

Proceed to Phase 07 after UI unit tests and production build succeed against the finalized Socket.IO contracts.
