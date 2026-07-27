# Phase 04 — Bounded Web Audio Timeline Scheduler

## Context links

- [Parent plan](./plan.md)
- [Phase 03](./phase-03-binary-browser-audio-cutover.md)
- [Architecture invariants](../../ARCHITECTURE.md#binary-browser-audio-and-bounded-playback)

## Overview

- Date: 2026-06-28
- Description: schedule validated frames on arrival with bounded latency and explicit recovery.
- Priority: P1
- Implementation status: Done (2026-06-30)
- Review status: Approved (2026-06-30, 9.5/10)
- Effort: 7h

## Key Insights

- Recursive 40 ms dequeue for 50 ms frames is deterministically unstable.
- Web Audio sources can be pre-scheduled; continuity tolerance equals scheduled-ahead horizon.
- Burst delivery must not grow latency without bound.
- AudioContext lifecycle and one-shot source cleanup require ownership outside CameraViewer rendering.

## Requirements

- No `setTimeout`/`setInterval` controls per-buffer playback.
- Default leads: minimum 10 ms, restart/target 50 ms, maximum scheduled end 150 ms.
- Schedule one validated frame immediately on arrival.
- Drop duplicate/regressed, malformed, too-old, or max-horizon-overflow frames with reason counters.
- If next start is inside minimum lead, count late/underrun and restart at target lead.
- Track scheduled sources; remove on end and stop/cancel on disable, reconnect, stream change, or unmount.
- Reset sequence/timeline on new `stream_id`; do not treat it as regression.
- Preserve gain and low-pass chain.
- Resume AudioContext only from user-triggered stream/audio enable path; expose suspended state.
- UI consumes a throttled metrics snapshot, never frame-rate React updates.

## Architecture

```text
Socket event -> validate/decode -> sequence policy -> horizon policy
  -> AudioBufferSourceNode.start(context timestamp) -> onended cleanup
```

Scheduling policy:

1. First/restart frame starts at `currentTime + 0.050`.
2. Normal frame starts at previous scheduled end.
3. If start lead is `<0.010`, restart at `currentTime + 0.050`; count underrun/reset.
4. If candidate end exceeds `currentTime + 0.150`, drop incoming frame; keep current schedule.
5. No JavaScript queue beyond in-flight normalization and tracked scheduled sources.

## Related code files

- Create — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-timeline-scheduler.ts`: pure policy plus Web Audio source ownership.
- Create — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-timeline-scheduler.test.ts`: deterministic fake-clock/source tests.
- Create — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-audio-stream.ts`: Socket.IO, AudioContext, scheduler, diagnostics lifecycle.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-stream-metrics.ts`: scheduled horizon/reset/drop counters.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/CameraViewer.tsx`: replace inline audio refs/effects with hook.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/index.ts`: focused exports if needed.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/index.ts`: export hook if needed.

## Implementation Steps

1. Implement scheduler policy as a small deterministic state machine independent of React.
2. Inject context clock/source factory for unit tests; avoid browser globals in policy tests.
3. Track source node, start, end, and ended state. Detach callbacks during reset.
4. Implement `push`, `reset`, `suspend`, and `dispose` with idempotent cleanup.
5. Handle stream ID changes, sequence gaps, late frames, max-horizon drops, and context state transitions explicitly.
6. Build `useAudioStream` to create/resume context from user action, wire gain/filter, attach one socket handler, and dispose cleanly.
7. Move frame validation/conversion and diagnostics into the hook; return volume/state/metrics controls to CameraViewer.
8. Remove `audioQueueRef`, `isPlayingRef`, `nextPlayTimeRef`, threshold/max queue, recursive scheduler, and frame-rate logging.
9. Throttle metrics state to one update/second while internal counters remain immediate.
10. Test stable 20 Hz frames, jitter, one-second main-thread stall simulation, burst overflow, gap/duplicate, stream reset, suspend/resume, disable, and unmount.
11. Verify no overlapping sources after reset and no unbounded source/history growth.

## Todo list

- [x] Implement deterministic scheduling policy.
- [x] Implement source ownership and cleanup.
- [x] Add React/Socket lifecycle hook.
- [x] Refactor CameraViewer audio section.
- [x] Throttle UI statistics.
- [x] Add scheduler/lifecycle tests.
- [x] Verify timer removal and stable memory.

## Success Criteria

- Search confirms no recursive audio playback timer remains in the shipped audio path.
- Ideal 20 Hz simulation never underruns or grows beyond 150 ms horizon.
- Burst simulation bounds latency and reports dropped incoming frames.
- Context disable/reconnect/unmount leaves zero tracked/sounding future sources.
- Ten-minute deterministic scheduler simulation passes; real browser/LAN smoke is deferred to Phase 05.
- Frontend tests, type checks, lint, web build, and native build pass.

## Completion Notes

- Implemented deterministic scheduler policy in `packages/ui/src/lib/audio-timeline-scheduler.ts` with explicit 10 ms minimum lead, 50 ms target lead, and 150 ms maximum scheduled horizon.
- Added `useAudioStream` ownership for `AudioContext`, gain/filter chain, stream resets, throttled metrics, and source cleanup outside `CameraViewer` rendering.
- Removed the legacy recursive playback queue/timer from the UI audio path.
- Added scheduler, buffer conversion, hook lifecycle, and RAF cleanup regression tests.
- Verification completed on 2026-06-30:
  - `pnpm --filter @robo-fleet/ui test`: 51/51 tests passed
  - `pnpm lint`: 0 errors, 0 warnings
  - `pnpm exec turbo check-types --force`: passed
  - `pnpm exec turbo build --force`: passed
  - `git diff --check`: passed

## Risk Assessment

- Async Blob conversion can reorder frames. Serialize normalization or reject stale completion by sequence.
- Browser autoplay may leave context suspended. Surface state and require explicit user action; do not retry per frame.
- 150 ms maximum may be too tight on remote paths. Keep policy constants centralized, change only from Phase 05 evidence.
- Extracting 200+ audio lines may collide with dirty video edits. Preserve video diff and review final component carefully.

## Security Considerations

- Keep all payload bounds from Phase 03 before AudioBuffer allocation.
- Dispose nodes/listeners to prevent client-side resource exhaustion.
- Do not expose raw samples through diagnostics.

## Next steps

- Run full acceptance matrix in Phase 05.
- Do not tune lead constants from subjective listening alone; use continuity/latency metrics.

## Unresolved Questions

- Whether the 150 ms ceiling applies to scheduled start or hardware-audible output.
