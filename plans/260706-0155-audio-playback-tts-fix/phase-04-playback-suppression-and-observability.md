# Phase 04 — Playback, Suppression, and Observability

## Context Links

- [Parent plan](./plan.md)
- [TTS pacing](./phase-03-tts-pacing-and-lifecycle.md)
- Playback runtime: `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/runtime.rs`
- Capture gate: `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_capture/src/capture_gate.rs`

## Overview

- Date: 2026-07-06
- Priority: P1
- Description: Eliminate destructive TTS overflow and make speaker activity impossible to miss at microphone capture.
- Implementation status: Complete
- Review status: Approved
- Completion: 2026-07-06

## Key Insights

- Current TTS enqueue partially writes, clears all queued speech, then blocks later chunks.
- CPAL can produce `Active -> Idle` between Dora turns; draining both states into a small queue can hide activity.
- A 1,000 ms jitter buffer is enough after source pacing and remains below 200 KiB at 48 kHz mono F32.
- A full-buffer failure must occur before the four-frame Dora queue can overwrite unprocessed TTS.

## Requirements

- Enqueue each resampled TTS frame atomically or retain it for ordered retry.
- Use a 1,000 ms configurable TTS buffer and 60 ms stall deadline.
- Preserve walkie-first arbitration and begin live audio within one 20 ms frame.
- Publish `Active` if any real sample was consumed during a scheduler interval; publish `Idle` only after a later fully idle interval.
- Suppress microphone publication during playback and for 400 ms after valid idle.
- Emit periodic and terminal accounting for every loss/failure path.

## Architecture

`PlaybackBuffers::try_enqueue_tts_frame` first checks free capacity and then writes the complete frame; CPAL is the only consumer, so capacity cannot shrink during the producer operation. `TtsArbiter` owns an ordered pending FIFO capped at three 20 ms frames. The first capacity miss starts a monotonic 60 ms deadline. Recovery drains pending frames in order; deadline expiry or pending overflow fails the command explicitly and clears only that command.

Replace transition-event replay with an atomic interval activity summary. CPAL records the latest non-idle `(source, token)` consumed since the last scheduler exchange plus its current source. Each tick reports interval activity first; only a later tick with no activity and current idle reports `Idle`.

## Related code files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/buffers.rs` — atomic enqueue and activity summary.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/tts-arbiter.rs` — pending FIFO, deadline, explicit failure.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/runtime.rs` — configuration, retry, metrics.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/state.rs` — coalescing and sequence IDs.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_capture/src/capture_gate.rs` and `main.rs` — stale-state rejection and suppression metrics.

## Implementation Steps

1. Parse and validate buffer/stall environment values with safe bounds; defaults are 1,000 ms TTS, 80 ms walkie, and 60 ms stall.
2. Replace sample-by-sample lossy TTS enqueue with complete-frame admission. Increment drop counters only for explicit terminal clears, not temporary fullness.
3. Add three-frame pending FIFO and retry it before accepting later TTS audio. Preserve tokens and sequence order.
4. Fail on 60 ms deadline, fourth pending frame, TTS sequence gap, device stream error, or invalid command transition. Emit one `PlaybackFailed` result.
5. Keep `clear_tts` only for walkie preemption, cancellation, device failure, or terminal stall. Remove ordinary-capacity destructive clears.
6. Preserve walkie oldest-drop behavior, but bound the playback queue to 80 ms and count frames/samples/age represented by every drop.
7. Replace consumption-event draining with interval coalescing. Report `Active` for any consumed TTS/walkie sample and defer `Idle` to the next fully idle tick.
8. Increment `PlaybackState.sequence_id` on every emitted state. Capture and edge voice store the last ID and ignore duplicate/regressed states.
9. Keep the capture tail anchored to the first accepted idle after active playback; unavailable must not shorten an active suppression lease.
10. Add five-second/shutdown logs for received, enqueued, consumed, pending, cleared, dropped, gaps, state sequence, suppression frames/samples, queue duration, and terminal reasons.

## Todo list

- [x] Configurable bounded buffers implemented
- [x] Complete-frame TTS enqueue implemented
- [x] 60 ms ordered retry implemented
- [x] Destructive ordinary overflow removed
- [x] Interval playback activity coalesced
- [x] Playback state sequencing enforced
- [x] Capture suppression metrics added
- [x] Arbitration and lifecycle tests passing

## Success Criteria

- Temporary fullness never partially enqueues or clears queued TTS.
- Every accepted TTS command completes after final consumption or terminates once with an explicit reason.
- Walkie preempts TTS once and starts within 20 ms of the first valid rover frame.
- Capture publishes zero frames during active playback and the following 400 ms.
- `Active` and `Idle` cannot be published back-to-back in one scheduler turn.

## Risk Assessment

- Pending FIFO adds a second small buffer. Its three-frame/60 ms cap is fixed to remain below Dora overwrite time.
- Atomic activity exchange can lose source detail if multiple sources play in one interval. Walkie priority makes the latest walkie activity authoritative; tests must cover the transition.

## Security Considerations

- Bound all environment-derived capacities before multiplication/allocation.
- Reject malformed/stale states instead of changing suppression state.
- Logs expose command/stream IDs but never PCM or TTS text.

## Next steps

Run Phase 05 end-to-end verification and acoustic acceptance.
