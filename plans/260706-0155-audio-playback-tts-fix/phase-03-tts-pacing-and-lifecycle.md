# Phase 03 — TTS Pacing and Lifecycle

## Context Links

- [Parent plan](./plan.md)
- [Phase 01 contract](./phase-01-contract-and-architecture.md)
- Runtime: `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_voice/src/runtime.rs`
- Worker: `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_voice/src/worker.rs`

## Overview

- Date: 2026-07-06
- Priority: P1
- Description: Prevent offline synthesis from flooding Dora while preserving cancellation and completion ordering.
- Implementation status: Complete
- Review status: Approved
- Completion timestamp: 2026-07-06

## Key Insights

- Supertonic emits 882-sample/20 ms chunks but the runtime currently drains every ready event immediately.
- The bounded worker channel limits memory but does not pace downstream publication.
- Completion means final PCM consumption, not synthesis completion or final Dora send.
- Sleep-loop iteration counts drift; pacing must use cumulative samples and monotonic deadlines.

## Requirements

- Publish TTS chunks in worker order no faster than media duration.
- Keep Dora/control input response within one 20 ms interval during pacing.
- Cancel walkie-preempted synthesis and unsent chunks without affecting other commands.
- Emit one synthesis terminal edge and one public terminal result per accepted command.
- Expose pacing lag and generated/emitted sample accounting.

## Architecture

Add a small `TtsPacer<C: Clock>` between `WorkerEvent::AudioChunk` and Dora output. Production uses `Instant`; tests use a fake clock. The runtime accepts at most the pacer's bounded pending capacity and stops draining worker audio events when full, allowing the existing synchronous worker channel to apply backpressure.

The first chunk may publish immediately. After each emission, the next deadline is the later of the cumulative media deadline and `actual_emit_time + emitted_chunk_duration`, preventing catch-up bursts after scheduler stalls.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_voice/src/tts_pacer.rs` — clock abstraction and pacing state.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_voice/src/runtime.rs` — paced worker drain and lifecycle handling.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_voice/src/worker.rs` — expose constants/accounting only as needed; retain bounded channels.
- Add focused pacer/runtime tests under `rover-kiwi/edge_voice/src/`.

## Implementation Steps

1. Implement `Clock::now()` and fake-clock-compatible `TtsPacer` with command ID, expected frame ID, sample rate, pending chunk, next deadline, and generated/emitted counters.
2. Validate worker frames are one command, 44.1 kHz, monotonic, and normally 882 samples; allow only the final partial frame.
3. Change runtime iteration order: emit one due paced chunk, process at most one eligible worker event, process Dora input with timeout capped at 20 ms or the next media deadline, then dispatch queued commands.
4. Do not receive another worker audio event while the pacer slot is occupied. Continue consuming cancellation, playback-state, and playback-result inputs.
5. On worker `Completed`, wait until every accepted chunk has been emitted before sending `tts_synthesis_state=Completed`.
6. On cancellation/preemption, clear the current command's pending chunk, signal the worker, discard later stale worker events for that command, and emit one interrupted synthesis state.
7. On send failure or invalid worker sequence, cancel synthesis and emit explicit failed state with sanitized detail.
8. Add periodic metrics: generated/emitted frames and samples, pending depth, pacing lag, worker-channel backpressure count, cancellations, and terminal reason.

## Todo list

- [x] Pacer and clock abstraction implemented
- [x] Runtime worker drain bounded
- [x] No-burst deadline policy implemented
- [x] Cancellation remains responsive
- [x] Synthesis completion waits for final send
- [x] Pacing metrics emitted
- [x] Fake-clock tests passing

## Verification Status

- `cargo test -p edge_voice`: 24 tests passed.
- `cargo clippy -p edge_voice --all-targets --no-deps -- -D warnings`: passed.
- Code review completed after review-cycle fixes for stale worker audio discard, per-command pacing metrics, terminal metric emission, and lifecycle-safe best-effort metrics transport.

## Success Criteria

- A 60-second synthetic utterance never publishes two 20 ms chunks less than 20 ms apart.
- Scheduler delay shifts later deadlines instead of producing a catch-up burst.
- Frame ordering and sample totals match worker output exactly.
- Walkie cancellation becomes observable within 20 ms and produces one interrupted result.
- `tts_synthesis_state=Completed` follows the final `tts_audio` send.

## Risk Assessment

- Blocking the synthesis callback can keep the worker occupied for playback duration. This is intentional bounded backpressure; later commands remain in the existing bounded priority queue.
- A poorly structured loop could starve Dora control inputs. Mitigation: hard 20 ms poll ceiling and fake-clock cancellation tests.

## Security Considerations

- Sanitize internal errors before public results.
- Keep existing text/config validation and queue capacity unchanged.
- Do not allow pacing configuration from unauthenticated client input.

## Next steps

With source burst removal proven, replace destructive playback overflow in Phase 04.
