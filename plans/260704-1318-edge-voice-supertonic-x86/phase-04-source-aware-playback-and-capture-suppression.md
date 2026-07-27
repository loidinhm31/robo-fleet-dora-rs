# Phase 04 — Source-Aware Playback and Capture Suppression

## Context Links

- [Parent plan](./plan.md)
- [Phase 01 contracts](./phase-01-architecture-and-contract-gate.md)
- [Phase 03 edge voice](./phase-03-edge-voice-engine.md)
- [Current audio playback](../../rover-kiwi/audio_playback/src/main.rs)
- Depends on: Phases 01 and 03

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Make one node own the speaker, resample all sources, enforce walkie priority, and gate rover microphone publication. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Approved |
| Recommended model | GPT-5.5; GPT-5.4 for DSP fixtures |
| Estimated effort | 10h |

## Key Insights

- Current playback assumes every source is 16 kHz mono.
- Supertonic is 44.1 kHz; relabeling samples would corrupt pitch and duration.
- Live walkie audio cannot wait behind long robot speech.
- Playback state must represent samples actually consumed by CPAL, not merely enqueued.
- Capture suppression and manual capture enablement are independent state variables.

## Requirements

### Functional

- Accept named `walkie_audio` and `tts_audio` inputs with validated metadata.
- Open one native CPAL output stream and convert both sources to its rate/channel count.
- Walkie preempts active/queued TTS on first valid walkie frame.
- Reject/interrupt TTS while walkie is active; consider walkie inactive 250 ms after its last valid frame.
- Emit playback lifecycle and interruption events.
- Suppress rover microphone publication during any playback and 400 ms after idle.

### Non-functional

- No allocation, logging, locks with unbounded work, or resampler construction in CPAL callback.
- Bounded per-source buffers. Walkie backlog drops oldest samples to preserve live latency.
- TTS source never mixes with walkie.
- Playback unavailable degrades to explicit error/silent state without crashing dataflow.

## Architecture

```text
walkie_audio 16 kHz ─┐
                     ├─ validate -> source arbiter -> rubato -> bounded output ring -> CPAL
tts_audio 44.1 kHz ──┘                     │
                                          ├─ playback_state -> edge_voice cancellation
                                          └─ playback_state -> audio_capture suppression
```

Use exact `rubato = 3.0.0`, asynchronous sinc resampling, preallocated input/output adapters, and one resampler state per source/rate pair. Duplicate mono samples to hardware channels after resampling.

Playback state machine:

```text
Idle -> TtsActive -> Idle
Idle -> WalkieActive -> Idle
TtsActive --walkie frame--> WalkieActive + TtsInterrupted
WalkieActive --tts request--> TtsRejected(walkie_active)
```

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Refactor | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/main.rs` | Split into focused playback modules |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/Cargo.toml` | Pin resampler dependency |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_capture/src/main.rs` | Playback-derived suppression |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/rover-kiwi-dataflow.yml` | Named inputs and state wiring |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/rover-kiwi-direct-dataflow.yml` | Equivalent direct wiring |

## Implementation Steps

1. Extract metadata parsing, source arbiter, resampler, ring buffer, and state reporter from playback main.
2. Select default CPAL output config; support F32/I16/U16 device formats without changing internal F32 pipeline.
3. Prebuild source resamplers outside callback; rebuild only in Dora thread when a validated source rate changes.
4. Implement bounded TTS and low-latency walkie buffers.
5. Track actual CPAL consumption; emit active after first consumed non-silent sample and idle after final sample.
6. On first walkie frame: clear TTS buffer, emit interruption, signal `edge_voice` cancellation, then enqueue walkie.
7. Track walkie deadline at last frame +250 ms; reject TTS until deadline expires.
8. Update `audio_capture` with `capture_enabled_by_user`, `playback_suppressed`, and suppression-tail deadline.
9. Drain capture ring while suppressed; flush queued microphone samples before resume.
10. Add deterministic tone/duration tests for 16k→native and 44.1k→native resampling.
11. Add concurrency review focused on callback safety, locks, cancellation races, and shutdown.

## Todo List

- [x] Playback modularized
- [x] Native device format handling added
- [x] Rubato resampling added
- [x] Buffers bounded
- [x] Walkie preemption implemented
- [x] Actual playback state emitted
- [x] Capture suppression/tail implemented
- [x] Resampling tests added
- [x] State/race tests added
- [x] GPT-5.5 real-time review complete

## Success Criteria

- 44.1 kHz TTS duration/pitch remains correct on host output device.
- Walkie preempts TTS within one received audio frame.
- TTS interrupted result reaches `edge_voice` with original command ID.
- No rover microphone frame is emitted while playback active or during 400 ms tail.
- Browser-origin STT remains unaffected.
- 100 sequential short utterances produce no buffer overrun/underrun in target test.

## Risk Assessment

- Risk: CPAL callback blocking causes underruns. Mitigation: preallocation and minimal ring operations.
- Risk: resampler tail truncation. Mitigation: explicit flush and duration assertions.
- Risk: walkie activity inferred from packet timing. Mitigation: fixed 250 ms deadline and sequence tests.
- Risk: cyclic dataflow events. Mitigation: playback state is control-only and edge voice never echoes it unchanged.

## Security Considerations

- Reject malformed sample metadata and oversized chunks before buffer writes.
- Reject NaN/Inf samples or normalize them to silence with metric increment.
- Bound volume scaling and clip output to valid sample range.

## Next Steps

- Proceed to [Phase 05](./phase-05-fleet-transport-and-runtime-authority.md) after playback lifecycle is reliable locally.

## Completion Notes

- Public TTS completion now follows actual CPAL sample retirement instead of enqueue state.
- Walkie transport validates sequence and payload bounds before admission and enforces the 250 ms authority window.
- Playback interruption and failure paths clear matching queued TTS audio without emitting duplicate lifecycle results.
- Capture publication stays suppressed during active playback and for the 400 ms post-playback tail.
- Validation passed with 124 targeted Rust tests, workspace `cargo check`, strict changed-crate Clippy, `cargo fmt --check`, `git diff --check`, and both rover Dora graph loads.
