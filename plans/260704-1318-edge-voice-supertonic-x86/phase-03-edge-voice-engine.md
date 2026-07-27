# Phase 03 — Edge Voice Engine

## Context Links

- [Parent plan](./plan.md)
- [Rust implementation comparison](./research/01-supertonic-rust-production-comparison.md)
- [Phase 01 contracts](./phase-01-architecture-and-contract-gate.md)
- [Phase 02 models](./phase-02-model-cache-reset-and-bootstrap.md)
- Depends on: Phases 01–02

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Replace Piper node with a responsive, bounded Supertonic synthesis service. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Approved |
| Completed at | 2026-07-04 23:19 ICT |
| Recommended model | GPT-5.5; GPT-5.4 for focused tests/review |
| Estimated effort | 10h |

## Key Insights

- Current `sherpa_tts` blocks its Dora loop and owns a second audio device.
- Sherpa `generate_with_config` callback supports progress and cancellation.
- Callback samples are cumulative; emission must track the last emitted index.
- `OfflineTts` is Send/Sync but one sequential worker avoids resource contention and ordering ambiguity.

## Requirements

### Functional

- Load one Supertonic INT8 engine at startup.
- Accept TTS commands, runtime config, playback state, and stop.
- Queue by priority with capacity 8; reject newest normal/low command when full.
- Snapshot config at dequeue; config update never mutates active utterance.
- Emit 20 ms F32 PCM chunks, status, command results, and metrics.
- Cancel generation when walkie preempts or Dora stops.

### Non-functional

- Exact `sherpa-onnx = 1.13.3`, static feature.
- No model downloads, speaker device, persistence, or arbitrary language detection.
- Control loop remains responsive during model load and synthesis.
- Sanitized errors; no absolute model paths in external status.

## Architecture

```text
Dora loop
  ├── validates/configures runtime state
  ├── owns bounded priority queue
  ├── receives worker events
  └── sends Dora outputs
          │
          └── synthesis worker
                ├── one OfflineTts
                ├── config snapshot
                ├── cancellation token
                └── cumulative callback -> 20 ms deltas
```

Recommended crate modules:

```text
main.rs             wiring only
config.rs           env/runtime validation
model.rs            paths and OfflineTts construction
queue.rs            bounded priority behavior
protocol.rs         Dora metadata and result mapping
worker.rs           synthesis/cancellation
runtime.rs          state machine/event loop
```

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Rename/replace | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/sherpa_tts` → `rover-kiwi/edge_voice` | New node |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml` | Workspace member |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.lock` | Remove `sherpa-rs`; add official Sherpa |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/performance_monitor/src/main.rs` | Monitor `edge-voice` process |

## Implementation Steps

1. Preserve old node behavior only as test/reference; create modular `edge_voice` crate.
2. Parse immutable deployment config: model directory, threads, defaults, queue size, debug flag.
3. Validate defaults and all seven model files before native object construction.
4. Initialize Dora first, publish `loading`, then load model on worker startup.
5. Verify engine sample rate 44,100 and speaker count 10; publish `ready` or durable `error`.
6. Implement bounded priority queue and deterministic emergency clearing behavior.
7. Map runtime config to `GenerationConfig`: language extra, SID, speed, steps; apply volume with finite/clipping validation.
8. Track callback delta and emit complete 20 ms chunks; flush final partial chunk.
9. Emit accepted/rejected/speaking/completed/interrupted/failed results with command ID.
10. Add cancellation token for walkie-active state and stop events.
11. Rename workspace/package/process references and remove old `sherpa-rs` source.
12. Request GPT-5.4 focused code review; GPT-5.5 resolves any FFI/concurrency findings.

## Todo List

- [x] Crate renamed and modularized
- [x] Static Sherpa dependency pinned
- [x] Model validation implemented
- [x] Worker/state machine implemented
- [x] Queue policy tested
- [x] Config mapping tested
- [x] Chunk emission tested
- [x] Cancellation tested
- [x] Status/results/metrics tested
- [x] Old Piper source removed

## Success Criteria

- Missing/corrupt model produces responsive error state, not dataflow cascade failure.
- English and Vietnamese synthesize without model reload.
- All SIDs 0–9 generate non-empty 44.1 kHz output.
- Config updates are processed during synthesis and affect only later dequeues.
- Queue never exceeds capacity and cancellation ends generation promptly.
- Node never opens CPAL/Rodio output.

## Risk Assessment

- Risk: callback lifetime/cancellation misuse across FFI. Mitigation: isolated worker tests and GPT-5.5 review.
- Risk: duplicate cumulative samples. Mitigation: monotonic emitted-index assertions.
- Risk: model load blocks status. Mitigation: Dora initialized before worker load.
- Risk: static Sherpa increases binary size. Mitigation: accepted for runtime isolation; verify Docker size later.

## Security Considerations

- Trim/reject empty text; enforce 1,000-character maximum at rover boundary too.
- Strip angle-bracket markup before synthesis.
- Reject NaN/infinite config values and unknown enum variants.
- Avoid logging full user text at info level.

## Next Steps

- Proceed to [Phase 04](./phase-04-source-aware-playback-and-capture-suppression.md) once PCM chunks and cancellation events are stable.

## Validation Summary

- `cargo test -p edge_voice`
- `cargo clippy -p edge_voice --all-targets --no-deps -- -D warnings`
- `cargo check --workspace`
- `cargo check -p edge_voice --release`
- `bash -n docker/scripts/entrypoint-rover.sh`
- stale refs search

## Known Deferrals

- Phase 04/05 wire `playback_state`, `tts_audio`, `config`, `status` transport.
