# Phase 03 — Central VAD/Offline Recognizer

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phase 01](./phase-01-architecture-contracts-baseline.md), [Phase 02](./phase-02-sherpa-runtime-models.md)
- Reference VAD: `/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/silero_vad_remove_silence.rs`
- Reference offline flow: `/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/zipformer_transducer_simulate_streaming_microphone.rs`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-02 |
| Description | Replace Whisper buffering with validated dual-source audio, per-stream VAD, and bounded offline decode. |
| Priority | P1 |
| Implementation status | Complete |
| Review status | Approved (2026-07-03) |
| Completed | 2026-07-03 |
| Effort | 12h |

## Key Insights

- Rover input is already validated transport S16LE; central must revalidate Dora metadata at its trust boundary.
- Browser AudioContext sample rate may differ from requested 16 kHz and needs stateful resampling.
- A single global buffer would mix fleet sources and is forbidden.
- Offline decoding must not block Dora input processing.
- A browser stop event must flush VAD because silence may not arrive after capture stops.

## Requirements

- Keep every Rust source module near or below 200 lines; use a thin `main.rs` and testable library modules.
- Accept rover BinaryArray S16LE and browser Float32Array only on their named inputs.
- Keep independent sequence, resampler, remainder, and VAD state per source stream.
- Emit one final result per completed VAD segment.
- Remain alive and answer status requests when model initialization fails.
- Apply bounded non-blocking decode backpressure.

## Architecture

```text
Dora input -> source adapter -> metadata/sequence validation -> optional resampler
           -> per-session 512-sample accumulator -> Silero VAD -> DecodeJob
           -> bounded worker owning OfflineRecognizer -> SpeechTranscription
```

Session keys:

- Rover: `(entity_id, stream_id)`
- Browser: `stream_id`, with web-bridge-assigned target

## Related Code Files

- Replace `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/central_speech_recognizer/src/main.rs`: thin runtime entry.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/central_speech_recognizer/src/lib.rs`: module exports.
- Create focused modules under `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/central_speech_recognizer/src/` for audio input, sessions, VAD, decoder worker, status, and runtime.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/central_speech_recognizer/Cargo.toml`: test/runtime dependencies only as needed.

## Implementation Steps

1. Initialize Dora before model loading and emit `loading` status.
2. Validate model files; create one probe Silero VAD and selected `OfflineRecognizer`.
3. On failure, store/emit sanitized `error`, keep event loop alive, reject audio, and answer status requests.
4. On success, move recognizer into one dedicated decode thread and emit `ready`.
5. Implement rover metadata parsing using existing `AudioFrameMetadata` validation. Require S16LE, 16 kHz, one channel, exact payload size, and nonempty BinaryArray element.
6. Decode little-endian pairs with `i16::from_le_bytes` and divide by `32768.0`; reject odd bytes before conversion.
7. Implement browser metadata parsing for F32, finite values, exact sample count, one channel, supported sample-rate bounds, stream identity, and target.
8. Create a stateful `LinearResampler` only when browser rate differs from 16 kHz; flush it on stop.
9. Track monotonically increasing frame IDs. Reset and discard current utterance on stream replacement, any gap, duplicate, or regression; process the first valid frame after reset as a new utterance.
10. Accumulate exactly 512 normalized samples per VAD call and retain only the incomplete tail.
11. Drain all available VAD segments after each accepted window. Copy segment samples into immutable decode jobs with source metadata and duration.
12. On browser stop/disconnect, flush VAD, enqueue remaining segments, then retire session state.
13. Submit jobs with `try_send` to a bounded queue. On full/disconnected queue, drop the new utterance and increment explicit metrics.
14. Worker creates one offline stream per job, accepts 16 kHz samples, decodes, trims text, and emits only nonempty results through an `Arc<Mutex<DoraNode>>`.
15. Generate UUID utterance ID, preserve source/target/stream/profile, and set confidence to `None`.
16. Add interval and shutdown logs for frames, validation errors, resets, speech segments, empty results, queue drops, decode count, RTF, and latency percentiles.
17. Add unit tests with a fake decoder boundary so most tests do not require model files.

## Todo List

- [x] Modularize central crate.
- [x] Implement startup/status lifecycle.
- [x] Implement rover adapter.
- [x] Implement browser adapter/resampler.
- [x] Implement session sequencing and reset.
- [x] Implement 512-sample VAD feed/flush.
- [x] Implement bounded decoder worker.
- [x] Implement transcription output.
- [x] Add metrics and unit tests.

## Completion Evidence

- Completed: 2026-07-03.
- `central_speech_recognizer` now runs through a thin runtime entry with dedicated modules for audio validation, browser control, session state, segmenting, decode worker, metrics, startup, and status handling.
- Dual-source handling is implemented with browser resampling, rover/browser format validation, per-session sequencing, reset behavior, and stop-time VAD flush paths.
- Bounded offline decode submission and nonblocking drop behavior are present in the runtime/decoder split, with status-safe startup and error handling kept alive for `stt_status_request`.
- Unit coverage now exists for audio input validation, browser control, session isolation/reset behavior, browser resampler flush, decode boundary behavior, runtime submission handling, and status sanitization.

## Success Criteria

- Boundary conversion maps `-32768` to `-1.0`, `0` to `0.0`, and `32767` below `1.0`.
- Invalid format, dimensions, payload, or sequence never reaches VAD.
- Two rover streams and one browser stream cannot share samples or VAD state.
- Browser stop produces a final result when VAD has buffered speech.
- Queue saturation never blocks Dora input handling.
- Status requests work in loading, ready, and error states.
- No confidence value is synthesized.

## Risk Assessment

- Risk: VAD construction per source may be expensive. Mitigation: measure creation/RSS; retain one instance for each active stream and clean it on stop/change.
- Risk: Worker result after browser stop loses routing. Mitigation: Phase 04 retains closing ownership until result or timeout.
- Risk: Dora node mutex contention. Mitigation: lock only for `send_output`, never around decode.

## Security Considerations

- Bound sample count, sample rate, channels, session count through authenticated sources, and decode queue size.
- Sanitize model/native errors before status output.
- Avoid transcript audio logging and never log raw sample buffers.

## Next Steps

Proceed to Phase 04.
