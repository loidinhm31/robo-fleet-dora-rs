# Supertonic Rust Production Comparison

Date: 2026-07-04
Status: complete

## Candidates

### Direct Supertone Rust example

Source: https://github.com/supertone-inc/supertonic/tree/main/rust

- Uses `ort = 2.0.0-rc.7` directly.
- Reimplements normalization, chunking, voice loading, tensor preparation, diffusion, and vocoder orchestration in application code.
- README documents a macOS cleanup workaround using `mem::forget()` and `libc::_exit()`.
- GPU flag exists but GPU support is not implemented.
- Valuable as algorithm/reference evidence, not as the rover production dependency.

### Sherpa-ONNX Rust API

Sources:

- https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/supertonic_tts.rs
- Local example: `/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/supertonic_tts.rs`

- Provides `OfflineTts`, `OfflineTtsSupertonicModelConfig`, and per-request `GenerationConfig`.
- Supports callback progress/cancellation, language through `extra["lang"]`, SID, speed, and denoising steps.
- Wraps the maintained Sherpa C++ implementation and returns generated F32 samples with sample rate.
- Workspace already pins `sherpa-onnx = 1.13.3` with static linkage for central STT.
- Static linkage isolates Sherpa's native runtime from the dynamic ONNX Runtime used by rover vision.

## Selected Model Bundle

Source: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

```text
name: sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
size: 128774318 bytes
sha256: 82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427
```

Required files:

- `duration_predictor.int8.onnx`
- `text_encoder.int8.onnx`
- `vector_estimator.int8.onnx`
- `vocoder.int8.onnx`
- `tts.json`
- `unicode_indexer.bin`
- `voice.bin`

Voice pack creation sorts JSON filenames. Resulting mapping is F1-F5 at SIDs 0-4 and M1-M5 at SIDs 5-9. Default M1 is SID 5.

## Decision

Use exact `sherpa-onnx = 1.13.3`, `default-features = false`, `features = ["static"]` in `edge_voice`. Use the direct Supertone project only for behavioral comparison and test corpus ideas.

## Production Constraints

- One resident `OfflineTts`; one synthesis worker; no concurrent generations.
- Model paths deployment-owned; never accepted from Socket.IO.
- Emit bounded F32 chunks, not JSON sample arrays or one unbounded utterance.
- Callback cancellation must respond to walkie preemption and Dora stop.
- Validate 44.1 kHz output and exactly 10 speakers at startup/model test.

## Unresolved Questions

- OpenRAIL-M commercial redistribution approval is outside repository engineering.
