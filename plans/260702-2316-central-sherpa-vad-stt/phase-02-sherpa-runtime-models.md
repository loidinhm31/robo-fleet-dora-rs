# Phase 02 — Sherpa Runtime and Models

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phase 01](./phase-01-architecture-contracts-baseline.md)
- Research: [research synthesis](./research/research-synthesis.md)
- Reference examples: `/mnt/data/ws/sherpa-onnx/rust-api-examples/`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-02 |
| Description | Pin official Sherpa runtime, provision two offline profiles, and validate startup configuration. |
| Priority | P1 |
| Implementation status | Complete |
| Review status | Approved (2026-07-03) |
| Completed | 2026-07-03 |
| Effort | 6h |

## Key Insights

- Official examples pin `sherpa-onnx` 1.13.3 and explicitly select static/shared native linkage.
- Online and offline Zipformer models are not interchangeable.
- One model is active per process; startup-only selection avoids simultaneous model memory cost.
- Model download and Docker wiring currently assume GGML Whisper.

## Requirements

- Use `sherpa-onnx = 1.13.3`, `default-features = false`, static feature for the Orchestra binary.
- Support exactly two named profiles and CPU provider.
- Default to `en-vad-offline`.
- Validate every required file before native model creation.
- Make model download repeatable and idempotent.

## Architecture

```text
STT_PROFILE -> closed profile catalog -> encoder/decoder/joiner/tokens
                                      -> language code
STT_MODEL_ROOT -> Silero VAD + selected profile bundle
```

Profile catalog is code-owned. Clients cannot submit profile or filesystem paths.

## Related Code Files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/central_speech_recognizer/Cargo.toml`: Sherpa dependency, remove Whisper/hound.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/central_speech_recognizer/src/config.rs`: validated configuration and profile catalog.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/scripts/download-models.sh`: ASR bundles.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Makefile`: checks and help.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/README.md`: layout and profile setup.

## Implementation Steps

1. Replace `whisper-rs` and unused `hound` dependencies with pinned official `sherpa-onnx` static linkage.
2. Define model root `models/.cache/sherpa-onnx/asr` and subdirectories for Silero, English, and Vietnamese bundles.
3. Pin English bundle `icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04` with its nested tokens and int8 encoder/joiner paths.
4. Pin Vietnamese bundle `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` with direct tokens and int8 encoder/joiner paths.
5. Download `silero_vad.onnx` from the pinned Sherpa ASR release source.
6. Extend download script with file-existence guards, temporary archive names, extraction validation, and cleanup on success.
7. Extend `make check-models` to report each required file for both profiles.
8. Parse and validate `STT_PROFILE`, `STT_MODEL_ROOT`, `STT_NUM_THREADS`, VAD threshold/durations, and queue capacity.
9. Fix Silero sample rate to 16 kHz and window to 512; reject configuration attempting incompatible values.
10. Add configuration tests for defaults, each profile mapping, invalid enum, missing files, invalid VAD ranges, and queue bounds.
11. Confirm a minimal official example builds with the same crate feature selection on the Orchestra host.

## Todo List

- [x] Pin Sherpa crate and linkage.
- [x] Add profile catalog.
- [x] Add model downloads.
- [x] Add model checks.
- [x] Add configuration validation.
- [x] Add configuration tests.
- [x] Verify native example build.

## Completion Evidence

- Completed: 2026-07-03.
- `central_speech_recognizer` now links pinned `sherpa-onnx` `1.13.3` with static native support and no normal `whisper-rs` dependency.
- Closed English and Vietnamese offline profile catalogs validate encoder, decoder, joiner, tokens, and Silero VAD files before native startup.
- `make models` now provisions repeatable Sherpa ASR bundles with atomic extraction and idempotent re-runs.
- Phase 02 runtime intentionally stops at startup validation and native model loading; live audio decode remains deferred to Phase 03 and is called out in the probe logs and dataflow comments.
- Validation passed: `cargo test -p central_speech_recognizer`.
- Validation passed: `STT_PROFILE=en-vad-offline STT_MODEL_ROOT=/mnt/data/ws/sharing/robo-fleet-dora-rs/models/.cache/sherpa-onnx/asr cargo test -p central_speech_recognizer --test model_loading -- --ignored`.
- Validation passed: `STT_PROFILE=vi-vad-offline STT_MODEL_ROOT=/mnt/data/ws/sharing/robo-fleet-dora-rs/models/.cache/sherpa-onnx/asr cargo test -p central_speech_recognizer --test model_loading -- --ignored`.
- Validation passed: `cargo clippy -p central_speech_recognizer --all-targets --no-deps -- -D warnings`.
- Validation passed: `cargo build --release -p central_speech_recognizer`.
- Validation passed: `cargo build --manifest-path /mnt/data/ws/sherpa-onnx/rust-api-examples/Cargo.toml --example version`.
- Validation passed: `bash -n docker/scripts/download-models.sh` and `make check-models`.

## Success Criteria

- Clean `make models` produces every required file in the documented layout.
- Re-running model download does not redownload valid bundles.
- Each profile resolves to an existing encoder, decoder, joiner, tokens file, and language.
- Invalid profile/config fails with a concise status-safe message.
- Central package resolves no Whisper dependency.

## Risk Assessment

- Risk: Static native artifacts increase build time and binary size. Mitigation: explicit dependency cache layer and release-build measurement.
- Risk: Upstream archive layout changes. Mitigation: pin release names and verify expected files after extraction.
- Risk: Download interruption leaves a false-positive directory. Mitigation: extract to temporary location and rename only after validation.

## Security Considerations

- Use HTTPS pinned release URLs; never execute downloaded scripts.
- Treat model paths as server configuration, not Socket.IO input.
- Avoid logging secrets or full environment dumps during validation.

## Next Steps

Proceed to Phase 03 once both profile catalogs load their native recognizers on the Orchestra host.
