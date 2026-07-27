# Implementation Report — Central Sherpa VAD STT Phase 02

- Date: 2026-07-03
- Plan: [Central Sherpa VAD STT and Dual-Source Voice](../260702-2316-central-sherpa-vad-stt/plan.md)
- Phase: [Phase 02 — Sherpa Runtime and Models](../260702-2316-central-sherpa-vad-stt/phase-02-sherpa-runtime-models.md)
- Status: complete

## Summary

- Replaced the central STT runtime dependency set with pinned `sherpa-onnx` `1.13.3` static linkage and removed normal Whisper usage from `central_speech_recognizer`.
- Added a closed English/Vietnamese offline profile catalog, strict startup configuration validation, and native CPU loading for Silero VAD plus offline recognizers.
- Extended `make models`, model documentation, and the download script to provision repeatable Sherpa ASR bundles under `models/.cache/sherpa-onnx/asr`.
- Kept Phase 02 scoped to startup validation only; the binary now warns that live audio handling remains deferred to Phase 03.

## Verification

- `cargo test -p central_speech_recognizer`: `6/6` passed
- `STT_PROFILE=en-vad-offline STT_MODEL_ROOT=/mnt/data/ws/sharing/robo-fleet-dora-rs/models/.cache/sherpa-onnx/asr cargo test -p central_speech_recognizer --test model_loading -- --ignored`: `1/1` passed
- `STT_PROFILE=vi-vad-offline STT_MODEL_ROOT=/mnt/data/ws/sharing/robo-fleet-dora-rs/models/.cache/sherpa-onnx/asr cargo test -p central_speech_recognizer --test model_loading -- --ignored`: `1/1` passed
- `cargo clippy -p central_speech_recognizer --all-targets --no-deps -- -D warnings`: passed
- `cargo build --release -p central_speech_recognizer`: passed
- `cargo build --manifest-path /mnt/data/ws/sherpa-onnx/rust-api-examples/Cargo.toml --example version`: passed
- `make models`: passed on initial download and idempotent re-run
- `make check-models`: passed
- `bash -n docker/scripts/download-models.sh`: passed
- Scoped `git diff --check` for Phase 02 files: passed

## Onboarding Impact

- Fresh environments should run `make models` before starting Orchestra STT work.
- Sherpa startup now uses `STT_PROFILE`, `STT_MODEL_ROOT`, `STT_NUM_THREADS`, `STT_VAD_THRESHOLD`, `STT_VAD_MIN_SILENCE_SECS`, `STT_VAD_MIN_SPEECH_SECS`, `STT_VAD_MAX_SPEECH_SECS`, and `STT_QUEUE_CAPACITY`.
- The current Orchestra dataflow wiring is a non-production probe for native-load validation only until Phase 03 lands audio processing.

## Next Steps

- Implement Phase 03 live VAD session management, bounded decode worker flow, and final-only transcription emission on the loaded Sherpa runtime.

## Unresolved Questions

- None.
