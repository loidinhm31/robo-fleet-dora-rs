# Phase 03 Implementation Report

Date: 2026-07-03
Plan: `plans/260702-2316-central-sherpa-vad-stt/phase-03-central-vad-recognizer.md`
Status: complete
Review: approved after fix pass

## Summary

Phase 03 replaced the central Whisper buffering path with a modular Sherpa VAD/offline recognizer runtime that:

- validates rover and browser audio at the Dora boundary
- keeps isolated per-stream sequencing, resampling, VAD, and flush behavior
- starts Dora before model loading and remains status-responsive on startup failure
- uses one bounded decode worker and emits final-only transcriptions without synthetic confidence

## Files

- `orchestra/central_speech_recognizer/` modular runtime split across input, session, segmenter, decoder, status, startup, and metrics modules
- `orchestra/orchestra-dataflow.yml` adds rover audio fan-in to central STT and exposes `stt_status`
- `ARCHITECTURE.md` updates central STT runtime wording from Whisper baseline to Sherpa runtime
- `plans/260702-2316-central-sherpa-vad-stt/plan.md` and Phase 03 detail mark completion

## Verification

- `cargo fmt --all -- --check`
- `make check-models`
- `cargo test -p central_speech_recognizer`
- `STT_MODEL_ROOT=... cargo test -p central_speech_recognizer --test model_loading -- --ignored`
- `STT_MODEL_ROOT=... STT_PROFILE=vi-vad-offline cargo test -p central_speech_recognizer --test model_loading -- --ignored`
- `cargo clippy -p central_speech_recognizer --all-targets --no-deps -- -D warnings`
- `cargo build --release -p central_speech_recognizer`
- `dora graph orchestra/orchestra-dataflow.yml`
- `dora graph rover-kiwi/rover-kiwi-dataflow.yml`

## Onboarding Impact

No new secrets or API keys were introduced.

Runtime prerequisites remain:

- `STT_MODEL_ROOT` pointing at the Sherpa ASR bundle root when overriding defaults
- existing Sherpa model bundles present under `models/.cache/sherpa-onnx/asr`

## Deferred To Phase 04

- wiring live `browser_control` and `stt_status_request` transport through `web_bridge`
- authoritative browser socket ownership and target snapshot transport
