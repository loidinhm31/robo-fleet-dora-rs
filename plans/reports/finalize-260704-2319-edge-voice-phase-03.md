# Phase 03 Finalization - Edge Voice Engine

Date: 2026-07-04 23:19 ICT
Plan: `plans/260704-1318-edge-voice-supertonic-x86`
Phase: 03 - Edge voice engine

## Summary

- Replaced retired `rover-kiwi/sherpa_tts` with modular `rover-kiwi/edge_voice`.
- `edge_voice` loads one Supertonic INT8 engine on a worker after Dora init, emits `loading`/`ready`/`error` status, and keeps the Dora loop responsive.
- Added bounded priority queue, config snapshot-at-dequeue, text sanitization, 20 ms F32 PCM chunking, cancellation, terminal command results, and metrics.
- Updated workspace, rover dataflows, rover Docker rename references, docs, and performance monitor process name.
- Pinned rover Docker ONNX Runtime to `1.16.3` to match rover `ort 1.16.3` vision crates.

## Validation

- `cargo test -p edge_voice` passed, 13 tests.
- `cargo clippy -p edge_voice --all-targets --no-deps -- -D warnings` passed.
- `cargo check -p edge_voice --release` passed.
- `cargo check --workspace` passed with unrelated `kokoro_tts` unused import warnings.
- `bash -n docker/scripts/entrypoint-rover.sh` passed.
- Stale runtime reference search for retired `sherpa_tts`, `sherpa-tts`, `sherpa-rs`, VITS-Piper path, legacy `TTS_MODEL_DIR`, and `ORT_VERSION=1.17.1` passed.
- Code review cycle completed at 8.5/10 with no critical issues; user approved.

## Onboarding Notes

- Required Supertonic files are present under `models/.cache/sherpa-onnx/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/`.
- Override model path with `EDGE_VOICE_MODEL_DIR` only when needed.
- Key runtime env vars: `EDGE_VOICE_NUM_THREADS`, `EDGE_VOICE_QUEUE_CAPACITY`, `TTS_DEFAULT_LANGUAGE`, `TTS_DEFAULT_SPEAKER_ID`, `TTS_DEFAULT_SPEED`, `TTS_DEFAULT_STEPS`, `TTS_DEFAULT_VOLUME`.
- `audio_playback` remains the only physical speaker owner; Phase 04 wires `tts_audio` playback consumption and `playback_state`.
- Phase 05 wires config/status/result transport through Zenoh/web authority.

## Unresolved Questions

- None for Phase 03. Supertonic redistribution legal approval remains a global external item.
