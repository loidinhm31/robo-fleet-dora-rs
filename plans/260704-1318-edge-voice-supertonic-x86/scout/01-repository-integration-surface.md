# Repository Integration Surface

Date: 2026-07-04
Status: complete

## Backend State

- `orchestra/kokoro_tts`: current workstation playback engine; direct CPAL/Rodio ownership through `kokoro-tiny`.
- `rover-kiwi/sherpa_tts`: disabled Piper/VITS node; `sherpa-rs`; direct Rodio speaker ownership.
- `rover-kiwi/audio_playback`: fixed-rate 16 kHz CPAL ring buffer for walkie audio.
- `rover-kiwi/audio_capture`: 16 kHz capture with manual start/stop but no playback-derived suppression.
- `robo_rover_lib/src/types/tts_types.rs`: basic command/priority types; no command UUID, runtime config, status, or result.
- Shared audio types already support F32LE/S16LE and validated sample metadata.
- Dora already transports Arrow `Float32Array` with metadata parameters in browser STT code.

## Transport State

- Web bridge receives authenticated `tts_command` and emits Dora output.
- Orchestra bridge targets selected rover on existing `rover/{id}/cmd/tts`.
- Rover bridge emits Dora `tts_command`, but default rover dataflow has no consumer.
- No global config fan-out, applied revision, voice status, or command result path.

## Models and Runtime

- `make models` calls `docker/scripts/download-models.sh`.
- Current cache includes retired GGML Whisper and Piper assets.
- `models/README.md`, root README, setup docs, and architecture contain stale Kokoro/Piper claims.
- Rover Docker installs ONNX Runtime 1.17.1 although vision crates target `ort 1.16.3` and repository guidance requires 1.16.x.

## Docker Redundancy Provenance

- `cmake`, `clang`, and `libclang-dev` are pulled by retired Kokoro/`sherpa-rs` native builds.
- Rover `binutils` stage exists only to inspect/copy possible `sherpa-rs` shared libraries.
- Orchestra runtime `libasound2` is not required by current non-Kokoro built binaries.
- Rover ALSA/GStreamer runtime packages and dynamic ONNX Runtime remain required.

## UI State

- Active UI is the adjacent Turborepo, not a backend subdirectory.
- `VoiceControls` sends `{ text }`, owns browser STT and walkie controls, and has regression tests.
- Shared socket types contain no TTS config/status/result contracts.
- Existing live Playwright suite covers video/audio streaming only.

## Validation Environment

```text
OS: Fedora Linux 44, x86_64
CPU: AMD Ryzen 7 8840U, 8 cores / 16 threads
Memory: 23 GiB
Container runtime: Podman 5.8.3 through Docker-compatible CLI
Dora CLI: 0.5.0
Rust: 1.88.0
Node: 24.16.0
pnpm: 9.1.0
```

## Unresolved Questions

- None; phase files define remaining implementation choices.
