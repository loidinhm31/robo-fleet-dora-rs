# Central Sherpa STT Current-State Research

Date: 2026-07-03

## Sources Reviewed

- `plans/reports/brainstorm-260702-1750-central-sherpa-stt.md`
- `plans/260702-2316-central-sherpa-vad-stt/`
- Commits `3d8c57c`, `bbbe251`, and `96b54e6`
- `ARCHITECTURE.md` and `docs/codebase-summary.md`
- `orchestra/central_speech_recognizer/`
- `common/web_bridge/src/main.rs`
- `orchestra/{command_parser,zenoh_bridge}/src/main.rs`
- `orchestra/orchestra-dataflow.yml`
- External UI repository at `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- [Official Sherpa-ONNX documentation](https://k2-fsa.github.io/sherpa/onnx/index.html)
- [Official Sherpa-ONNX repository](https://github.com/k2-fsa/sherpa-onnx)
- Local official Rust examples at `/mnt/data/ws/sherpa-onnx/rust-api-examples/`

## Verified Baseline

- Phase 01 contracts and architecture are committed.
- Phase 02 pinned `sherpa-onnx = 1.13.3`, model catalogs, downloads, and startup loading are committed.
- Phase 03 dual-source VAD/offline runtime is committed.
- Central rover input requires mono 16 kHz S16LE with exact metadata/payload validation.
- Central browser input supports bounded sample-rate validation and stateful resampling.
- Session isolation key is `(entity_id, stream_id)` for rover and `stream_id` for browser.
- Offline decode uses a bounded nonblocking worker; confidence remains absent.
- Current dataflow connects rover audio and legacy browser audio, but lacks browser lifecycle control and status request edges.
- Web bridge still owns legacy unbounded browser audio buffering and broadcasts all transcriptions.
- Parser outputs still lose target metadata; bridge parser channels still use selected-rover fallback.
- UI contracts partly landed, but browser worklet forwarding and source-separated rendering remain incomplete.

## Superseding Decisions

The original brainstorm proposed online English, runtime switching, and per-rover profiles. The newer committed plan and architecture intentionally narrow scope:

- Profiles: `en-vad-offline` and `vi-vad-offline` only.
- Selection: global startup-only `STT_PROFILE`.
- Output: final utterances only; no partial event.
- Browser speech: private to origin socket; target captured at stream start.
- Rover speech: broadcast to authenticated clients; target source rover.
- Interpretation: deterministic parser only.
- TTS: remove automatic parser feedback; manual TTS remains with echo risk documented.

This residual plan preserves these newer decisions. Restoring online recognition or runtime switching requires a separate architecture decision and model/resource plan.

## Official API Fit

- Sherpa officially supports streaming ASR, non-streaming ASR, VAD, and Rust APIs.
- Current code uses the official offline recognizer, Silero VAD, and linear resampler patterns.
- Offline recognizer results do not justify a fabricated cross-model confidence value.
- Separate streaming/non-streaming capabilities exist upstream, but current product scope does not require both in one process.

## Remaining Delivery Gaps

1. Authenticated browser start/audio/stop ownership and bounded transport.
2. Authoritative status cache/request and reconnect behavior.
3. Private browser versus fleet rover transcript routing.
4. Required target propagation through parser and final Zenoh publication.
5. Functional browser worklet forwarding and source-separated UI.
6. Recorded/live concurrency, privacy, routing, language, and performance validation.
7. Conditional Whisper/GGML and disabled edge recognizer retirement.

## Risks

- Current worktree contains user changes in both dataflows; implementation must reconcile, not overwrite.
- UI is a separate repository and needs coordinated write access and atomic contract deployment.
- Decode queue and per-source VAD state need live fleet bounds, not unit-only confidence.
- Manual rover TTS remains audible to rover STT until playback suppression/AEC is implemented separately.

## Unresolved Questions

None. Scope follows the newer committed architecture and prior plan decisions.
