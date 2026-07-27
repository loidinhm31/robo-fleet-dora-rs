# Central Sherpa VAD STT Research Synthesis

Date: 2026-07-02

## Sources Reviewed

- `plans/reports/brainstorm-260702-1750-central-sherpa-stt.md`
- `orchestra/central_speech_recognizer/src/main.rs`
- `orchestra/orchestra-dataflow.yml`
- `orchestra/zenoh_bridge/src/main.rs`
- `common/web_bridge/src/main.rs`
- `orchestra/command_parser/src/main.rs`
- `robo_rover_lib/src/types/{audio_types,pcm_frame_packet,speech_types}.rs`
- `/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/`
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/{shared,ui}/`

## Current State

- Central recognizer uses `whisper-rs`, one global 5-second F32 buffer, energy thresholding, forced English, and fabricated confidence.
- Central input is only `web-bridge/voice_command_audio`; the browser worklet never forwards its posted messages.
- Rover audio already reaches Orchestra as validated 16 kHz mono S16LE with `entity_id`, `stream_id`, `frame_id`, and capture timestamp.
- Orchestra bridge outputs all active rover audio on one Dora output, so recognizer state must be separated by source metadata.
- Web bridge has a dormant authenticated `voice_command_audio` handler but uses an unbounded queue and loses browser stream identity.
- Parser-origin commands currently route to the UI-selected rover, which is unsafe for rover-origin speech.
- UI has separate Voice Commands and Speech Transcription surfaces, but both assume numeric confidence and neither understands STT status/source.

## Sherpa API Findings

- Pin official `sherpa-onnx` Rust crate `1.13.3`; reference examples use `default-features = false` plus `static` or `shared`.
- `OfflineRecognizer` shares one model across short-lived offline streams.
- `VoiceActivityDetector` requires independent state per audio source and supports `accept_waveform`, `detected`, `front/pop`, `flush`, and `reset`.
- Silero VAD uses fixed 512-sample windows at 16 kHz in the official examples.
- `LinearResampler` can normalize browser sample rates to 16 kHz before VAD.
- Offline result exposes text/tokens/timestamps, not a comparable confidence score.
- Sherpa recognizer, stream, VAD, and resampler wrappers implement `Send`/`Sync`, allowing a bounded decode worker.

## Selected Design

- One startup-only global profile: `en-vad-offline` or `vi-vad-offline`; default English.
- Silero VAD plus offline Zipformer only. No online backend, runtime profile switching, or partial results.
- Separate session key per rover `(entity_id, stream_id)` and per browser `stream_id`.
- One bounded worker owns the offline recognizer; event path owns input validation, resampling, sequencing, and VAD state.
- Browser capture uses explicit start/audio/stop messages. Web bridge owns stream-to-socket and stream-to-target mappings.
- Browser target is the authoritative selected rover captured at stream start. Rover target is its source `entity_id`.
- One central transcription contract carries source and target. Web bridge emits browser results only to the owner and rover results to all clients.
- Parser propagates target as Dora metadata. Orchestra bridge rejects parser commands without a valid active target.
- Remove automatic parser TTS feedback. Keep manual Web UI TTS.

## Model Bundles

- Silero: `silero_vad.onnx`
- English: `icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04`
- Vietnamese: `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09`

## Primary Risks

- Per-source VAD instances consume memory; active sources remain bounded by authenticated browser sessions and active rover subscriptions.
- Offline decode can block audio ingestion; bounded worker and non-blocking submission prevent this.
- Browser F32 JSON audio is less efficient than binary PCM but acceptable for the current single-browser command path; binary migration is deferred.
- Manual rover TTS can still be heard by rover STT; automatic parser feedback is removed and full playback suppression remains backlog.
- UI is a separate Git worktree/repository and requires coordinated contract deployment.

## Unresolved Questions

None. Prior planning interview resolved scope, profiles, visibility, routing, and TTS behavior.
