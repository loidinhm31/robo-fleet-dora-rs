# Codebase Summary

Snapshot date: 2026-07-01

## Scope

- Distributed rover control stack with Dora and Zenoh.
- Workstation/orchestra side for UI, speech, TTS, and bridge fan-out.
- Rover side for capture, ML inference, control, and rover-side JPEG view output.
- Shared Rust types in `robo_rover_lib`.
- Web UI in `robo-control-app`.

## Current Architecture

- Rover keeps capture, ML, and servo local; `kornia_capture` now isolates vision work in a dedicated worker.
- Main loop submits frames only when detection/tracking is enabled.
- Latest-frame slot is capacity-one, so fresh frames replace stale unprocessed frames.
- Worker results older than 150ms are dropped before publish.
- View/video output is throttled separately with `SOURCE_FPS` and `VIEW_STREAM_FPS`.
- Published video topic is `rover/{entity_id}/video/jpeg/v1`.
- Web UI receives `video_frame` as metadata plus binary JPEG bytes.
- Web UI emits authenticated, rate-limited `stream_control` demand.
- Web bridge aggregates demand and only forwards 0->1 / 1->0 transitions upstream.
- `kornia_capture` gates view publication plus worker frame submission; local capture continues even when ML/tracking is disabled.
- Rover dataflows set `DETECTOR_INTRA_THREADS=2` and `REID_INTRA_THREADS=1`.
- `audio_capture` auto-selects the preferred input device, assigns `stream_id`, `frame_id`, and `capture_timestamp_ms` to F32 frames, and records signal-level observability for silence detection.
- Rover `audio_converter` converts capture F32 to S16LE while preserving capture identity.
- Rover Zenoh publishes versioned PCM v1 packets; orchestra validates them and only accepts bounded legacy F32LE during rollout.
- Orchestra forwards S16LE directly to `web_bridge`, which emits binary audio attachments to browsers.
- `central_speech_recognizer` now follows the Phase 01 STT contract: `SpeechTranscription` carries `source_kind`, `profile`, `target_entity_id`, `entity_id`, `stream_id`, `utterance_id`, `language`, `timestamp`, `duration_ms`, and optional `confidence`; `SttStatus` carries `state`, `profile`, `language`, `timestamp`, `error`.
- Browser STT path still uses legacy transport (`voice_command_audio` / `transcription`) pending later phases; current browser runtime behavior is still transitional, not the final routed STT path.
- `central_speech_recognizer` consumes only web-microphone F32 audio and uses `ggml-base.bin`; the edge recognizer remains a disabled placeholder.
- `web_bridge` maintains process-level `Arc<AudioDeliveryCounters>` (atomic, relaxed ordering) in `SharedState` so shutdown totals survive client disconnects; per-client `ClientState` counters remain for live debugging. Resolves the Approach A Phase 5 backlog item (Phase 06 completion report).

## Documentation Notes

- `ARCHITECTURE.md` is the main system reference.
- `README.md` is the quick-start and feature entry point.
- `SETUP_ENVIRONMENT.md` is the local dependency checklist.
