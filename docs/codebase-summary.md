# Codebase Summary

Snapshot date: 2026-07-17

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
- Web UI emits authenticated, rate-limited media demand. `MediaDemandRegistry` scopes each hold by rover, consumer, and resource (camera, JPEG, microphone), so one browser or future recorder cannot release another consumer's media.
- Browser demand pins its rover at acquisition and is migrated only for that browser on fleet selection; disconnect, expiry, idle cleanup, and process shutdown release owned demand idempotently.
- Effective `0 -> 1` / `1 -> 0` transitions become `TargetedMediaControl` messages. Orchestra validates the target against the active fleet and decomposes only changed resources into the existing exact-rover camera, stream, and audio topics.
- `robo_rover_lib::recording_types` provides version-1, validated JSON contracts for future file-session commands/statuses, clip queries/results, and playback tickets; file-session Socket/Dora names are reserved as `recording_session_*`.
- `kornia_capture` gates view publication plus worker frame submission; local capture continues even when ML/tracking is disabled.
- Rover dataflows set `DETECTOR_INTRA_THREADS=2` and `REID_INTRA_THREADS=1`.
- `audio_capture` auto-selects the preferred input device, assigns `stream_id`, `frame_id`, and `capture_timestamp_ms` to F32 frames, and records signal-level observability for silence detection.
- Rover `audio_converter` converts capture F32 to S16LE while preserving capture identity.
- Rover Zenoh publishes versioned PCM v1 packets; orchestra validates them and only accepts bounded legacy F32LE during rollout.
- Orchestra forwards S16LE directly to `web_bridge`, which emits binary audio attachments to browsers.
- `orchestra/media_recorder` is the Phase 2 Dora node for validated rover JPEG and PCM ingestion. It reads `RECORDING_ROOT`, optional `FFMPEG_PATH`/`FFPROBE_PATH`, and `RECORDING_*` limits; uses one bounded FIFO queue where a new video frame replaces the oldest queued video, or the oldest queued audio if no video is queued; writes `.partial/<recording_id>.mp4.partial` plus adjacent manifests; and atomically promotes finalized MP4/manifest pairs only after FFmpeg and ffprobe validation succeed.
- Phase 3 now wires `media-recorder` into `orchestra/orchestra-dataflow.yml`,
  exposes authenticated/rate-limited `recording_session_command`,
  `recording_clip_list`, and `recording_playback_ticket` Socket.IO events, and
  correlates recorder responses by request ID with reconnect status replay.
  `RECORDING_CONTROL_QUEUE_CAPACITY` and `RECORDING_REQUEST_TIMEOUT_SECONDS`
  bound bridge control state. Playback tickets are short-lived in-memory
  capabilities; `/recordings/playback/:ticket` streams finalized MP4 content
  with one-range `GET`/`HEAD` support after component-wise no-follow path
  authorization and file/manifest identity checks. Phase 4 still owns container
  deployment wiring around recorder output.
- Phase 5 adds the shared `MediaRecordingPage` with authenticated relative-path
  session controls, concurrent per-rover status cards, finalized clip browsing,
  short-lived ticket playback, reconnect/auth cleanup, and deterministic Vitest
  plus fake-Socket.IO Playwright coverage for desktop and mobile layouts.
- `recording_scheduler` persists private Mongo outbox snapshots for replay and
  acknowledgement, deterministically merges bridging overlap groups using the
  earliest `(planned_start, occurrence_id)` directory, and uses schedule
  supersession/tombstones to prevent cancelled future occurrences from reviving.
- The scheduler lifecycle reaches signed-in clients through authenticated
  `recording_occurrence_status` and `recording_scheduler_status` broadcasts.
  The scheduler UI is a third shared CONTROL/RECORDINGS/SCHEDULER view and is
  bound to the selected rover. Its normalized store uses authoritative
  snapshots, accepts only monotonic same-rover occurrence updates, and resets
  pending state before reconnect, authentication, or selected-rover resync.
  Schedule CRUD uses request IDs and revision compare-and-set; the UI does not
  make durable optimistic changes, and a conflict installs the server's current
  schedule before the user reapplies an edit. Manual recording remains usable
  when scheduler readiness is degraded.
- `central_speech_recognizer` now follows the Phase 01 STT contract: `SpeechTranscription` carries `source_kind`, `profile`, `target_entity_id`, `entity_id`, `stream_id`, `utterance_id`, `language`, `timestamp`, `duration_ms`, and optional `confidence`; `SttStatus` carries `state`, `profile`, `language`, `timestamp`, `error`.
- Authenticated browsers control STT streams with `voice_command_control` start/stop events and send ordered Float32 frames with `voice_command_audio`; the web bridge owns stream identity, snapshots the selected rover at start, and forwards bounded start/audio/stop messages to central STT.
- `central_speech_recognizer` has completed the Sherpa Phase 02 runtime cutover: it provisions fixed English/Vietnamese offline profile catalogs under `models/.cache/sherpa-onnx/asr`, validates required files, and loads Silero VAD plus the selected offline recognizer at startup.
- The live central STT decode loop accepts browser and rover streams, applies Sherpa VAD/offline recognition, and emits final-only transcriptions plus lifecycle status.
- Browser transcripts use the private `voice_command_transcription` event for their owning authenticated socket. Rover transcripts use authenticated `transcription` broadcasts. `stt_status` is cached and replayed on reconnect, with an explicit Dora status request when no cache exists.
- The UI now enforces the source split end to end: `VoiceControls` owns browser-private STT history and authoritative backend status/profile display, while `TranscriptionDisplay` shows only rover-origin fleet transcription with rover badges and null-safe confidence.
- Browser microphone transport lives in `packages/ui/src/hooks/use-browser-voice-capture.ts`. It emits exactly one start and one stop per stream, uses the actual `AudioContext` sample rate, forwards bounded Float32 frames in roughly 50 ms batches, flushes any partial frame before stop, and tears down media/worklet/context resources on stop, disconnect, unmount, mode switch, and deferred resume cancellation.
- Orchestra dataflow wires web-bridge `voice_command_audio`, `voice_command_control`, and `stt_status_request` outputs to central STT, and routes central `transcription` and `stt_status` outputs back to web-bridge.
- Browser STT transport limits are configured with `WEB_STT_QUEUE_CAPACITY`, `WEB_STT_STREAM_IDLE_SECONDS`, and `WEB_STT_CLOSING_SECONDS`.
- `web_bridge` maintains process-level `Arc<AudioDeliveryCounters>` (atomic, relaxed ordering) in `SharedState` so shutdown totals survive client disconnects; per-client `ClientState` counters remain for live debugging. Resolves the Approach A Phase 5 backlog item (Phase 06 completion report).
- Phase 05 added verification coverage for walkie/TTS pacing, queue policy assertions, and benchmark harness contract alignment; physical acoustic acceptance is still pending.

## Documentation Notes

- `ARCHITECTURE.md` is the main system reference.
- `README.md` is the quick-start and feature entry point.
- `SETUP_ENVIRONMENT.md` is the local dependency checklist.
