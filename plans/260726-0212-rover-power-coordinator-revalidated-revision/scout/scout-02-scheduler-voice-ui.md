# Scout 02 — scheduler / voice / UI

## Orchestra recording scheduler

- `orchestra/recording_scheduler/src/node_loop.rs`: Dora node boundary; inputs `recording_schedule_command`, `recording_schedule_query`, `recording_scheduler_recorder_feedback`; outputs `recording_scheduler_status`, `recording_schedule_command_result`, `recording_schedule_snapshot`, `recording_scheduler_manual_suppression_ack`. Main event/retry/reconciliation loop (`run`, due processing, recorder feedback, timeout/metrics).
- `orchestra/recording_scheduler/src/runtime.rs`: `SchedulerRuntime<C>`; occurrence/group hydration, due-window evaluation, acquire/release intents, feedback application, manual suppression and retry transitions.
- `domain.rs`, `state_machine.rs`, `node_intents.rs`: `RecordingGroup`, legal `RecordingOccurrenceState` transitions, `ScheduledRecordingIntent` creation and transient reason handling.
- `service.rs`, `mongo_repository.rs`, `mongo_documents.rs`, `node_persistence.rs`, `runtime_groups.rs`, `recurrence.rs`: authenticated schedule CRUD/validation, Mongo collections (`recording_schedules`, `recording_scheduler_groups`, `recording_scheduler_outbox`), persistence/recovery, recurrence/grouping.
- Tests: `orchestra/recording_scheduler/tests/{runtime-reconciliation.rs,recurrence.rs,mongo-recovery.rs,mongo-integration.rs}`.

## Orchestra bridge / voice transport

There is no `orchestra/web_bridge` directory in this checkout. The relevant bridge is `orchestra/zenoh_bridge/src/main.rs`: Socket/Dora input routing for `audio_stream_web` (publishes `rover/{entity}/cmd/audio_stream`), `lifecycle_wake_lease_authorized` (publishes lifecycle wake lease), voice config fanout, and rover subscriptions for `voice_status` / `voice_result`; topic helpers around lines 1171–1180 and publish helpers around 1275–1301. Existing tests assert parser exclusion and voice topic names (around 1494–1508, 1638–1649).

## Rover audio / edge voice

- `rover-kiwi/audio_capture/src/main.rs`: cpal input node, `AudioControl::{Start,Stop}`, `CaptureGate`, `LifecycleGate` quiesce/resume handling, lifecycle status, fresh-start requirement after resume, PCM output. `capture_gate.rs` owns `CaptureGate`/metrics and start/stop buffering policy; `audio_dump.rs` is optional bounded WAV dumper; `signal_metrics.rs` preflight levels.
- `rover-kiwi/audio_playback/src/runtime.rs`: playback node lifecycle gate, TTS/walkie input dispatch, device open/close and telemetry; `arbiter.rs` (`SourceArbiter`) and `tts-arbiter.rs` (`TtsArbiter`) enforce walkie preemption/TTS completion/failure; `buffers.rs` (`PlaybackBuffers`) queues and counters; `state.rs` emits `PlaybackState`.
- `rover-kiwi/edge_voice/src/runtime.rs`: edge TTS lifecycle/config/command handling, queue dispatch, worker drain, paced PCM output and status/result/metrics. Key functions include `handle_lifecycle_command`, `handle_tts_command`, `dispatch_if_idle`, `drain_worker_events`, `emit_status`, `emit_result`; `queue.rs` has priority/eviction and emergency clear; `protocol.rs` parses commands and emits `VoiceStatus`/result metadata; `worker.rs` runs Supertonic synthesis and `PcmChunker`.
- Voice/audio tests: playback `tests/dataflow-queue-policy.rs`, `src/{arbiter-tests.rs,buffers-tests.rs,resampler-tests.rs}`; scheduler tests above; edge voice has extensive inline runtime/worker/queue tests.

## External UI checkout (`/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`)

- Lifecycle/policy/status foundation: `packages/shared/src/types/lifecycle.ts`, `lifecycle-fixtures.ts`, `socket.ts`; `packages/ui/src/hooks/use-fleet-resource-state.ts` and `lib/fleet-resource-state.ts`; `components/features/FleetResources.tsx` (stale-resource lock, authoritative lifecycle controls) and `apps/web/e2e/fleet-resources-lifecycle.spec.ts`.
- Wake/live voice: `packages/ui/src/components/features/voice-command-panel.tsx` (authoritative voice status, captured target, bounded private transcript history), `voice-config-card.tsx`, `voice-alert-region.tsx`; hooks `use-browser-voice-capture.ts`, `use-audio-stream.ts`; page wiring/history in `components/pages/RoboRoverControl.tsx`; shared contracts `packages/shared/src/types/voice.ts`, `voice-tts.ts`.
- Recording scheduler/history: shared `recording-schedule.ts`, `recording.ts`; stores/hooks `use-recording-schedule-store*`, `use-recording-store*`; feature pages/components `recording-scheduler-page.tsx`, `recording-schedule-{list,editor}.tsx`, `recording-occurrence-status.tsx`, `recording-session-control.tsx`, `recording-clip-browser.tsx`, `recording-playback-panel.tsx`; e2e `apps/web/e2e/{recording-scheduler,recording-control}.spec.ts`.
- Existing tests specifically cover authoritative wake/lifecycle and history: `FleetResources.test.tsx` (scheduled wake + stale lock), `voice-command-panel.test.tsx`, `RoboRoverControl.test.tsx`, `TranscriptionDisplay.test.tsx`, `use-fleet-resource-state.test.tsx`, `edge-voice-live.spec.ts`.

## Revalidation mismatches / gaps against `docs/power-coordinator-architecture.md`

- Doc says UI talks to a dedicated power coordinator and Mongo has `power_lifecycle_events` / `power_current_state`; current Rust bridge has lifecycle lease routing but no scheduler-to-coordinator or power-history UI files found in this repo.
- Doc’s “UI Wake” exact-Rover authenticated command and local voice KWS bounded demand are not represented by a dedicated UI power-policy/demand contract; current UI lifecycle hooks target node lifecycle, while voice panel/browser capture handles transcripts/TTS.
- Doc requires live coordinator status authoritative and Mongo history timeline/filter; current UI has bounded in-memory/private speech history and recording history stores, but no `power_*` event projection consumer identified.
- Doc invariant says voice wake never executes recording/tracking/motion; current bridge still routes generic `audio_stream_web`, voice config, and existing command parser paths—policy enforcement must be verified at coordinator boundary.
- Doc says fresh restart starts Awake and transient demands are discarded; capture/playback/edge_voice lifecycle gates do enforce quiesce/resume and fresh audio start, but no power-coordinator restart state was found.

Unresolved: exact power-coordinator package/dataflow and web-bridge Socket.IO event names may exist on another branch or untracked checkout; this branch contains no `power_coordinator`/`web_bridge` path.
