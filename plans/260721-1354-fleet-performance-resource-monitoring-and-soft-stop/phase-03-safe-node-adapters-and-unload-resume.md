# Phase 03 — Safe Node Adapters and Unload/Resume

## Context links

- [Parent plan](./plan.md)
- [Lifecycle research](./research/researcher-01-lifecycle-control.md)
- Rover nodes: `rover-kiwi/rover-kiwi-dataflow.yml`
- Orchestra nodes: `orchestra/orchestra-dataflow.yml`

## Overview

- Date: 2026-07-21
- Priority: P1
- Implementation status: Done (2026-07-22)
- Review status: Approved after targeted re-review
- Description: implement application-level quiesce/unload/resume only for safe workloads.

## Key Insights

- Camera/microphone Stop already release devices; tracking Disable retains ONNX sessions/models.
- Freezing processes cannot acknowledge, drain, unload, or service emergency commands.
- Dropping model objects releases ownership but allocator RSS may not immediately fall.

## Requirements

- Node pause hooks stop admission, resolve bounded in-flight work, clear queues/buffers, close devices, unload models, then acknowledge.
- Resume reconstructs resources safely/lazily and never replays pre-pause commands.
- Any actuator-affecting workload issues/observes safe Stop before quiescing.
- Pause closes admission first, then cancels/finalizes active recording, synthesis/playback, or STT through an explicit terminal contract before teardown.
- Supported capabilities are node-specific; unimplemented nodes report `Unsupported`.

## Architecture

Initial supported set:

- Rover `gst-camera`: disable tracking/servo, safe-stop autonomous motion, stop view/capture, shut worker, drop detector/ReID/tracker sessions/buffers; recreate capture and lazy-load ML on resume.
- Rover `audio-capture`: reuse stream drop/recreate; clear buffers.
- Rover `edge-voice` and `audio-playback`: stop accepting work, terminalize accepted synthesis/playback as `interrupted_by_lifecycle`, stop/flush playback, shut worker/device, and unload TTS model; resume lazy. `audio-converter`/`video-encoder` drain through upstream cancellation and report state without process freeze.
- Orchestra `central-speech-recognizer`: close active browser/rover streams with lifecycle-cancel status and no partial transcript, close decoder worker, and drop VAD/recognizer models; resume/load with readiness status.
- Orchestra `media-recorder`: stop admission, request graceful stop, and wait for the current clip to report finalized or failed before teardown. Never silently discard a valid active clip. Scheduler remains running.

Always-on/locked: web/Zenoh bridges, lifecycle/resource managers, scheduler, command parser, rover/arm controllers, emergency/watchdog path.

## Related code files

- Modify: `rover-kiwi/kornia_capture/{main.rs,vision_worker.rs,vision_pipeline.rs}`.
- Modify: `audio_capture`, `edge_voice`, `audio_playback`, and status wiring/dataflows.
- Modify: `orchestra/central_speech_recognizer` runtime/startup and `media_recorder` control loop.
- Modify: rover/arm/visual-servo command routing for safe-stop sequencing and confirmation.
- Create focused lifecycle adapter modules beside each runtime where needed; avoid generic unsafe signal control.

## Implementation Steps

1. Add lifecycle input/status output to supported nodes; filter exact target/revision.
2. Implement shared gate helpers for admission and terminal status, keeping resource teardown node-owned.
3. Implement camera/vision ordered stop, worker shutdown, model drop, buffer clear, lazy restart.
4. Implement audio/voice cancellation/result integrity, playback flush, device/model teardown, and restart.
5. Implement central STT stream cancellation, admission gate, worker/model teardown, and readiness-controlled resume.
6. Implement recorder graceful cancellation/finalization with a bounded terminal result before teardown.
7. Add manager dependency ordering and deadlines. A cancellation/finalization failure must not report `Quiesced`; keep admission closed and return terminal `Degraded`/`Failed` for explicit recovery.
8. Reject movement/arm/media/audio commands addressed to quiesced workloads; require fresh post-resume commands.
9. For every scheduled recording occurrence/resource target, derive a deterministic lease ID and acquire it only after revalidating schedule revision/window, active rover, recorder health/capacity, storage, lifecycle capability/status, and media-demand authority; recheck before recorder start.
10. Wait for effective Running before recording. Refcount independent overlapping leases; make duplicate acquire/release idempotent, release each owner on every exit path, expire/reconcile orphaned leases after restarts, and reconcile final release to the latest user desired state. A later user Pause revokes automation leases, cancels/finalizes recording, and quiesces; a later Resume remains Running.

## Todo list

- [x] Vision adapter and unload tests
- [x] Audio/edge voice adapters
- [x] Central STT adapter
- [x] Recorder cancel/finalize adapter
- [x] Scheduler no-auto-wake policy (no lifecycle wake lease is emitted)
- [x] Safe-stop and dependency ordering
- [x] Resume/readiness/failure tests

## Success Criteria

- Supported node reaches Quiesced only after owned devices/workers/models are dropped.
- Controllers and emergency commands remain responsive throughout pause/resume.
- Accepted active work reaches its documented lifecycle-cancel/finalize terminal result; no operation is stranded and no failed cancellation is mislabeled Quiesced.
- Scheduled wake never overwrites user desired state, and final lease release deterministically reconciles the latest revision.
- Repeated pause/resume and mid-transition disconnect do not leak workers/devices or replay commands.

## Risk Assessment

- Model reload OOM/storm: one transition at a time, bounded startup, lazy load.
- Partial teardown: terminal `Degraded`, explicit component error, manager remains responsive.
- Allocator retains RSS: report measured state; hard process stop/restart remains out of v1.

## Security Considerations

- Node trusts only manager-origin Dora commands; manager validates external actor/target.
- Failure details sanitized; filesystem/model paths never reach UI.

## Next steps

- Phase 04 exposes capabilities and authoritative status in Fleet Resources UI.

## Unresolved Questions

- Define arm safe state (hold/brake/home/power-off) before actuator-adjacent pause acceptance.
- Define bounded transition/finalization deadlines and recovery after a terminal cancellation failure.
- Define scheduled-wake retry/backoff when per-occurrence validation or resume fails.
