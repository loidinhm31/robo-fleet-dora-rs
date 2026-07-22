# Lifecycle control research: safe soft-stop

## Conclusion

Use an application-level desired-state protocol, not `SIGSTOP` and not Dora process suspension. Keep a small control/safety spine always running; ask workload nodes to quiesce, drain, close devices, and explicitly unload large models. Existing Start/Stop gates are a useful base, but there is no acknowledgement, persisted desired state, or node lifecycle contract today.

`performance_control` is only a global web-delivery boolean in `common/web_bridge/src/main.rs:1676-1685`. It neither reaches Orchestra nodes nor a rover, is unauthenticated/unthrottled, and cannot establish that work stopped. Do not extend this event into lifecycle control.

## Existing control and topology

- `CameraControl`, `AudioControl`, and `StreamControl` already expose Start/Stop; stream also has Pause/Resume (`robo_rover_lib/src/types/video_types.rs:428-474`).
- Camera Stop closes and drops the camera; Start recreates it (`rover-kiwi/kornia_capture/src/main.rs:307-328`).
- Microphone Stop drops the CPAL stream and clears its buffer; Start recreates it (`rover-kiwi/audio_capture/src/main.rs:219-277`).
- `TrackingCommand::DisableDetection` disables detection and tracking (`robo_rover_lib/src/types/detection_types.rs:187-208`, `rover-kiwi/kornia_capture/src/vision_pipeline.rs:204-231`).
- Detection disable does **not** unload the detector/ReID/tracker. They remain in `VisionPipeline`; loading is only guarded by `Option` (`rover-kiwi/kornia_capture/src/vision_pipeline.rs:242-245`). The worker and pipeline live until shutdown (`rover-kiwi/kornia_capture/src/vision_worker.rs:143-190`).
- Orchestra already targets selected-rover web commands through `rover/{entity}/cmd/*` (`orchestra/zenoh_bridge/src/main.rs:1023-1033`). Targeted media control translates demand to camera, JPEG, and microphone commands (`orchestra/zenoh_bridge/src/main.rs:1036-1106`).
- Rover bridge is the always-on ingress and forwards camera/audio/tracking/stream commands to Dora (`rover-kiwi/zenoh_bridge/src/main.rs:599-622`).
- Rover static topology: resource monitor, camera/vision, encoder, servo, controllers, and bridge (`rover-kiwi/rover-kiwi-dataflow.yml:109-265`). Orchestra static topology: bridge, STT, parser, scheduler, recorder, web bridge (`orchestra/orchestra-dataflow.yml:43-316`).
- Fleet subscribe/deactivate only creates/drops Orchestra Zenoh subscribers; it does not stop rover work (`orchestra/zenoh_bridge/src/main.rs:173-245`).

## Recommended architecture

Control path:

`UI -> authenticated web_bridge -> Orchestra lifecycle coordinator -> local nodes + Orchestra Zenoh bridge -> rover/{id}/cmd/lifecycle/v1 -> rover lifecycle coordinator -> rover workload nodes`

Status path:

`node status -> coordinator -> rover/{id}/lifecycle/status/v1 -> Orchestra -> Socket.IO requesting client/authenticated viewers`

Prefer a small, always-on coordinator on each machine. On Orchestra this can initially be a module/task in `web_bridge`; on rover use a dedicated Dora node fed by the always-on Zenoh bridge. Keep transport routing separate from lifecycle policy. Never make the browser or Zenoh delivery itself authoritative.

Add shared versioned types in `robo_rover_lib/src/types/`:

- `LifecycleCommand { protocol_version, request_id, scope, desired_state, resource_groups, expected_revision, issued_at_ms, expires_at_ms, actor }`
- `desired_state`: `Running | Quiesced`; expose safe resource groups/profiles, not arbitrary process names.
- `LifecycleCommandResult { request_id, accepted, reason, revision }` for admission.
- `LifecycleStatus { entity_id, revision, desired_state, effective_state, components, updated_at_ms, failed_components }` where effective state is `Running | Quiescing | Quiesced | Resuming | Degraded | Failed`.
- Per-component state includes `Running | Draining | Quiesced | Unloading | Unloaded | Starting | Failed | Unsupported` plus last error.

Semantics:

1. Snapshot explicit target entity in web bridge; never retarget an in-flight request when fleet selection changes.
2. Validate session, authorization, target membership, profile, TTL, and revision; rate-limit with the existing command limiter pattern (`common/web_bridge/src/security.rs:99-140`). Audit actor/request/target/result without secrets.
3. Coordinator records desired state/revision, returns `Accepted`, and converges nodes. Same `request_id` returns the same result; same desired state is a no-op success. Reject stale `expected_revision`.
4. UI changes from “pending” only after authoritative status; show partial/degraded nodes and last-seen age. Never infer pause from missing metrics.
5. Publish periodic state plus transition events. On reconnect, Orchestra resends/query-reconciles desired state; rover reports its current revision. Command TTL prevents an old queued Pause/Resume from applying late.
6. Resume in dependency order and do not replay buffered commands/media. Rover remains stopped until a new explicit movement/arm command.

## Pause safety classification

### Must remain running

- Orchestra: `web-bridge` control/auth endpoint and `orchestra-bridge` transport. If scheduled recording remains supported while idle, `recording-scheduler` must also stay running.
- Rover: `zenoh-bridge`, lifecycle coordinator, `rover-controller`, `arm-controller`, and a lightweight resource/health monitor.
- Hardware watchdog/emergency-stop path (not evident in current dataflow) must be independent and always active.

Controllers stay alive so Stop, emergency commands, heartbeats, and resume remain possible. A process frozen by `SIGSTOP` cannot drain queues, acknowledge state, release memory/devices, service emergency commands, or reconnect.

### Safe to quiesce with contracts

- Rover camera/vision and JPEG encoder: first disable tracking/servo, command zero motion and confirm controller stopped, stop view output, then close camera and unload inference sessions. Encoder naturally becomes idle when input stops; add a status contract rather than freezing it.
- Rover microphone/audio converter: stop capture and drain/discard buffers. Playback/edge voice may quiesce only after cancel/drain policy and terminal results for accepted TTS; never strand callers.
- Orchestra central STT and command parser: stop new admission, finish/cancel bounded in-flight speech, drain queues, unload recognizer model. Parser is negligible memory; pausing it alone has little value.
- Orchestra media recorder: pause only when no active/finalizing session, or explicitly stop/finalize first. Keep scheduler running or explicitly suppress schedules with durable acknowledgement.

Expose profiles such as `Vision`, `Voice`, `Media`, and `FullIdle`, backed by a server-side allowlist/dependency graph. Individual-node controls invite unsafe combinations and couple UI to deployment IDs.

## Transition ordering

Rover quiesce: reject new workload -> send tracking disable -> command rover Stop and arm safe hold/Stop -> await controller confirmation -> cancel/drain TTS/audio -> stop microphone/view/camera -> unload ML -> report terminal state.

Rover resume: initialize coordinator dependencies -> reopen requested devices -> load models lazily on first feature enable -> report Running. Keep actuators stopped.

Orchestra quiesce: reject new STT/recording admission -> resolve in-flight work -> finalize recorder -> unload STT/model resources -> retain web/bridge/scheduler/control state. Resume reverses dependencies and restores admission only after readiness.

Use per-step deadlines. Failure to quiesce one component yields `Degraded`, not a false global `Quiesced`. Best-effort rollback is profile-specific; safety Stop remains latched regardless.

## Memory/resource reality

- Closing camera/microphone releases device handles and stops CPU/I/O; clearing buffers reduces live allocations.
- Disabling output or stopping upstream makes encoder/bridges idle but usually leaves process RSS allocated.
- Current vision Disable retains ONNX sessions/models, so it saves compute but not the main ML memory.
- Meaningful memory reclamation requires explicit `drop()` of detector/ReID/tracker/session/buffers and rebuilding lazily. Allocator RSS may still not fall immediately even after drops.
- Process exit under an external supervisor is the strongest memory reclaim, but needs a separate always-on supervisor, restart/readiness protocol, and safety ownership. Treat it as phase 2 only if measured application-level unload misses the memory target.

## Risks and tests

- Lost/duplicated/reordered commands: request ID, revision CAS, TTL, periodic reconciliation, idempotency tests.
- Partial network failure: rover continues last durable desired state; Orchestra shows stale/degraded, never assumes success.
- Resume storms/model OOM: bounded concurrency, readiness timeout, lazy model load.
- Unsafe queued actuator commands: clear queues at quiesce boundary; reject commands while quiesced; require fresh post-resume commands.
- Active recording/TTS corruption: explicit drain/cancel/finalize result contracts and integration tests.
- Authorization: lifecycle is privileged; require valid session plus operator/admin capability, command rate limit, entity allowlist, and audit trail. Current performance handler lacks all of these (`common/web_bridge/src/main.rs:1676-1685`).
- Validate CPU, RSS/PSS, device closure, command latency, emergency-stop behavior, lost-link recovery, stale command rejection, and repeated Pause/Resume across every profile.

## Suggested delivery phases

1. Typed lifecycle protocol, authenticated UI command, target-safe routing, rover/local coordinators, acknowledgements/status; only existing camera/audio/tracking gates.
2. Safety sequencing and node lifecycle adapters; STT/vision explicit unload/reload; recorder/TTS drain contracts.
3. Measure actual CPU/RSS/PSS savings and startup latency. Add supervised process stop/restart only for nodes that still miss an agreed memory target.

## Unresolved questions

- Is pause desired state persistent across rover/coordinator restart, and should boot default to Quiesced or Running?
- Must scheduled recordings wake a quiesced Orchestra/rover automatically, or should pause suppress schedules?
- What actuator-safe arm state is required: hold torque, brake, home, or power-off?
- Which roles may pause an entire rover, and may one client override another client's desired state?
- Required memory target and maximum resume latency per profile/device?
