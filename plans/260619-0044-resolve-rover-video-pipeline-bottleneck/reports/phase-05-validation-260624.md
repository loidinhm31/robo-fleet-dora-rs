# Phase 05 Validation Evidence

Validated on: 2026-06-25
Historical smoke evidence from: 2026-06-24
Plan: `plans/260619-0044-resolve-rover-video-pipeline-bottleneck`
Phase: Phase 05 Final Validation and Release
Status: Headless release validation complete; browser/field certification deferred.

## Environment

- Workspace: `/mnt/data/ws/sharing/robo-fleet-dora-rs`
- UI workspace: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Camera: `/dev/v4l/by-id/usb-Vimicro_corp._PC-LM1E_Camera_PC-LM1E_Audio-video-index0`
- Camera format observed: 640x480 at 30 FPS, MJPG/YUYV
- YOLO model: `models/.cache/yolo/yolo12n.onnx`
- ReID model: `models/.cache/reid/osnet_x0_25.onnx`
- ONNX Runtime: `/home/loidinh/.cache/sherpa-rs/.../lib/libonnxruntime.so`
- Container runtime: Podman Docker compatibility, rootless cgroup v2, CPU and memory controllers present
- Paths and local endpoints in this report are non-secret operational context for reproducing the workstation run.

## Headless Quick Gate

- `scripts/benchmark-rover-video-pipeline.sh preflight`
  - Passed camera, camera format, YOLO model, ReID model, ONNX Runtime, Dora, Docker info, and Docker smoke checks.
- `cargo test -p robo_rover_lib video_types`
  - Passed: 8 passed, 0 failed.
- `cargo test -p web_bridge`
  - Passed: 19 passed, 0 failed.
- `cargo test -p kornia_capture`
  - Passed: 18 passed, 0 failed.
- `cargo check -p robo_rover_lib -p web_bridge -p kornia_capture -p rover_zenoh_bridge -p orchestra_zenoh_bridge`
  - Passed with existing warnings only.
- `cargo test -p rover_zenoh_bridge -p orchestra_zenoh_bridge --no-run`
  - Passed: bridge test binaries compiled.
- `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app check-types`
  - Passed: 2 successful tasks.
- `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app lint`
  - Command succeeded, but no lint tasks were configured. This is command availability evidence, not lint coverage evidence.
- `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app build`
  - Passed: web and native builds replayed from cache and succeeded.
- `dora graph rover-kiwi/rover-kiwi-dataflow.yml`
  - Passed: graph generated.
- `dora graph rover-kiwi/rover-kiwi-direct-dataflow.yml`
  - Passed: graph generated.
- `dora graph orchestra/orchestra-dataflow.yml`
  - Passed: graph generated.
- `dora up && dora check`
  - Passed in unrestricted environment.
- `dora check --dataflow rover-kiwi/rover-kiwi-dataflow.yml`
  - Passed as Dora coordinator status check.
- `dora check --dataflow rover-kiwi/rover-kiwi-direct-dataflow.yml`
  - Passed as Dora coordinator status check.
- `dora check --dataflow orchestra/orchestra-dataflow.yml`
  - Passed as Dora coordinator status check.
- `docker info` and `docker run --rm hello-world`
  - Passed using Podman Docker compatibility.

## Native Direct Smoke

Command shape:

```bash
dora up
ROVER_CAMERA_URI=/dev/v4l/by-id/usb-Vimicro_corp._PC-LM1E_Camera_PC-LM1E_Audio-video-index0 \
ROVER_YOLO_MODEL_PATH=/mnt/data/ws/sharing/robo-fleet-dora-rs/models/.cache/yolo/yolo12n.onnx \
ROVER_REID_MODEL_PATH=/mnt/data/ws/sharing/robo-fleet-dora-rs/models/.cache/reid/osnet_x0_25.onnx \
ROVER_ORT_DYLIB_PATH=/home/loidinh/.cache/sherpa-rs/.../lib/libonnxruntime.so \
ALLOW_DEFAULT_CREDENTIALS=true \
JWT_SECRET=<smoke-secret-redacted> \
timeout 45 dora start rover-kiwi/rover-kiwi-direct-dataflow.yml --name phase05-direct-smoke --attach
```

Evidence:

- Dataflow output: `rover-kiwi/out/019efa51-3ec7-7261-a617-f7771cd204f7/`
- Timeout exit code `124` was expected for the bounded smoke window.
- `gst-camera`: webcam camera started successfully.
- `video_pipeline` metrics emitted for capture, JPEG encode, web receive, and web emit stages.
- Representative JPEG branch throughput: about 75-78 frames per 5 seconds, about 3.8-4.0 MB per 5 seconds.
- Web emit showed zero clients, so emitted count stayed zero and frames were dropped by demand/viewer state. This is expected without a browser client.

## Split Local Smoke

Default rover Zenoh config did not validate local split delivery because it connects to a fixed Tailscale endpoint:

- Rover config endpoint: `tcp/100.110.100.96:34969`
- Local orchestra listener observed during smoke: `tcp/192.168.1.73:7447`
- Result: rover published `rover/rover-kiwi/video/jpeg/v1`, but orchestra did not receive frames in the default local workstation setup.

Rerun used a temporary local rover Zenoh config at `/tmp/rover-zenoh-local-phase05.json5`:

```json5
{
  "mode": "peer",
  "connect": { "endpoints": ["tcp/192.168.1.73:7447"] },
  "transport": { "link": { "tx": { "lease": 10000, "keep_alive": 4, "batch_size": 8192 } } },
  "scouting": { "multicast": { "enabled": true, "address": "224.0.0.224:7446", "interface": "auto" } }
}
```

Evidence:

- Orchestra output: `orchestra/out/019efa55-4f74-7a94-ab0a-f3b2861cfcd5/`
- Rover output: `rover-kiwi/out/019efa55-5d13-70be-8b14-2f2e3446b10e/`
- Timeout exit code `124` was expected for the bounded smoke window.
- Rover bridge published `rover/rover-kiwi/video/jpeg/v1`.
- Orchestra bridge received `rover/rover-kiwi/video/jpeg/v1`.
- Representative rover publish metrics: 75-78 frames per 5 seconds, p95 publish time under 3 ms, publish age p95 about 31-37 ms.
- Representative orchestra receive metrics: 75-76 frames per 5 seconds, p95 receive time under 0.2 ms, receive age p95 about 29-36 ms after warmup.
- Web emit still had zero browser clients, so emitted count stayed zero and drops were expected.

## Deferred Field Certification

- Browser render validation was not run in this headless environment, so viewer FPS, browser capture-to-display p95, browser-side frame ID/timestamp preservation, and UI control toggles remain field certification tasks.
- Constrained rover container validation was not run with exact 3 CPU / 4 GiB cgroup evidence.
- 10-minute camera-only, detection-only, typical tracking, crowded corpus, stream toggle/disconnect, and Zenoh reconnect cases were not run.
- Failure matrix was not run: camera stop/start, viewer disconnect with browser, corrupt/oversized packet, slow browser decode, encoder error, worker failure, and resource pressure.
- 30-minute typical full-tracking soak was not run, so p50/p95/p99, bandwidth, CPU equivalents, RSS, cgroup throttling/OOM state, frames, drops, and control latency are not certified.
- Full tracking with servo contract remains a field certification task: servo >=10 Hz and input age p95 <=150 ms.
- Phase 04 live/constrained full-tracking milestone is deferred to field certification.

## Not Verified In Headless Gate

| Contract | Reason |
|---|---|
| Browser FPS and render age | No browser session in headless execution environment |
| Browser runtime payload confirmation | Protocol-level binary tests passed; runtime browser capture deferred |
| Exact 3 CPU / 4 GiB constrained rover soak | Requires representative field/container run |
| 10-minute scenario matrix | Requires stable camera and runtime setup |
| 30-minute full-tracking soak | Requires stable camera and runtime setup |
| Physical failure matrix | Disruptive hardware/browser actions are field certification tasks |

## Rollback

- Branch: `main`
- Revision used for headless closure: `928cb9e`
- Rollback action: revert this plan's coordinated rover/orchestra/UI video-pipeline changes to the previous accepted revision, or restore the last deployment image/config that published the prior video path.
- Configuration note: local split smoke required rover Zenoh to connect to the active local orchestra endpoint instead of the stale fixed Tailscale endpoint.

## Warnings Observed

- `robo_rover_lib`: `ForwardKinematics::link_lengths` is never read.
- `web_bridge`: `ClientState::jpeg_quality` is never read.
- `web_bridge`: `AuthErrorReason::TokenExpired` and `AuthErrorReason::RateLimited` are never constructed.
- `web_bridge`: `validate_detection_index` is never used.
- Audio nodes reported ALSA host-down warnings and entered silent/no-input mode during smoke runs. This is non-blocking for video pipeline validation.

## Worktree Notes

- Existing dirty file before validation: `.idea/workspace.xml`.
- Generated Dora graph HTML files were untracked artifacts and were removed after graph validation: `orchestra-dataflow-graph.html`, `rover-kiwi-dataflow-graph.html`, and `rover-kiwi-direct-dataflow-graph.html`.
- Adjacent UI repository has `.turbo/` after Turborepo checks.
- Temporary Mongo container `robo-phase05-mongo` was used for web bridge smoke startup and removed after validation. It should not be treated as a deployment dependency.

## Unresolved Questions

- None. Remaining work is deferred field certification, not an open design question.
