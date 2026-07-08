# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robo-Fleet is a distributed robotic rover control system with autonomous object tracking and visual servoing. Computation splits between:
- **Orchestra (Workstation)**: Web interface, fleet management, speech recognition, TTS
- **Rover-Kiwi (rover target)**: ML inference (YOLO, ReID, tracking), motor control, visual servoing. Phase 10 Docker verification runs this target on the current x86_64 workstation via `linux/amd64`; it is not ARM acceptance.

Communication: **Zenoh** (pub/sub) for cross-machine data, **Dora** framework for local node orchestration.

## Build Commands

```bash
cargo build --release              # All nodes (production)
cargo build --release -p <package> # Specific node
cargo build                        # Dev build (faster)
cargo test                         # All tests
cargo test -p <package>            # Specific package tests
```

## Running the System

**Start order matters**: Orchestra first, then Rover. Zenoh peer discovery takes 1-2 seconds.

```bash
# Orchestra (Workstation)
dora up && dora start orchestra/orchestra-dataflow.yml --name orchestra --attach

# Rover
dora up && dora start rover-kiwi/rover-kiwi-dataflow.yml --name rover-kiwi --attach
```

### Docker (alternative)

```bash
make models              # Download ML models first
make build-orchestra     # Build x86_64 image
make build-rover         # Build ARM64 image (native)
make build-rover-cross   # Build ARM64 from x86_64
make up-orchestra        # Start orchestra container
make up-rover            # Start rover container
make down                # Stop all
make status              # Check native Dora node status outside the phase 10 container flow
```

For the current workstation amd64 verification workflow, use:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
docker compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.workstation.yml \
  --profile mongodb --profile orchestra --profile rover-kiwi \
  up -d --build
```

### Web UI

The web UI is a **Turborepo pnpm monorepo** at `robo-control-app/`:

```
apps/web/       - Vite browser app (port 25010)
apps/native/    - Tauri v2 desktop app (port 1420)
packages/ui/    - Shared React components (Atomic Design: atoms/molecules/organisms/features/templates/pages)
packages/shared/ - Pure TypeScript types (zero deps)
```

```bash
cd robo-control-app
pnpm install
pnpm dev          # All apps
pnpm dev:web      # Web only (http://localhost:25010)
pnpm dev:native   # Tauri only (http://localhost:1420)
pnpm build
pnpm check-types
pnpm lint
```

Tech: React 19, Vite 7, Tailwind CSS v4 (CSS-first, no tailwind.config.js), Socket.IO, shadcn/ui, TypeScript 5.8.

### PyBullet Simulation

```bash
cd orchestra/pybullet_sim
pip install -e .         # or ./quickstart.sh
# Configure in orchestra-dataflow.yml (currently commented out)
```

Physics simulation at 240 Hz with mecanum wheel kinematics and 6-DOF arm. Env vars: `URDF_PATH`, `GUI_ENABLED`, `REAL_TIME`, `PHYSICS_TIME_STEP`, etc.

## Monitoring & Debugging

```bash
dora list                                          # Running native/local dataflows
dora logs <dataflow-name>                          # Dataflow logs
dora logs <dataflow-name> <node-name>              # Specific node logs
dora graph rover-kiwi/rover-kiwi-dataflow.yml --open  # Visualize graph
```

For the current Docker verification flow, prefer `docker ps`, container
healthchecks, `docker top`, and `docker logs`. `dora list` is not a reliable
in-container status probe for these images.

## Architecture

### Critical Design Decision: ML on Rover

Object detection, ReID, and tracking run ON THE ROVER (not orchestra) for:
- **Low latency**: Visual servoing needs <5ms tracking; network adds 10-20ms
- **Autonomy**: Rover tracks/follows even if network drops
- **CMC**: BoTSORT Camera Motion Compensation requires local frame access

**Vision pipeline** (all on rover, single process):
`kornia_capture` → [internal: YOLO → ReID → BoTSORT+CMC] → `visual-servo-controller` (PID)

`object_detector`, `reid_extractor`, `object_tracker` are library crates consumed by `kornia_capture`.
ML pipeline is lazy-loaded and gated by `TrackingCommand::Enable/Disable`. Default: camera-only (zero ML overhead).

### Zenoh Bridge Split

Two independent bridges handle cross-machine communication:

1. **Orchestra Bridge** (`orchestra/zenoh_bridge/`): Subscribes to `rover/{entity_id}/*`, publishes `rover/{entity_id}/cmd/*`. Env var `ACTIVE_ROVERS` controls multi-rover subscription.
2. **Rover Bridge** (`rover-kiwi/zenoh_bridge/`): Publishes sensor data/ML results, subscribes to commands. Env var `ENTITY_ID` identifies the rover.

### Shared Library (`robo_rover_lib`)

All cross-node data types live here. Key types in `src/types/`:
- `rover_types.rs`: `RoverCommand` (Legacy, Velocity, JointPositions, Stop), `RoverTelemetry`
- `arm_types.rs`: `ArmCommand` (JointPosition, CartesianMove, Home, Stop, EmergencyStop)
- `arm_telemetry.rs`: `ArmTelemetry` (end_effector_pose, joint_angles, joint_velocities)
- `detection_types.rs`: `BoundingBox`, `DetectionResult` (with `reid_features`, `tracking_id`)
- `fleet_types.rs`: `FleetSubscriptionCommand`, `ActiveRoversStatus`, `FleetRosterUpdate`, `RoverStatus`
- `mod.rs`: `CommandMetadata`, `CommandPriority`, `InputSource`, `CompleteJointState` (9 joints: 3 wheels + 6 arm)

When adding new commands/telemetry:
1. Define types in `robo_rover_lib/src/types/`
2. Update Zenoh bridge topic mappings in both bridges
3. Add Socket.IO event in `orchestra/web_bridge/src/main.rs`
4. Update TypeScript types in `robo-control-app/packages/shared/src/types/`
5. Add Dora input/output in the relevant dataflow YAML

### Command Priority System

Rover controller uses priority-based arbitration:
- **Emergency** (0): Emergency stop
- **Normal** (2): Manual web UI commands
- **High** (3): Visual servo commands
- **Low** (1): Voice commands

Higher priority overrides lower.

### Additional Nodes (not in main dataflows)

- **`rover-kiwi/sim_interface/`**: urdf-viz HTTP client (`http://127.0.0.1:7777`), outputs arm/rover telemetry. Currently commented out in rover dataflow.
- **`rover-kiwi/dispatcher_keyboard/`**: Dev tool for keyboard control (WASD+QR for rover, IJKL+UO for arm). Not in any dataflow YAML.
- **`orchestra/pybullet_sim/`**: Python PyBullet physics sim. Currently commented out in orchestra dataflow.

Note: `object_detector`, `reid_extractor`, `object_tracker` are now library crates, not Dora nodes. They are compiled into `kornia_capture`.

## Key Configuration

### AI Models

Models are cached in `models/.cache/{yolo,reid,sherpa-onnx}/`, and native x86
ONNX Runtime is managed under `models/.runtime/`. Use `make models`:

```bash
make models
make check-models
```

**ONNX Runtime**: Current rover vision crates are pinned to Rust `ort` `1.16.3`,
so use an ONNX Runtime `1.16.x` shared library and set `ORT_DYLIB_PATH` in the
dataflow YAML or via `ROVER_ORT_DYLIB_PATH`.

### Environment Variables (Dataflow YML)

All node configuration is via environment variables in the dataflow YAML files. Key tuning parameters:

- **Object Tracker** (on `kornia_capture` node): `IOU_THRESHOLD` (0.3), `REID_WEIGHT` (0.8), `REID_THRESHOLD` (0.5), `MAX_TRACKING_AGE` (50), `MIN_HITS` (3), `ENABLE_CMC` (true)
- **Object Detector** (on `kornia_capture` node): `CONFIDENCE_THRESHOLD`, `NMS_THRESHOLD`, `ORT_DYLIB_PATH`
- **ReID Extractor** (on `kornia_capture` node): `REID_MODEL_PATH`, `ORT_DYLIB_PATH`
- **Visual Servo**: `LATERAL_PID_KP/KI/KD`, `LONGITUDINAL_PID_KP/KI/KD`, `MIN_DISTANCE`, `MAX_VELOCITY`, `MAX_ANGULAR_VELOCITY`, `DEAD_ZONE`, `TARGET_BBOX_HEIGHT`
- **Web Bridge**: `MODE` (orchestra/standalone), `SOCKET_IO_PORT` (3030), `MONGODB_URI` (mongodb://localhost:27017), `MONGODB_DATABASE` (db), `JWT_SECRET` (auto-generated if unset), `ALLOW_DEFAULT_CREDENTIALS` (false), `SESSION_TTL_SECONDS` (3600), `FLEET_ROSTER`, `SELECTED_ENTITY_ID`, rate-limiting vars (`RATE_LIMIT_AUTH_PER_MINUTE`, `RATE_LIMIT_COMMANDS_PER_SECOND`)
- **Camera** (`kornia_capture` node): `SOURCE_TYPE` (webcam/rtsp), `SOURCE_URI` (/dev/video0 or rtsp://...)

### Socket.IO Events

**Web UI -> Backend**: `rover_command`, `tracking_command`, `camera_control`, `audio_control`, `tts_command`, `audio_stream`
**Backend -> Web UI**: `video_frame` (JPEG), `tracked_detections`, `servo_telemetry`, `speech_transcription`, `performance_metrics`, `tracking_telemetry`

## References

- [Dora Framework](https://github.com/dora-rs/dora) | [Zenoh Protocol](https://zenoh.io)
- [YOLOv12](https://github.com/ultralytics/ultralytics) | [BoT-SORT](https://arxiv.org/abs/2206.14651) | [OSNet](https://arxiv.org/abs/1905.00953)
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) | [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)
- Architecture details: `ARCHITECTURE.md`
