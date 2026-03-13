# Researcher 02: Dataflow YAMLs, Dora Capabilities, robo_rover_lib

## rover-kiwi-dataflow.yml (14 Active Nodes)

### Pipeline
```
gst-camera (30fps tick) → object-detector (YOLOv12n) → reid-extractor (OSNet)
  → object-tracker (BoTSORT+CMC) → visual-servo-controller (PID)
                                                         ↓
rover-controller ←───────────────────────────────────────┘
arm-controller
audio-capture (50ms) → zenoh-bridge
audio-playback ← zenoh-bridge
performance-monitor (5s) → zenoh-bridge
sherpa-tts ← zenoh-bridge
zenoh-bridge (ENTITY_ID=rover-kiwi) ←→ [Zenoh network]
```

### Key Env Vars (all use `${HOME}` substitution)
- `MODEL_PATH`, `REID_MODEL_PATH` — ONNX model paths
- `ZENOH_CONFIG` — peer config JSON5
- `IOU_THRESHOLD=0.3`, `REID_WEIGHT=0.8`, `REID_THRESHOLD=0.5`
- `MAX_TRACKING_AGE=50`, `MIN_HITS=3`
- `LATERAL/LONGITUDINAL_PID_KP/KI/KD`, `MAX_VELOCITY`, `MAX_ANGULAR_VELOCITY`

## orchestra-dataflow.yml (7 Active + 3 Commented Nodes)

### Pipeline
```
orchestra-bridge (Zenoh ↔ Dora gateway)
  ↓ video_frame, audio_frame, telemetry, detections
video-encoder (RGB8→JPEG) → web-bridge
audio-converter (Float32→PCM) → web-bridge
speech-recognizer ← orchestra-bridge audio
  ↓ transcription
command-parser → rover_command, tracking_command, tts_command → web-bridge
web-bridge (Socket.IO :3030)
# kokoro-tts (commented)
# pybullet-sim (commented)
```

### web-bridge wiring in orchestra
- **Inputs**: `video_frame` from video-encoder, `audio_frame` from audio-converter, all telemetry/detections from orchestra-bridge, `speech_transcription` from command-parser
- **Outputs**: All command types → orchestra-bridge → [Zenoh] → rover
- **Mode**: `MODE=orchestra`, `ZENOH_ENABLED=false`

## robo_rover_lib Modules

**Binary crate**: No — library only.

### Types exposed (via `pub use types::*`)
- `rover_types`: `RoverCommand` (Legacy/Velocity/JointPositions/Stop), `RoverTelemetry`
- `arm_types`: `ArmCommand` (JointPosition/CartesianMove/Home/Stop/EmergencyStop)
- `arm_telemetry`: `ArmTelemetry` (end_effector_pose, joint_angles, velocities)
- `detection_types`: `BoundingBox`, `DetectionResult` (reid_features, tracking_id)
- `fleet_types`: `FleetSubscriptionCommand`, `ActiveRoversStatus`, `FleetRosterUpdate`, `RoverStatus`
- `video_types`, `speech_types`, `nlu_types`, `tts_types`, `performance_types`

### Utils
- `kinematics`, `mecanum_kinematics`, `tracing`

### Key Structs
- `CommandMetadata { command_id, timestamp, source: InputSource, priority: CommandPriority }`
- `CompleteJointState` — 9-joint unified state (3 wheels + 6 arm)
- `CommandPriority`: Emergency(0), Low(1), Normal(2), High(3)

## Dora YAML Capabilities

### Supported
- **Env var substitution**: `${HOME}`, `${VAR}` syntax (shell expansion by Dora runtime)
- **Multi-input nodes**: nodes can declare multiple named inputs
- **Node ordering**: implicit from input→output graph

### NOT Supported (confirmed absent in both YAMLs)
- Conditional node activation (`if:`, `when:`, `enabled:`)
- Build profiles / YAML variants
- Optional nodes or node groups
- Runtime feature flags at YAML level

### Workaround pattern used
Unused nodes are **manually commented out** (e.g., pybullet-sim, kokoro-tts in orchestra).

## Key Finding: Direct-Connect Dataflow Requirements

For rover-kiwi to run web_bridge directly (no zenoh), it needs:
1. **video-encoder** node — converts kornia_capture RGB8 → JPEG for web_bridge
2. **audio-converter** node — converts Float32 audio → PCM for web_bridge
3. **web-bridge** node — wired directly to rover nodes
4. **Remove zenoh-bridge** — not needed in direct mode

Both `video-encoder` and `audio-converter` are currently `orchestra/` crates but are standard Cargo workspace members — their binaries (`target/release/video_encoder`, `target/release/audio_converter`) are available if compiled.

## Unresolved Questions
- Does Dora support multiple dataflow definitions merged at startup? (No evidence found)
- Does `MODE` env var in web_bridge disable fleet commands at runtime? Need source read.
- Can rover Dockerfile be extended with minimal additions vs full rebuild?
