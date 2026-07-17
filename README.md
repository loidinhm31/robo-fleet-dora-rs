# Robo Rover Dora

A hybrid robotic rover control system with autonomous object tracking and visual servoing capabilities, built on the [Dora](https://github.com/dora-rs/dora) dataflow framework.

## Features

### 🤖 Core System
- **6-DOF Robotic Arm** control with safety checks and kinematics validation
- **3-Wheel Mecanum Base** for omnidirectional movement
- **Real-time Telemetry** streaming and monitoring
- **Web-based Control Interface** with responsive design

### 👁️ Vision Pipeline
- **Object Detection** using YOLOv12n (80 COCO classes)
- **Multi-Object Tracking** with BoTSORT algorithm:
  - Camera Motion Compensation (CMC) for moving rover cameras
  - ReID-based re-identification using OSNet x0.25
  - Two-stage matching for robust associations
  - Track state management (New → Tracked → Lost)
- **Real-time Video Streaming** with JPEG encoding to web clients
- **Bounding Box Visualization** with class labels, confidence scores, and persistent tracking IDs

### 🎯 Autonomous Control
- **Visual Servoing** for autonomous object following
- **PID Control** for smooth centering and distance maintenance
- **Distance Estimation** using monocular vision (pinhole camera model)
- **Control Mode Display** showing Manual/Autonomous operation in web UI
- **Command Priority Arbitration** for safe manual override
- **Safety Constraints** with minimum distance and velocity limits

### 🔊 Audio & Voice System
- **Real-time Audio Streaming** from microphone to web clients
- **Dynamic Audio Control** (start/stop without dataflow restart)
- **Speech Recognition** using central Sherpa VAD/offline STT for voice commands
- **Natural Language Understanding** with Aho-Corasick pattern matching
- **Text-to-Speech** with rover-resident `edge_voice` using Sherpa-ONNX Supertonic 3 INT8
- **Audio Playback** for walkie-talkie/intercom functionality
- **Multi-modal Voice Communication** (command, feedback, and direct streaming)
- **Manual Media Recording** via `orchestra/media_recorder`, which turns validated rover JPEG and PCM into finalized MP4 clips under a deployment-controlled `RECORDING_ROOT`
- **Docker-verified workstation path** for amd64 Orchestra + Rover containers on the current host

## Prerequisites

### System Dependencies

Install GStreamer (required for video capture):
```shell
# Arch/Manjaro
sudo pacman -S gstreamer gst-plugins-base

# Ubuntu/Debian
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

Install Dora CLI:
```shell
cargo install dora-cli --locked
```

Install ALSA development headers (required for audio capture):
```shell
# Arch/Manjaro
sudo pacman -S alsa-lib

# Ubuntu/Debian / Raspberry Pi OS
sudo apt install libasound2-dev
```

Install CMake (required for native Sherpa/TTS builds):
```shell
# Arch/Manjaro
sudo pacman -S cmake

# Ubuntu/Debian
sudo apt install cmake build-essential
```

### ONNX Runtime Setup

The rover vision nodes currently use the Rust `ort` crate `1.16.3`, so use an
ONNX Runtime `1.16.x` shared library.

Use the repo-local bootstrap:

```shell
make models
export ROVER_ORT_DYLIB_PATH="$PWD/models/.runtime/onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so"
```

`make models` validates the pinned runtime archive and installs it under
`models/.runtime/onnxruntime-linux-x64-1.16.3`. The legacy
`./models/scripts/download_onnxruntime.sh` command now forwards to the same
repo-local workflow and prints the required `ROVER_ORT_DYLIB_PATH`.

### AI Models

Download the repo-local bootstrap assets for object detection, speech
recognition, active rover Supertonic TTS, and native ONNX Runtime:
```shell
make models
make check-models
```

This ensures the pinned repo-local cache and runtime layout described in
[models/README.md](models/README.md), including:
- Sherpa ASR bundles under `models/.cache/sherpa-onnx/asr/`
- YOLO and OSNet ONNX exports under `models/.cache/`
- The active Supertonic bundle under `models/.cache/sherpa-onnx/tts/`
- ONNX Runtime under `models/.runtime/onnxruntime-linux-x64-1.16.3/`

Select one startup profile with `STT_PROFILE=en-vad-offline` or
`STT_PROFILE=vi-vad-offline`.

For detailed model setup instructions, see [models/README.md](models/README.md).

## Quick Start

### 1. Build the Project

```shell
# For production (optimized release builds)
cargo build --release
```

### 2. Start Dora

```shell
dora up
```

### 3. Run Dataflow

**Web dataflow with autonomous tracking**:
```shell
dora start web-dataflow.yml --name robo-rover-web --attach
```

**Development dataflow** (keyboard control, local/dev use):
```shell
dora start dev-dataflow.yml --name robo-rover-dev --attach
```

**Run nodes with environment variables**:
```shell
set -a; source .env; set +a; dora start rover-kiwi/rover-kiwi-direct-dataflow.yml --name robo-rover-dev --attach
```

### 4. Start Web UI

```shell
cd robo-control-app
pnpm install
pnpm dev
```

Access at: `http://localhost:5173`

**Default Credentials**: Managed via MongoDB. Set `ALLOW_DEFAULT_CREDENTIALS=true` on first run to bootstrap an admin user, then disable it. See [SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md) for MongoDB setup.

### 5. Test Features

#### Autonomous Tracking
1. **Connect** to Socket.IO server using the web UI
2. **Enable tracking** (send tracking_command via Socket.IO)
3. **Select a target** by clicking on a detected object
4. **Watch the rover follow** the target automatically!

The web UI displays:
- **Control Mode**: AUTO (blue, pulsing) or MANUAL (purple)
- **Distance Estimate**: Real-time distance to target in meters
- **Tracking State**: Current tracking status
- **Video Feed**: Live camera with bounding box overlays

#### Voice Commands
1. **Enable microphone** in the web UI
2. **Speak commands** like:
   - "Move forward"
   - "Turn left"
   - "Track person"
   - "Stop"
3. **Hear voice feedback** confirming commands via TTS

#### Walkie-Talkie Mode
1. **Enable audio streaming** in web UI
2. **Speak into your microphone** - your voice plays through rover speakers
3. Use for remote communication or announcements

### 6. Stop and Cleanup

```shell
# Press Ctrl+C in the dataflow terminal, or:
dora destroy
```

## Media Recording Notes

The Phase 2 recorder is Orchestra-side only and already present in the repo as
`orchestra/media_recorder`.

- `RECORDING_ROOT` is required and must point to a dedicated existing
  directory below `/home`.
- `FFMPEG_PATH` and `FFPROBE_PATH` are optional overrides; otherwise the node
  resolves `ffmpeg` and `ffprobe` from `PATH`.
- `RECORDING_MAX_CONCURRENT`, `RECORDING_MAX_DURATION_SECONDS`,
  `RECORDING_MAX_OUTPUT_BYTES`, `RECORDING_STARTUP_TIMEOUT_SECONDS`,
  `RECORDING_FINALIZATION_TIMEOUT_SECONDS`, `RECORDING_MIN_FREE_BYTES`,
  `RECORDING_QUEUE_CAPACITY`, `RECORDING_AUDIO_SAMPLE_RATE`,
  `RECORDING_AUDIO_CHANNELS`, and `RECORDING_VIDEO_FPS` bound the recorder.
- The node ingests FIFO JPEG video and FIFO S16LE PCM audio, writes partial
  MP4s under `.partial/`, and publishes final MP4 plus manifest pairs only
  after FFmpeg exits cleanly.
- Phase 3 and Phase 4 still add the remaining web/control/playback wiring and
  deployment integration around the recorder.

## System Architecture with Dataflow Pipeline
Check - [ARCHITECTURE](ARCHITECTURE.md)

### Nodes

**Vision & Detection:**
- **gst-camera**: GStreamer video capture (V4L2/RTSP)
- **object-detector**: YOLOv12n inference with ONNX Runtime
- **reid-extractor**: OSNet x0.25 ReID feature extraction (512-dim appearance features)
- **object-tracker**: BoTSORT tracking with CMC, ReID, and Kalman filter
- **visual-servo-controller**: PID-based autonomous following with distance estimation

**Audio & Voice:**
- **audio-capture**: cpal-based audio capture (Rust)
- **central-speech-recognizer**: Sherpa VAD/offline speech-to-text with browser-private and rover source-aware routing
- **command-parser**: NLU for voice command intent extraction
- **edge-voice** (rover): Supertonic 3 INT8 TTS service for edge synthesis
- **audio-playback**: Real-time audio playback for walkie-talkie mode

**Control & Communication:**
- **rover-controller**: Command arbitration, priority handling, mecanum kinematics
- **arm-controller**: 6-DOF arm control with safety checks
- **web-bridge**: Socket.IO server (port 3030) with authentication
- **media-recorder**: Orchestra Dora node for validated rover media ingestion and finalized clip storage
- **sim-interface**: Unity simulation communication (port 4567)

### Visual Servoing Pipeline

The autonomous tracking system works as follows:

1. **Detection**: object-detector identifies objects using YOLOv12n
2. **ReID Feature Extraction**: reid-extractor extracts 512-dim appearance features using OSNet
3. **Tracking**: object-tracker assigns persistent IDs using BoTSORT algorithm:
   - Camera Motion Compensation (CMC) for moving rover
   - Two-stage matching (high-conf: IoU+ReID, low-conf: IoU only)
   - Track state management (only confirmed tracks output)
4. **Target Selection**: User selects target via web UI
5. **Visual Servoing**:
   - **Distance Estimation**: Pinhole camera model calculates distance from bounding box height
   - **PID Control**:
     - Lateral PID: Centers target horizontally (controls omega_z)
     - Longitudinal PID: Maintains target distance (controls v_x)
   - **Safety**: Enforces minimum distance, maximum velocity limits
6. **Command Arbitration**: rover-controller prioritizes commands (Emergency > Autonomous > Manual)
7. **Telemetry**: Servo controller sends enhanced telemetry with distance and mode to web UI

### Socket.IO Events

#### From Web UI to Backend
- `arm_command`: Control robotic arm
- `rover_command`: Manual rover control (priority: Normal)
- `tracking_command`: Enable/disable tracking, select target
- `camera_control`: Start/stop camera
- `audio_control`: Start/stop audio
- `tts_command`: Send text for TTS synthesis
- `audio_stream`: Stream raw audio for walkie-talkie mode

#### From Backend to Web UI
- `video_frame`: JPEG video frames
- `audio_frame`: PCM audio data (S16LE format)
- `detections`: Raw object detections
- `tracked_detections`: Detections with tracking IDs
- `tracking_telemetry`: Basic tracking state from object-tracker
- `servo_telemetry`: Enhanced telemetry with distance and control mode
- `speech_transcription`: Transcribed voice commands
- `arm_telemetry`: Arm position and status
- `rover_telemetry`: Rover position and velocity

### Object Detection

```yaml
object-detector:
  env:
    CONFIDENCE_THRESHOLD: "0.5"              # Min confidence (0.0-1.0)
    NMS_THRESHOLD: "0.4"                     # Non-maximum suppression
    TARGET_CLASSES: "person,dog,cat"         # Filter specific classes (or empty for all)
    MODEL_PATH: "models/.cache/yolo/yolo12n.onnx"  # Path to YOLO model
    ORT_DYLIB_PATH: "onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so"
```

### Object Tracking (BoTSORT)

```yaml
object-tracker:
  env:
    MAX_TRACKING_AGE: "50"    # Max frames to keep lost tracks
    MIN_HITS: "3"             # Min detections before track confirmed
    IOU_THRESHOLD: "0.3"      # IoU threshold for matching detections
    REID_WEIGHT: "0.8"        # Balance between IoU and ReID (0.0-1.0)
    REID_THRESHOLD: "0.5"     # Minimum ReID cosine similarity
    ENABLE_CMC: "true"        # Camera motion compensation for moving rover
```

### Camera Source

```yaml
gst-camera:
  env:
    SOURCE_TYPE: "webcam"        # or "rtsp"
    SOURCE_URI: "/dev/video0"    # or RTSP URL
    IMAGE_COLS: "640"
    IMAGE_ROWS: "480"
    SOURCE_FPS: "30"             # Capture cadence for ML and servo
    VIEW_STREAM_FPS: "15"        # Rover-side JPEG publish cadence
```

### Speech Recognition & Voice Commands

```yaml
central-speech-recognizer:
  env:
    STT_PROFILE: "en-vad-offline"                # or vi-vad-offline
    STT_MODEL_ROOT: "models/.cache/sherpa-onnx/asr"
    STT_NUM_THREADS: "2"
    STT_SAMPLE_RATE: "16000"                     # Fixed; incompatible overrides are rejected
    STT_DECODE_QUEUE_CAPACITY: "8"

command-parser:
  env:
    # No configuration needed - uses built-in pattern matching

# Rover edge TTS
edge-voice:
  env:
    EDGE_VOICE_MODEL_DIR: "models/.cache/sherpa-onnx/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
    EDGE_VOICE_NUM_THREADS: "2"                  # CPU threads for inference
    EDGE_VOICE_QUEUE_CAPACITY: "8"               # Bounded priority queue
    TTS_DEFAULT_LANGUAGE: "en"                   # en or vi
    TTS_DEFAULT_SPEAKER_ID: "5"                  # M1, valid range 0-9
    TTS_DEFAULT_SPEED: "1.0"                     # Speech speed multiplier
    TTS_DEFAULT_STEPS: "8"                       # Supertonic diffusion steps
    TTS_DEFAULT_VOLUME: "0.8"                    # PCM gain before playback

```

**Supertonic Rover Voice**:
- English and Vietnamese share one resident Supertonic engine
- Ten voice SIDs are available; default M1 is SID 5
- Supertonic OpenRAIL-M notice is tracked in `models/SUPERTONIC-OPENRAIL-M-NOTICE.txt`

### Web Bridge Authentication

Authentication uses MongoDB + bcrypt + JWT. Configure via environment variables in the dataflow YAML:

```yaml
web-bridge:
  env:
    MONGODB_URI: "mongodb://localhost:27017"
    MONGODB_DATABASE: "db"
    JWT_SECRET: "your-secret"           # auto-generated with warning if unset
    ALLOW_DEFAULT_CREDENTIALS: "false"  # set true only for first-run bootstrap
    SESSION_TTL_SECONDS: "3600"
```

The web UI authenticates via a login form and receives a JWT token. No hardcoded credentials in the frontend.

## Web UI Implementation

### TypeScript Types

The system uses strongly-typed Socket.IO communication:

```typescript
// Control mode for visual servoing
export type ControlMode = "Manual" | "Autonomous";

// Enhanced tracking telemetry with distance and mode
export interface TrackingTelemetry {
  state: TrackingState;                    // "Disabled" | "Enabled" | "Tracking" | "TargetLost"
  target: TrackingTarget | null;           // Current tracked object
  distance_estimate: number | null;        // Distance in meters (from visual servo)
  control_output: ControlOutput | null;    // PID outputs for debugging
  control_mode: ControlMode;               // "Manual" or "Autonomous"
  timestamp: number;
}
```

### Display Component

The control mode and distance are displayed in the header:

```tsx
{servoTelemetry && (
  <div className="glass-card-light rounded-2xl px-4 md:px-6 py-3">
    {/* Mode indicator */}
    {servoTelemetry.control_mode === "Autonomous" ? (
      <>
        <Zap className="w-4 h-4 text-blue-400 animate-pulse" />
        <span className="text-blue-300">AUTO</span>
      </>
    ) : (
      <>
        <Gauge className="w-4 h-4 text-purple-400" />
        <span className="text-purple-300">MANUAL</span>
      </>
    )}

    {/* Distance display */}
    {servoTelemetry.distance_estimate !== null && (
      <div className="text-white/80 font-mono">
        {servoTelemetry.distance_estimate.toFixed(2)}m
      </div>
    )}
  </div>
)}
```

### Listening to Telemetry

```typescript
// In your React component
socket.on("servo_telemetry", (data: TrackingTelemetry) => {
  setServoTelemetry(data);

  // Access the data
  console.log("Mode:", data.control_mode);           // "Manual" or "Autonomous"
  console.log("Distance:", data.distance_estimate);   // meters or null
  console.log("State:", data.state);                 // tracking state
});
```

## Distance Estimation

The visual servo controller calculates distance using a pinhole camera model:

```
distance = (real_height × focal_length_pixels) / bbox_height_pixels
```

**Default Object Heights** (used for estimation):
- Person: 1.7m
- Dog: 0.5m
- Cat: 0.3m
- Default: 0.5m

**Camera Configuration** (hardcoded, can be modified in code):
- Focal length: 500 pixels (typical for 640x480 webcam)
- Image height: 480 pixels

**Calibrating Focal Length** (optional, for better accuracy):
```python
# Measure a known object at known distance
focal_length_pixels = (bbox_height_pixels × distance_meters) / real_height_meters
```

Update in `visual_servo_controller/src/main.rs`:
```rust
impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            focal_length_pixels: 500.0,  // Update this value
            image_height: 480,
            // ...
        }
    }
}
```

## Troubleshooting

### ONNX Runtime Error
```
libonnxruntime.so: cannot open shared object file
```
**Solution**: Download ONNX Runtime and extract to project root (see Prerequisites)

### Tracing Subscriber Error
```
failed to set up tracing subscriber
```
**Solution**: Already fixed in current version. Build with `cargo build --release`

### Camera Not Found
```
Failed to open camera
```
**Solution**: Check available cameras and update `web-dataflow.yml`:
```shell
ls /dev/video*  # List cameras
v4l2-ctl --list-devices  # Detailed info
```

### Distance Shows Null
**Reasons**:
- Tracking state is not "Tracking" (must be actively tracking, not just "Enabled")
- No target selected in web UI
- Target bounding box too small

**Solution**:
1. Enable tracking: `socket.emit('tracking_command', {type: 'Enable', timestamp: Date.now()})`
2. Select target: Click on detected object in web UI
3. Verify tracking state is "Tracking" in telemetry

### Mode Stuck on Manual
**Check**:
1. Visual servo controller is running (check `dora list`)
2. Tracking state is "Tracking" (not just "Enabled")
3. Web bridge receiving `servo_telemetry` (check browser console)
4. Target is actively being tracked

### Manual Override Not Working
**Check command priority**:
- Manual commands: Normal priority (2)
- Servo commands: High priority (3)
- **Manual override**: Increase priority in `rover_controller` or send Emergency Stop

### Build Errors

**Missing dependencies**:
```shell
# Install all system dependencies
sudo pacman -S gstreamer gst-plugins-base cmake alsa-lib  # Arch
sudo apt install libgstreamer1.0-dev cmake build-essential libasound2-dev  # Ubuntu/Raspberry Pi OS
```

**TypeScript errors**:
```shell
cd robo-control-app
pnpm install
pnpm check-types
```

### Voice Command Issues

**Speech not recognized**:
- Check microphone is working: `arecord -l`
- Verify the selected Sherpa bundle: `make check-models`
- Confirm `STT_PROFILE` is one of `en-vad-offline` or `vi-vad-offline`
- Check `STT_MODEL_ROOT` points at `models/.cache/sherpa-onnx/asr`
- Check `SAMPLE_RATE` matches audio-capture (must be 16000)

**TTS not working on rover**:
- Verify the Supertonic bundle: `make check-models`
- Check required files under `models/.cache/sherpa-onnx/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/`
- Run `cargo test -p edge_voice` and inspect `edge-voice` logs for `voice_status` error detail
- Confirm `EDGE_VOICE_MODEL_DIR` points at the Supertonic directory
- Playback output is owned by `audio-playback`; Phase 04 wires `tts_audio` consumption and suppression

**Walkie-talkie audio choppy**:
- Check network latency (ping between client and server)
- Reduce audio chunk size in web UI
- Verify audio-playback node is running: `dora list`
- Check CPU usage - high load may cause audio dropouts

## Performance Metrics

**Vision Pipeline:**
- **Video Stream**: 15 FPS view output @ 640x480 (`VIEW_STREAM_FPS`); capture may run faster for ML
- **Object Detection**: ~20-30 FPS (YOLOv12n on Raspberry Pi 5)
- **ReID Feature Extraction**: 5-15ms per detection (OSNet x0.25)
- **Object Tracking**: 25-30 FPS with BoTSORT + CMC
- **Control Loop**: 10-20 Hz (limited by tracking rate)
- **Distance Estimation**: <1ms per frame (negligible overhead)
- **PID Update Rate**: Matches tracking frame rate

**BoTSORT Components:**
- **Camera Motion Compensation**: ~5-12ms per frame
- **Two-Stage Matching**: ~1-2ms overhead
- **Track State Management**: Negligible (<1ms)

**Audio & Voice:**
- **Audio Capture**: 16 kHz, Mono, 20 Hz chunks (50ms); F32 locally, S16LE after rover conversion
- **Speech Recognition**: Workstation central recognizer with Sherpa VAD/offline decode; browser transcripts stay private, rover transcripts stay fleet-visible
- **TTS Synthesis** (rover): Sherpa-ONNX Supertonic 3 INT8 via `edge_voice`
- **Walkie-talkie Latency**: <100ms on local network

**Network:**
- **Socket.IO Latency**: <50ms on local network
- **Video Streaming**: ~500-800 KB/s (JPEG quality 80, `VIEW_STREAM_FPS`-gated)
- **Audio Streaming**: ~32 KB/s (16 kHz S16LE)

## Development

### Build Commands

```shell
# Build all nodes
cargo build --release

# Build specific node
cargo build --release -p visual_servo_controller
cargo build --release -p object_detector
cargo build --release -p rover_controller

# Clean and rebuild
cargo clean
cargo build --release
```

### Testing

```shell
# Run Rust tests
cargo test

# Run specific test
cargo test --package visual_servo_controller

# Check TypeScript types
cd robo-rover-app
pnpm check-types
```

### Code Formatting

```shell
# Format Rust code
cargo fmt

# Format TypeScript
cd robo-rover-app
pnpm format
```

### Visualize Dataflow

```shell
# Generate and open dataflow graph
dora graph web-dataflow --open
```

### Monitoring

```shell
# List running dataflows
dora list

# View logs
dora logs robo-rover-web

# View specific node logs
dora logs robo-rover-web visual-servo-controller
```

## Advanced Usage

### Custom PID Tuning Workflow

1. Start with default values
2. Test with a stationary target
3. Observe oscillation and response time
4. Tune `Kp` first (proportional response)
5. Add `Kd` to reduce oscillation
6. Add `Ki` only if steady-state error exists
7. Test with moving targets
8. Adjust safety constraints as needed

### Testing Without Camera

Use a test video file:
```yaml
gst-camera:
  env:
    SOURCE_TYPE: "file"
    SOURCE_URI: "/path/to/test_video.mp4"
```

Or RTSP stream:
```yaml
gst-camera:
  env:
    SOURCE_TYPE: "rtsp"
    SOURCE_URI: "rtsp://example.com/stream"
```

### Multi-Object Tracking

The system can track multiple objects simultaneously. Each object gets a persistent tracking ID. Select a specific target:

```typescript
// Select target by detection index
socket.emit('tracking_command', {
  type: 'SelectTarget',
  detection_index: 0,
  timestamp: Date.now()
});

// Or select by tracking ID
socket.emit('tracking_command', {
  type: 'SelectTargetById',
  tracking_id: 5,
  timestamp: Date.now()
});
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

**Framework & Architecture:**
- [Dora](https://github.com/dora-rs/dora) - Dataflow-oriented robotic architecture

**Vision & Detection:**
- [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics) - Object detection
- BoT-SORT ([arXiv:2206.14651](https://arxiv.org/abs/2206.14651)) - Robust multi-object tracking with CMC
- OSNet ([arXiv:1905.00953](https://arxiv.org/abs/1905.00953)) - Re-identification features
- [GStreamer](https://gstreamer.freedesktop.org/) via [kornia-rs](https://github.com/kornia/kornia-rs) - Video capture
- [ONNX Runtime](https://onnxruntime.ai/) - ML inference

**Audio & Voice:**
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio I/O
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - STT and Supertonic TTS runtime
- [Aho-Corasick](https://docs.rs/aho-corasick/) - Efficient pattern matching

**Web & UI:**
- [React](https://react.dev/) + [Vite](https://vitejs.dev/) - Web framework
- [Tauri](https://tauri.app/) - Desktop app framework
- [Socket.IO](https://socket.io/) - Real-time communication
- [shadcn/ui](https://ui.shadcn.com/) + [Tailwind CSS](https://tailwindcss.com/) - UI components
