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
- **Multi-Object Tracking** with SORT algorithm and Kalman filtering
- **Real-time Video Streaming** with JPEG encoding to web clients
- **Bounding Box Visualization** with class labels and confidence scores

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
- **Speech Recognition** using Whisper.cpp for voice commands
- **Natural Language Understanding** with Aho-Corasick pattern matching
- **Text-to-Speech** using Kokoro-82M for voice feedback
- **Audio Playback** for walkie-talkie/intercom functionality
- **Multi-modal Voice Communication** (command, feedback, and direct streaming)

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
cargo install dora-cli
```

Install CMake (required for Whisper.cpp speech recognition):
```shell
# Arch/Manjaro
sudo pacman -S cmake

# Ubuntu/Debian
sudo apt install cmake build-essential
```

### ONNX Runtime Setup

The object detection node requires ONNX Runtime. Download and extract:

```shell
# Download ONNX Runtime (version 1.16.3)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz

# Extract in the project root
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
```

The `web-dataflow.yml` is configured to use this library via `ORT_DYLIB_PATH`.

**Alternative system-wide install** (requires sudo):
```shell
sudo cp onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
# Then remove ORT_DYLIB_PATH from web-dataflow.yml
```

### AI Models

Download required models for object detection, speech recognition, and text-to-speech:

**YOLO Model** (object detection):
```shell
cd models
# Download and convert YOLOv12n to ONNX (see models/README.md for details)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt
python3 scripts/export_yolo_to_onnx.py
```

**Whisper Model** (speech recognition):
```shell
cd models
# Download Whisper tiny model (recommended for Raspberry Pi 5)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin -O ggml-tiny.bin
```

**Kokoro TTS Models** (text-to-speech):
```shell
cd models/.cache
# Download Kokoro TTS model and voices
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin
```

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

**Development dataflow** (keyboard control):
```shell
dora start dev-dataflow.yml --name robo-rover-dev --attach
```

### 4. Start Web UI

```shell
cd robo-control-app
pnpm install
pnpm dev
```

Access at: `http://localhost:5173`

**Default Credentials**:
- Username: `admin`
- Password: `password`

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
1. **Enable microphone** in web UI or use rover's built-in microphone
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

## System Architecture

### Dataflow Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ gst-camera  │────▶│ object-      │────▶│ object-       │
│ (30 FPS)    │     │ detector     │     │ tracker       │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                                                  ▼
                    ┌─────────────┐     ┌───────────────┐
                    │ rover-      │◀────│ visual-servo- │
                    │ controller  │     │ controller    │
                    └─────────────┘     └───────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ web-bridge  │────▶ Web UI (React)
                    └─────────────┘
```

### Nodes

**Vision & Detection:**
- **gst-camera**: GStreamer video capture (V4L2/RTSP)
- **object-detector**: YOLOv12n inference with ONNX Runtime
- **object-tracker**: SORT tracking with persistent IDs and Kalman filter
- **visual-servo-controller**: PID-based autonomous following with distance estimation

**Audio & Voice:**
- **audio-capture**: cpal-based audio capture (Rust)
- **speech-recognizer**: Whisper.cpp speech-to-text (Raspberry Pi optimized)
- **command-parser**: NLU for voice command intent extraction
- **kokoro-tts**: Kokoro-82M text-to-speech for voice feedback
- **audio-playback**: Real-time audio playback for walkie-talkie mode

**Control & Communication:**
- **rover-controller**: Command arbitration, priority handling, mecanum kinematics
- **arm-controller**: 6-DOF arm control with safety checks
- **web-bridge**: Socket.IO server (port 3030) with authentication
- **sim-interface**: Unity simulation communication (port 4567)

### Visual Servoing Pipeline

The autonomous tracking system works as follows:

1. **Detection**: object-detector identifies objects using YOLOv12n
2. **Tracking**: object-tracker assigns persistent IDs using SORT algorithm
3. **Target Selection**: User selects target via web UI
4. **Visual Servoing**:
   - **Distance Estimation**: Pinhole camera model calculates distance from bounding box height
   - **PID Control**:
     - Lateral PID: Centers target horizontally (controls omega_z)
     - Longitudinal PID: Maintains target distance (controls v_x)
   - **Safety**: Enforces minimum distance, maximum velocity limits
5. **Command Arbitration**: rover-controller prioritizes commands (Emergency > Autonomous > Manual)
6. **Telemetry**: Servo controller sends enhanced telemetry with distance and mode to web UI

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

## Configuration

### PID Tuning (Visual Servoing)

Edit `web-dataflow.yml`:

```yaml
visual-servo-controller:
  env:
    # Lateral control (centering)
    LATERAL_PID_KP: "1.5"    # Proportional gain
    LATERAL_PID_KI: "0.0"    # Integral gain
    LATERAL_PID_KD: "0.2"    # Derivative gain

    # Longitudinal control (distance)
    LONGITUDINAL_PID_KP: "0.8"
    LONGITUDINAL_PID_KI: "0.0"
    LONGITUDINAL_PID_KD: "0.15"

    # Safety constraints
    MIN_DISTANCE: "1.0"           # meters - minimum approach distance
    MAX_VELOCITY: "0.5"           # m/s - maximum linear velocity
    MAX_ANGULAR_VELOCITY: "1.0"   # rad/s - maximum rotation speed

    # Control parameters
    TARGET_BBOX_HEIGHT: "0.3"  # Target 30% of frame height
    DEAD_ZONE: "0.05"          # 5% centering dead zone (reduces oscillation)
```

**Tuning Tips**:
- Increase `LATERAL_PID_KP` for faster centering (may oscillate)
- Increase `LONGITUDINAL_PID_KP` for faster approach
- Add `LATERAL_PID_KD` to reduce oscillation
- Increase `MIN_DISTANCE` to stay farther from target
- Decrease `MAX_VELOCITY` for smoother, slower motion

### Object Detection

```yaml
object-detector:
  env:
    CONFIDENCE_THRESHOLD: "0.5"              # Min confidence (0.0-1.0)
    NMS_THRESHOLD: "0.4"                     # Non-maximum suppression
    TARGET_CLASSES: "person,dog,cat"         # Filter specific classes (or empty for all)
    MODEL_PATH: "models/yolo12n.onnx"        # Path to YOLO model
    ORT_DYLIB_PATH: "onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so"
```

### Object Tracking

```yaml
object-tracker:
  env:
    MAX_TRACKING_AGE: "30"   # Max frames to keep lost tracks
    MIN_HITS: "3"            # Min detections before track confirmed
    IOU_THRESHOLD: "0.3"     # IoU threshold for matching detections
```

### Camera Source

```yaml
gst-camera:
  env:
    SOURCE_TYPE: "webcam"        # or "rtsp"
    SOURCE_URI: "/dev/video0"    # or RTSP URL
    IMAGE_COLS: "640"
    IMAGE_ROWS: "480"
    SOURCE_FPS: "30"
```

### Speech Recognition & Voice Commands

```yaml
speech-recognizer:
  env:
    WHISPER_MODEL_PATH: "models/ggml-tiny.bin"  # tiny/base/small
    SAMPLE_RATE: "16000"                         # Must match audio-capture
    BUFFER_DURATION_MS: "5000"                   # Audio buffer size
    CONFIDENCE_THRESHOLD: "0.5"                  # Min transcription confidence
    ENERGY_THRESHOLD: "0.02"                     # VAD threshold

command-parser:
  env:
    # No configuration needed - uses built-in pattern matching

kokoro-tts:
  env:
    MODEL_DIR: "models/.cache"                   # TTS model location
    VOICE: "af"                                  # Voice style (af/af_sarah/bf_emma/etc)
    VOLUME: "1.0"                                # Audio volume (0.0-2.0)
```

**Supported Voice Styles:**
- `af` - American Female
- `af_sarah` - American Female (Sarah)
- `bf_emma` - British Female (Emma)
- `am` - American Male
- `bm` - British Male

### Web Bridge Authentication

```yaml
web-bridge:
  env:
    AUTH_USERNAME: "admin"
    AUTH_PASSWORD: "password"
```

Update the corresponding values in the web UI:
```typescript
// robo-control-app/src/views/RoboRoverControl.tsx
const AUTH_USERNAME = "admin";
const AUTH_PASSWORD = "password";
```

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
sudo pacman -S gstreamer gst-plugins-base cmake  # Arch
sudo apt install libgstreamer1.0-dev cmake build-essential  # Ubuntu
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
- Verify Whisper model downloaded: `ls -lh models/ggml-tiny.bin`
- Increase `BUFFER_DURATION_MS` for longer phrases (e.g., 7000ms)
- Lower `ENERGY_THRESHOLD` if voice not detected (try 0.01)
- Check `SAMPLE_RATE` matches audio-capture (must be 16000)

**TTS not working**:
- Verify Kokoro models downloaded: `ls -lh models/.cache/kokoro-v1.0.onnx`
- Check audio output device: `pactl list sinks`
- Increase `VOLUME` in kokoro-tts config
- Check logs for model loading errors

**Walkie-talkie audio choppy**:
- Check network latency (ping between client and server)
- Reduce audio chunk size in web UI
- Verify audio-playback node is running: `dora list`
- Check CPU usage - high load may cause audio dropouts

## Performance Metrics

**Vision Pipeline:**
- **Video Stream**: 30 FPS @ 640x480
- **Object Detection**: ~20-30 FPS (YOLOv12n on CPU)
- **Object Tracking**: Real-time with persistent IDs
- **Control Loop**: 10-20 Hz (limited by tracking rate)
- **Distance Estimation**: <1ms per frame (negligible overhead)
- **PID Update Rate**: Matches tracking frame rate

**Audio & Voice:**
- **Audio Capture**: 16 kHz, Mono, 20 Hz chunks (50ms)
- **Speech Recognition**: 1-2s latency (Whisper tiny on RPi5)
- **TTS Synthesis**: 0.5-2s time-to-first-audio (Kokoro-82M)
- **Walkie-talkie Latency**: <100ms on local network

**Network:**
- **Socket.IO Latency**: <50ms on local network
- **Video Streaming**: ~500-800 KB/s (JPEG quality 80)
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

## Contributing

When adding new features:

1. Update relevant sections in this README
2. Update CLAUDE.md with technical details
3. Add types to `robo_rover_lib` for shared data structures
4. Follow existing patterns for node communication
5. Test with both dev and web dataflows
6. Update web UI TypeScript types if adding new telemetry
7. Run tests and type checking before committing

## Project Structure

```
robo-rover-dora/
├── Vision & Detection Nodes
│   ├── kornia_capture/          # GStreamer camera capture (gst-camera)
│   ├── object_detector/         # YOLOv12n inference (ONNX)
│   ├── object_tracker/          # SORT tracking with Kalman filter
│   └── visual_servo_controller/ # PID-based autonomous following
│
├── Audio & Voice Nodes
│   ├── audio_capture/           # cpal audio capture (Rust)
│   ├── audio_playback/          # Audio playback for walkie-talkie
│   ├── speech_recognizer/       # Whisper.cpp speech-to-text
│   ├── command_parser/          # NLU for voice commands
│   └── kokoro_tts/              # Kokoro-82M text-to-speech
│
├── Control Nodes
│   ├── arm_controller/          # 6-DOF arm control node
│   ├── rover_controller/        # Mecanum wheel control with arbitration
│   ├── dispatcher_keyboard/     # Keyboard input for dev mode
│   └── sim_interface/           # Unity simulation bridge
│
├── Communication
│   ├── web_bridge/              # Socket.IO server (port 3030)
│   └── robo_rover_lib/          # Shared types and utilities
│
├── Web Application
│   └── robo-control-app/        # Vite + React + Tauri app
│       ├── src/                 # React components and views
│       └── src-tauri/           # Tauri desktop wrapper
│
├── Configuration & Models
│   ├── config/                  # Arm configurations (*.toml)
│   ├── models/                  # AI models directory
│   │   ├── *.onnx              # YOLO detection models
│   │   ├── *.bin               # Whisper speech models
│   │   └── .cache/             # Kokoro TTS models
│   ├── web-dataflow.yml         # Production dataflow
│   ├── dev-dataflow.yml         # Development dataflow
│   └── README.md                # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

**Framework & Architecture:**
- [Dora](https://github.com/dora-rs/dora) - Dataflow-oriented robotic architecture

**Vision & Detection:**
- [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics) - Object detection
- SORT - Simple Online and Realtime Tracking
- [GStreamer](https://gstreamer.freedesktop.org/) via [kornia-rs](https://github.com/kornia/kornia-rs) - Video capture
- [ONNX Runtime](https://onnxruntime.ai/) - ML inference

**Audio & Voice:**
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio I/O
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Speech-to-text
- [whisper-rs](https://github.com/tazz4843/whisper-rs) - Rust bindings for Whisper
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - Text-to-speech
- [Aho-Corasick](https://docs.rs/aho-corasick/) - Efficient pattern matching

**Web & UI:**
- [React](https://react.dev/) + [Vite](https://vitejs.dev/) - Web framework
- [Tauri](https://tauri.app/) - Desktop app framework
- [Socket.IO](https://socket.io/) - Real-time communication
- [shadcn/ui](https://ui.shadcn.com/) + [Tailwind CSS](https://tailwindcss.com/) - UI components
>>>>>>> robo-rover-dora/main
