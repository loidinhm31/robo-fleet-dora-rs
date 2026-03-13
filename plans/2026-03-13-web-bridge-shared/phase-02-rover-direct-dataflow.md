# Phase 02 — Create rover-kiwi-direct-dataflow.yml

**Parent plan**: [plan.md](./plan.md)

## Overview

- **Date**: 2026-03-13
- **Description**: New Dora dataflow for rover standalone mode — drops zenoh-bridge, adds video-encoder + audio-converter + web-bridge for direct browser access.
- **Priority**: P2
- **Implementation status**: done
- **Review status**: approved

## Key Insights

1. **Dora has no conditional YAML support** — two separate YAML files is unavoidable.
2. **Binary paths unchanged** — both `rover-kiwi-dataflow.yml` and the new file use `../target/release/` prefix (relative to the `rover-kiwi/` YAML dir, targeting root `target/release/`).
3. **video-encoder input is `image`**, output is `encoded_frame`. **audio-converter input is `audio_data`**, output is `audio_output` — confirmed from orchestra-dataflow.yml.
4. **web-bridge `MODE` env var** is not used in code. `ENTITY_ID` and `FLEET_ROSTER` control fleet display — set to single rover for direct mode.
5. **rover_telemetry / arm_telemetry** not available (sim-interface commented out; hardware telemetry not yet implemented) — skip those web-bridge inputs.
6. **sherpa-tts** can be enabled in direct mode (was commented out in standard dataflow) — TTS is useful when running standalone.

## Requirements

- All current rover-kiwi nodes must work as-is (no changes to existing rover code)
- web-bridge receives: JPEG video, PCM audio, tracked_detections, tracking_telemetry, servo_telemetry, performance_metrics
- web-bridge commands route to: rover-controller, arm-controller, object-tracker, kornia_capture, audio-playback, sherpa-tts
- No zenoh dependency at runtime (ZENOH_CONFIG not needed)

## Architecture

```
gst-camera ──frame──► object-detector ──detections──► reid-extractor
    │                                                        │
    │                                                 detections_with_reid
    │                                                        ▼
    ├──frame──► reid-extractor                        object-tracker ──tracked_detections──►┐
    │                                                        │                               │
    │                                                 tracking_telemetry                     │
    │                                                        ▼                               │
    │                                            visual-servo-controller                     │
    │                                              /servo_command  \servo_telemetry          │
    │                                                   │                  │                 │
    │                                               rover-controller    web-bridge ◄─────────┘
    │                                                                        │
    ├──frame──► video-encoder ──encoded_frame──────────► web-bridge          │
    │                                                                        │
audio-capture ──audio──► audio-converter ──audio_output──► web-bridge        │
    │                                                                        │
performance-monitor ──metrics──────────────────────────► web-bridge          │
                                                                        ▼
                                              rover_command → rover-controller
                                              arm_command   → arm-controller
                                              tracking_cmd  → object-tracker
                                              camera_cmd    → gst-camera
                                              audio_cmd     → audio-capture
                                              audio_stream  → audio-playback
                                              tts_command   → sherpa-tts
```

## Related Code Files

- `rover-kiwi/rover-kiwi-dataflow.yml` (template)
- `orchestra/orchestra-dataflow.yml` (video-encoder/audio-converter/web-bridge wiring reference)
- `web_bridge/` (after Phase 01 move)

## Implementation Steps

### Create `rover-kiwi/rover-kiwi-direct-dataflow.yml`

Start from `rover-kiwi-dataflow.yml`. Apply these changes:

**Remove**: `zenoh-bridge` node entirely.

**Change inputs** — replace `zenoh-bridge/X` with `web-bridge/X`:
```yaml
# audio-capture
audio_control: web-bridge/audio_command        # was: zenoh-bridge/audio_command

# audio-playback
audio: web-bridge/audio_stream                 # was: zenoh-bridge/audio_stream

# gst-camera
camera_control: web-bridge/camera_command      # was: zenoh-bridge/camera_command

# object-tracker
tracking_command: web-bridge/tracking_command  # was: zenoh-bridge/tracking_command

# arm-controller
arm_command: web-bridge/arm_command            # was: zenoh-bridge/arm_command

# rover-controller
rover_command: web-bridge/rover_command        # was: zenoh-bridge/rover_command
```

**Enable sherpa-tts** (uncomment + wire to web-bridge):
```yaml
- id: sherpa-tts
  build: cargo build --release -p sherpa_tts
  path: ../target/release/sherpa_tts
  inputs:
    tts_command: web-bridge/tts_command
  env:
    TTS_MODEL_DIR: "${HOME}/.cache/sherpa-onnx/vits-piper-en_US-lessac-medium"
    TTS_VOLUME: "0.8"
    TTS_SPEED: "1.0"
    TTS_NUM_THREADS: "2"
    TTS_PROVIDER: "cpu"
    LD_LIBRARY_PATH: "${HOME}/ws/robo-fleet-dora-rs/target/release"
```

**Add video-encoder** (from orchestra-dataflow.yml, adapted):
```yaml
- id: video-encoder
  build: cargo build --release -p video_encoder
  path: ../target/release/video_encoder
  inputs:
    image: gst-camera/frame
  outputs:
    - encoded_frame
  env:
    JPEG_QUALITY: "80"
    IMAGE_WIDTH: "640"
    IMAGE_HEIGHT: "480"
```

**Add audio-converter** (from orchestra-dataflow.yml, adapted):
```yaml
- id: audio-converter
  build: cargo build --release -p audio_converter
  path: ../target/release/audio_converter
  inputs:
    audio_data: audio-capture/audio
  outputs:
    - audio_output
  env:
    OUTPUT_FORMAT: "int16"
    SAMPLE_RATE: "16000"
    CHANNELS: "1"
```

**Add web-bridge** (standalone single-rover config):
```yaml
- id: web-bridge
  build: cargo build --release -p web_bridge
  path: ../target/release/web_bridge
  inputs:
    video_frame: video-encoder/encoded_frame
    audio_frame: audio-converter/audio_output
    servo_telemetry: visual-servo-controller/servo_telemetry
    tracked_detections: object-tracker/tracked_detections
    tracking_telemetry: object-tracker/tracking_telemetry
    performance_metrics: performance-monitor/metrics
  outputs:
    - rover_command
    - arm_command
    - camera_command
    - audio_command
    - tracking_command
    - tts_command
    - audio_stream
    - voice_command_audio
  env:
    RUST_LOG: info
    BIND_ADDRESS: "0.0.0.0"
    SOCKET_IO_PORT: "3030"
    MODE: "standalone"
    AUTH_USERNAME: "admin"
    AUTH_PASSWORD: "password"
    ALLOWED_ORIGINS: "*"
    SELECTED_ENTITY_ID: "rover-kiwi"
    FLEET_ROSTER: "rover-kiwi"
    RATE_LIMIT_AUTH_PER_MINUTE: "5"
    RATE_LIMIT_COMMANDS_PER_SECOND: "100"
    MAX_TTS_TEXT_LENGTH: "1000"
    MAX_AUDIO_SAMPLES_PER_MESSAGE: "16000"
    MAX_WHEEL_VELOCITY: "2.0"
```

## Todo

- [x] Create `rover-kiwi/rover-kiwi-direct-dataflow.yml`
- [x] Verify all input/output names match actual node implementations
- [ ] Test locally: `dora start rover-kiwi/rover-kiwi-direct-dataflow.yml --name rover-direct`
- [ ] Confirm web UI connects on port 3030 when running direct dataflow

## Success Criteria

- `dora start rover-kiwi/rover-kiwi-direct-dataflow.yml` starts without errors
- Web browser connects to `http://<rover-ip>:3030`
- Video stream, control commands, and tracking all work
- No zenoh process running

## Risk Assessment

- **audio-converter input name**: Assumed `audio_data` from orchestra YAML — must verify against `orchestra/audio_converter/src/main.rs`
- **video-encoder input name**: Assumed `image` — must verify
- **sherpa-tts LD_LIBRARY_PATH**: The commented-out version has this path; verify it works with direct mode env

## Unresolved Questions

- Does `voice_command_audio` from web-bridge need a consumer? If not wired, Dora will warn but not fail. Can be left dangling in direct mode (no speech recognizer on rover).

## Next Steps

→ Phase 03: Update rover Docker infrastructure for direct mode
