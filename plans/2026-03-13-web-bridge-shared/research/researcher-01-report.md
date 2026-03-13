# Researcher 01: web_bridge Internals & Docker Compose

## web_bridge Cargo.toml Dependencies

- **HTTP/WebSocket**: `axum 0.7`, `socketioxide 0.12`, `tower 0.4`, `tower-http 0.5`
- **Security**: `bcrypt 0.15` (password hashing), `jsonwebtoken 9.2` (JWT), `governor 0.6` (rate limiting)
- **Data**: `arrow 53`, `base64 0.21`, `image 0.25` (JPEG encoding)
- **Async**: `tokio 1.24.2` (full features)
- **Local**: `robo_rover_lib` (types/commands/telemetry)

## web_bridge src/main.rs Structure

**Purpose**: Socket.IO relay bridging web UI clients ↔ Dora dataflow nodes.

### Dora Inputs (receives from dataflow)
`video_frame` (BinaryArray JPEG), `audio_frame` (PCM), `rover_telemetry`, `arm_telemetry`, `servo_telemetry`, `tracked_detections`, `tracking_telemetry`, `performance_metrics`, `speech_transcription`

### Dora Outputs (sends to dataflow)
`rover_command`, `arm_command`, `camera_command`, `audio_command`, `tracking_command`, `tts_command`, `audio_stream` (Float32Array), `voice_command_audio`, `fleet_subscription_command`, `fleet_select_command`

### Socket.IO Events
- **Client → Server**: `rover_command`, `arm_command`, `camera_control`, `audio_control`, `tracking_command`, `tts_command`, `audio_stream`, `voice_command_audio`, `fleet_select`, `fleet_subscription`
- **Server → Client**: `video_frame` (base64 JPEG), `audio_frame`, `rover_telemetry`, `tracked_detections`, `fleet_status`, `active_rovers_status`

### Key Environment Variables
| Var | Default | Purpose |
|-----|---------|---------|
| `BIND_ADDRESS` | 0.0.0.0 | HTTP bind |
| `SOCKET_IO_PORT` | 3030 | WebSocket port |
| `MODE` | orchestra | Behavior mode |
| `AUTH_USERNAME/PASSWORD` | admin/password | Auth |
| `ALLOWED_ORIGINS` | * | CORS |
| `RATE_LIMIT_AUTH_PER_MINUTE` | 5 | Auth rate limit |
| `RATE_LIMIT_COMMANDS_PER_SECOND` | 100 | Command rate limit |
| `SELECTED_ENTITY_ID` | - | Active rover ID |
| `FLEET_ROSTER` | - | Known rovers |
| `ACTIVE_ROVERS` | - | Subscribed rovers |

## Zenoh Bridge Role Comparison

### rover-kiwi/zenoh_bridge (on rover)
- Publishes raw sensor/ML data to Zenoh: `rover/{entity_id}/video_frame`, telemetry, detections
- Subscribes to commands: `rover/{entity_id}/cmd/rover_command`, etc.
- Tight Dora integration: wired to local nodes (camera, tracker, controllers)

### orchestra/zenoh_bridge (on workstation)
- Multi-rover aware: subscribes to `ACTIVE_ROVERS` list dynamically
- Routes commands to selected rover via `SELECTED_ENTITY_ID`
- Outputs all rover data to Dora (web_bridge, speech_recognizer, etc.)

### web_bridge (on orchestra)
- Never touches Zenoh directly (`ZENOH_ENABLED=false`)
- Gets pre-encoded media (JPEG, PCM) from video-encoder/audio-converter via Dora
- Sits at the top of the stack: web UI ↔ web_bridge ↔ orchestra-zenoh-bridge ↔ rover

## Docker Compose Profiles

File: `docker/docker-compose.yml` (version 3.8)

```yaml
services:
  orchestra:
    profiles: ["orchestra"]
    image: robo-fleet-orchestra:latest
    network_mode: host

  rover-kiwi:
    profiles: ["rover-kiwi"]
    image: robo-fleet-rover:latest
    network_mode: host
    privileged: true
    ipc: host        # ALSA audio access
    devices: [/dev/snd]
```

**Profile usage**: `docker compose --profile rover-kiwi up -d`

**Adding a new profile** is straightforward — add a new service with `profiles: ["rover-kiwi-direct"]`. Same image can run a different command/dataflow by changing the `command:` field.

## Makefile Relevant Targets

- `build-orchestra` / `build-rover` / `build-rover-cross` — Docker image builds
- `up-orchestra` / `up-rover` — Start containers
- `down` — Stop all
- `status` — Check Dora node status
- `logs-orchestra` / `logs-rover` — Tail logs

## Unresolved Questions
- Does `MODE` env var in web_bridge actually gate fleet-management Socket.IO events (fleet_select, fleet_subscription)? Need to confirm source.
- Does rover ARM64 Docker image use a different Dockerfile from orchestra x86_64?
