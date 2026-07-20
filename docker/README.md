# Robo-Fleet Docker Setup

This directory contains Docker configuration files for running the Robo-Fleet distributed robotic system in containers.

Phase 10 verified the full Orchestra + Rover stack on the current Fedora x86_64
workstation using Podman's Docker-compatible CLI plus the workstation override
compose file. ARM64 and Raspberry Pi deployment paths still exist, but they are
not the validated acceptance target for this phase.

## Quick Start

### 1. Download Models

Before building the Docker images, download the required ML models:

```bash
make models
```

This will download:
- ✅ Sherpa Silero VAD + offline ASR bundles - for central workstation speech recognition
- ✅ Supertonic rover TTS bundle
- ✅ YOLO ONNX export
- ✅ OSNet ReID ONNX export

For native x86 runs outside Docker, `make models` also installs the pinned
repo-local ONNX Runtime under `models/.runtime`.

### 2. Prepare Persistent Recording Storage

Orchestra requires an existing, dedicated host directory below `/home`. The
Compose file refuses to start if `HOST_RECORDING_PATH` is missing and mounts
only that directory at `/recordings`; it never mounts `/home` or the workspace.
The image user is UID/GID `1000:1000`, so make the ownership explicit before
starting rootful Docker or the workstation Podman override:

```bash
export HOST_RECORDING_PATH=/home/$USER/robo-fleet-recordings
install -d -m 700 -o "$(id -u)" -g "$(id -g)" "$HOST_RECORDING_PATH"
./docker/scripts/validate-recording-path.sh
```

Run that preflight before every Orchestra container start. It fails fast if the
path is missing, outside the allowed host area, or not writable by the container
user.

On Fedora/Podman, the workstation override adds `userns_mode: keep-id` so the
container UID maps to the invoking host user; the `:Z` bind-mount label is
applied by Compose. If a custom SELinux policy still denies access, inspect
`podman logs robo-orchestra` and relabel only this directory with
`chcon -Rt container_file_t "$HOST_RECORDING_PATH"`.

### Local MongoDB for Scheduler Development

The scheduler can use the local MongoDB Compose profile without starting the
full Orchestra stack. On this Fedora workstation, `docker` may be Podman's
Docker-compatible CLI; set its user runtime directory first. Compose still
expands the Orchestra recording-path variable while starting only the MongoDB
profile, so provide a dedicated absolute project storage path:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export HOST_RECORDING_PATH=/home/$USER/robo-fleet-dora-rs-local-recordings
install -d -m 700 -o "$(id -u)" -g "$(id -g)" "$HOST_RECORDING_PATH"
docker compose -f docker/docker-compose.yml --profile mongodb up -d mongodb
docker inspect --format '{{.State.Health.Status}}' robo-mongodb
```

The last command should print `healthy`. Point scheduler integration tests at
`mongodb://127.0.0.1:27017`, for example:

```bash
SCHEDULER_TEST_MONGODB_URI=mongodb://127.0.0.1:27017 \
  cargo test -p recording_scheduler --test mongo-integration
```

This Compose MongoDB service has no authentication configured. Do not add or
assume an `admin` username/password for this local development endpoint.

### 3. Build Images

**For Orchestra (Workstation):**
```bash
make build-orchestra
```

**For Rover-Kiwi image builds:**
```bash
# Native ARM64 build path:
make build-rover

# Cross-build from x86_64 workstation:
make build-rover-cross
```

Phase 10 runtime verification used the `linux/amd64` workstation override, not
an ARM64 runtime.

### 4. Run Containers

**Full amd64 workstation stack (phase 10 verified):**
```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export HOST_RECORDING_PATH=/home/$USER/robo-fleet-recordings
./docker/scripts/validate-recording-path.sh
docker compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.workstation.yml \
  --profile mongodb --profile orchestra --profile rover-kiwi \
  up -d --build
```

**Orchestra:**
```bash
make up-orchestra
```

Access the web UI at http://localhost:3030

Auth is MongoDB-based. Set `ALLOW_DEFAULT_CREDENTIALS=true` in your `.env` for first-run bootstrap, then log in with `admin` / `password` and change credentials. See `.env.example` for all auth vars.

**Rover:**
```bash
make up-rover
```

### 5. View Logs

```bash
make logs-orchestra
make logs-rover
```

### 6. Stop Containers

```bash
make down
```

## Architecture

### Two Profiles

The Docker setup uses docker-compose profiles to support two deployment scenarios:

1. **Orchestra Profile** (`orchestra`)
   - Runs on workstation (x86_64)
   - Heavy ML processing, web interface, fleet management
   - Nodes: web-bridge, central-speech-recognizer, command-parser, media-recorder, zenoh-bridge

2. **Rover-Kiwi Profile** (`rover-kiwi`)
   - Runs on rover hardware in production; phase 10 validated it on workstation `linux/amd64`
   - Hardware I/O, ML inference, motor control
   - Nodes: camera, edge_voice, object-detector, object-tracker, visual-servo-controller, audio I/O, arm/rover controllers

### Network Configuration

Both containers use **host networking mode**. The workstation override also
forces loopback Zenoh endpoints so the local amd64 stack avoids collisions with
native dataflows on the default Zenoh TCP port.

**Exposed Ports:**
- `3030` - Socket.IO web UI (orchestra)
- `7447` - Zenoh TCP (fallback)

### Volume Mounts

**Orchestra:**
- `models/.cache/sherpa-onnx` → `/models/sherpa-onnx` (shared Sherpa cache for central STT bundles)
- `orchestra/zenoh_bridge/zenoh_config.json5` → `/app/config/zenoh_config.json5`
- `$HOST_RECORDING_PATH` → `/recordings` (required, writable, SELinux relabeled)
- `userns_mode: keep-id` in `docker-compose.workstation.yml` for rootless Podman

**Rover:**
- `models/.cache/yolo` → `/models/yolo` (YOLO model)
- `models/.cache/reid` → `/models/reid` (OSNet model)
- `models/.cache/sherpa-onnx` → `/models/sherpa-onnx` (TTS model)
- `rover-kiwi/zenoh_bridge/zenoh_config.json5` → `/app/config/zenoh_config.json5`
- `rover-kiwi/config` → `/app/config/rover` (arm config, simulation config)

The verified workstation rover path uses `WORKSTATION_AUDIO_DEVICE` with
`sysdefault:CARD=Camera` by default because that ALSA path was stable in the
rootless container, while the Pulse plugin timed out on this host.

## Model Validation

Validate the pinned cache before building images:

```bash
make check-models
```

Use `make models-reset` when you need a full repo-local cache replacement that
preserves the old cache until the staging cache has passed validation.

## Environment Variables

### Orchestra

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DATABASE` | `db` | MongoDB database name |
| `JWT_SECRET` | *(auto-generated)* | JWT signing secret — set explicitly in production |
| `ALLOW_DEFAULT_CREDENTIALS` | `false` | Bootstrap mode: allow default admin/password login |
| `SESSION_TTL_SECONDS` | `3600` | Session expiry in seconds |
| `SELECTED_ENTITY_ID` | `rover-kiwi` | Default selected rover |
| `FLEET_ROSTER` | `rover-kiwi` | Comma-separated rover IDs |
| `ZENOH_MODE` | `peer` | Zenoh network mode |
| `SOCKET_IO_PORT` | `3030` | Socket.IO server port |
| `STT_PROFILE` | `en-vad-offline` | Startup-only central STT profile |
| `STT_MODEL_ROOT` | `/models/sherpa-onnx/asr` | Sherpa ASR bundle root |
| `ORCHESTRA_ZENOH_LISTEN_ENDPOINT` | `tcp/127.0.0.1:7448` in workstation override | Loopback endpoint used for local amd64 Docker verification |
| `HOST_RECORDING_PATH` | *(required)* | Existing dedicated host directory below `/home`; mounted at `/recordings` |
| `RECORDING_MAX_CONCURRENT` | `64` | Upper bound for concurrent rover sessions; lower it for constrained hosts |
| `RECORDING_MAX_DURATION_SECONDS` | `3600` | Per-session duration guard |
| `RECORDING_MAX_OUTPUT_BYTES` | `4294967296` | Per-session output-size guard |
| `RECORDING_MIN_FREE_BYTES` | `1073741824` | Minimum free space required before recording/startup |
| `RECORDING_QUEUE_CAPACITY` | `8` | Bounded per-session input queue |
| `RECORDING_STARTUP_TIMEOUT_SECONDS` | `30` | Recorder startup readiness timeout |
| `RECORDING_FINALIZATION_TIMEOUT_SECONDS` | `30` | FFmpeg/clip finalization timeout on stop/shutdown |
| `RECORDING_VIDEO_QUEUE_CAPACITY` | `16` | Bounded video-frame queue for recorder intake |
| `RECORDING_AUDIO_QUEUE_CAPACITY` | `32` | Bounded audio-frame queue for recorder intake |

### Rover-Kiwi

| Variable | Default | Description |
|----------|---------|-------------|
| `ENTITY_ID` | `rover-kiwi` | Rover's unique ID |
| `SOURCE_TYPE` | `webcam` | Camera type (webcam/rtsp) |
| `SOURCE_URI` | `/dev/video0` | Camera device path |
| `YOLO_CONFIDENCE` | `0.5` | YOLO detection threshold |
| `ZENOH_MODE` | `peer` | Zenoh network mode |
| `ROVER_ZENOH_CONNECT_ENDPOINT` | `tcp/127.0.0.1:7448` in workstation override | Loopback endpoint used for local amd64 Docker verification |
| `AUDIO_DEVICE` | `sysdefault:CARD=Camera` in workstation override | Stable capture device override for the current workstation container path |

Example:
```bash
MONGODB_URI=mongodb://mongo-host:27017 JWT_SECRET=$(openssl rand -base64 32) make up-orchestra
SOURCE_URI=/dev/video2 make up-rover
```

## File Structure

```
docker/
├── Dockerfile.orchestra          # Orchestra multi-stage build (x86_64)
├── Dockerfile.rover-kiwi         # Rover multi-stage build (ARM64)
├── Cargo.orchestra.toml          # Orchestra-only Cargo workspace (used in Docker)
├── Cargo.rover.toml              # Rover-only Cargo workspace (used in Docker)
├── docker-compose.yml            # Compose config with profiles
├── .dockerignore                 # Exclude build artifacts
├── README.md                     # This file
└── scripts/
    ├── download-models.sh        # Compatibility wrapper around models/scripts/setup-models.sh ensure
    ├── entrypoint-orchestra.sh   # Orchestra startup script
    └── entrypoint-rover.sh       # Rover startup script
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make models` | Ensure the pinned repo-local model and runtime cache |
| `make models-reset` | Rebuild the repo-local model cache atomically |
| `make build-orchestra` | Build orchestra image (x86_64) |
| `make build-rover` | Build rover image (ARM64, native) |
| `make build-rover-cross` | Build rover image (cross-compile) |
| `make build-all` | Build both images |
| `make up-orchestra` | Start orchestra container |
| `make up-rover` | Start rover container |
| `make down` | Stop all containers |
| `make logs-orchestra` | View orchestra logs |
| `make logs-rover` | View rover logs |
| `make shell-orchestra` | Open bash in orchestra |
| `make shell-rover` | Open bash in rover |
| `make status` | Check native Dora node status outside the phase 10 container flow |
| `make clean` | Remove containers and images |
| `make check-models` | Validate pinned model/runtime artifacts and fail on missing, corrupt, or unverified assets |

For the current verified Docker workflow, prefer `docker ps`, container
healthchecks, `docker top`, and service logs over `dora list` inside the
containers.

## Troubleshooting

### Orchestra

**Issue: Sherpa STT model missing**
```
ERROR: required STT model file missing: ...
```

**Solution:**
```bash
make models
```

**Issue: Web UI not accessible**

Check if the container is running:
```bash
docker ps | grep robo-orchestra
make logs-orchestra
```

**Issue: Orchestra reports recording readiness failure**

Typical messages include `HOST_RECORDING_PATH is not mounted`, `recording
directory is not writable`, or an unavailable FFmpeg encoder. Check the
deployment directory and image user ownership:

```bash
test -d "$HOST_RECORDING_PATH"
stat -c '%u:%g %a %n' "$HOST_RECORDING_PATH"
docker inspect robo-orchestra --format '{{json .State.Health}}'
docker logs robo-orchestra
```

The container must remain non-root and the host directory must be writable by
the mapped `dora` user. Rootless Podman uses `keep-id` in the workstation
override; rootful Docker uses the image UID/GID `1000:1000`. Do not solve this
by mounting `/home` broadly, using `:U` (which changes host ownership), or
running the recorder as root.

Recording files are finalized on the host, so they survive `docker compose
down`, image rebuilds, and container replacement. Stop the stack, preserve the
directory, correct ownership/labels, then start it again:

```bash
docker compose -f docker/docker-compose.yml --profile orchestra down
install -d -m 700 -o "$(id -u)" -g "$(id -g)" "$HOST_RECORDING_PATH"
docker compose -f docker/docker-compose.yml --profile orchestra up -d --build
```

### Rover

**Issue: Camera not found**
```
WARNING: /dev/video0 not found
```

**Solution:**
- Ensure camera is connected: `ls /dev/video*`
- Check if container is privileged: `docker inspect robo-rover-kiwi | grep Privileged`
- Try different camera: `SOURCE_URI=/dev/video2 make up-rover`

**Issue: YOLO/OSNet model not found**

**Solution:**
```bash
make models
make check-models
```

**Current workstation note:** the phase 10 verified amd64 override does not use
manual `AUDIO_GID` wiring. Prefer the workstation override plus
`WORKSTATION_AUDIO_DEVICE=sysdefault:CARD=Camera` if rover audio frames stay at
zero.

**Issue: Audio playback fails with ALSA `snd_pcm_open` error on a non-workstation or custom container path**
```
ALSA lib pcm_dmix.c:999:(snd_pcm_dmix_open) unable to open slave
```

This usually means the container process is not mapped into the host audio
group and you are not using the phase 10 workstation override path.

**Solution:** only for custom or legacy container launches, pass the correct
audio GID when starting the rover:
```bash
AUDIO_GID=$(getent group audio | cut -d: -f3) make up-rover
```

You can also add it to a `.env` file in the `docker/` directory so it's set automatically:
```bash
echo "AUDIO_GID=$(getent group audio | cut -d: -f3)" >> docker/.env
```

**Issue: Zenoh connection timeout**

Check network configuration:
```bash
# Verify both containers can reach each other
docker exec robo-orchestra ping <rover-ip>
docker exec robo-rover-kiwi ping <orchestra-ip>

# Check Zenoh config
make shell-orchestra
cat /app/config/zenoh_config.json5
```

### General

**Issue: Build fails with dependency errors**

Try clean rebuild:
```bash
make clean
make build-orchestra  # or build-rover
```

**Issue: Container exits immediately**

Check logs for errors:
```bash
make logs-orchestra  # or logs-rover
```

## Development

### Modifying Dataflow

During development, you can mount the dataflow YAML files to test changes without rebuilding:

```yaml
# In docker-compose.yml (uncomment):
volumes:
  - ../orchestra/orchestra-dataflow.yml:/app/dataflow/orchestra-dataflow.yml:ro
```

Then restart:
```bash
make down
make up-orchestra
```

### Building with Custom Tags

```bash
# Custom tag
docker compose -f docker/docker-compose.yml --profile orchestra build --build-arg TAG=dev

# No cache
docker compose -f docker/docker-compose.yml --profile orchestra build --no-cache
```

## Cross-Compilation for ARM64

To build rover images from an x86_64 workstation:

1. **Setup buildx:**
```bash
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx create --name multiarch --use
```

2. **Build:**
```bash
make build-rover-cross
```

3. **Save and transfer:**
```bash
docker save robo-fleet-dora-rs-rover-kiwi:latest | gzip > rover-image.tar.gz
scp rover-image.tar.gz pi@raspberry-pi:~
ssh pi@raspberry-pi 'docker load < rover-image.tar.gz'
```

## Production Deployment

### Recommended Configuration

1. **Use secrets for authentication:**
```bash
# Generate a strong JWT secret and point to your MongoDB instance
export JWT_SECRET=$(openssl rand -base64 32)
export MONGODB_URI=mongodb://your-mongo-host:27017
make up-orchestra
```

2. **Configure CORS for production:**
```yaml
# In docker-compose.yml
environment:
  ALLOWED_ORIGINS: "https://your-domain.com"
```

3. **Resource limits:**
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

4. **Restart policy:**
```yaml
restart: always
```

### Security Considerations

- The rover container runs with `privileged: true` for hardware access. In production, use explicit device mapping:
  ```yaml
  devices:
    - /dev/video0:/dev/video0
    - /dev/snd:/dev/snd
  cap_add:
    - SYS_RAWIO
  ```

- Store sensitive credentials in `.env` file (not committed to git):
  ```bash
  cp docker/.env.example docker/.env
  # Edit docker/.env — set MONGODB_URI, JWT_SECRET, and ALLOW_DEFAULT_CREDENTIALS
  ```

- Use HTTPS/TLS for web UI in production with reverse proxy (nginx, traefik)

## Support

For issues or questions:
- Check [CLAUDE.md](../CLAUDE.md) for system architecture
- Review [README.md](../README.md) for feature documentation
- Open an issue on the repository
