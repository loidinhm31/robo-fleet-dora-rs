# Robo-Fleet Docker Setup

This directory contains Docker configuration files for running the Robo-Fleet distributed robotic system in containers.

## Quick Start

### 1. Download Models

Before building the Docker images, download the required ML models:

```bash
make models
```

This will download:
- ✅ Whisper GGML tiny model (~75 MB) - for speech recognition
- ✅ Sherpa-ONNX VITS TTS model (~21 MB) - for text-to-speech

You'll also need to manually export:
- ⚠️ YOLO model (requires PyTorch)
- ⚠️ OSNet ReID model (requires PyTorch)

See the [Model Export](#model-export) section below.

### 2. Build Images

**For Orchestra (Workstation):**
```bash
make build-orchestra
```

**For Rover-Kiwi (Raspberry Pi 5):**
```bash
# On Raspberry Pi (native build):
make build-rover

# Or cross-compile from x86_64 workstation:
make build-rover-cross
```

### 3. Run Containers

**Orchestra:**
```bash
make up-orchestra
```

Access the web UI at http://localhost:3030
- Username: `admin`
- Password: `password` (or set `AUTH_PASSWORD` env var)

**Rover:**
```bash
make up-rover
```

### 4. View Logs

```bash
make logs-orchestra
make logs-rover
```

### 5. Stop Containers

```bash
make down
```

## Architecture

### Two Profiles

The Docker setup uses docker-compose profiles to support two deployment scenarios:

1. **Orchestra Profile** (`orchestra`)
   - Runs on workstation (x86_64)
   - Heavy ML processing, web interface, fleet management
   - Nodes: web-bridge, speech-recognizer, command-parser, audio/video converters, zenoh-bridge

2. **Rover-Kiwi Profile** (`rover-kiwi`)
   - Runs on Raspberry Pi 5 (ARM64)
   - Hardware I/O, ML inference, motor control
   - Nodes: camera, object-detector, object-tracker, visual-servo-controller, audio I/O, arm/rover controllers

### Network Configuration

Both containers use **host networking mode** to enable Zenoh multicast peer discovery (UDP multicast on 224.0.0.224:7446). This allows automatic discovery between orchestra and rovers without manual configuration.

**Exposed Ports:**
- `3030` - Socket.IO web UI (orchestra)
- `7447` - Zenoh TCP (fallback)

### Volume Mounts

**Orchestra:**
- `models/.cache/ggml` → `/models/ggml` (Whisper model)
- `orchestra/zenoh_bridge/zenoh_config.json5` → `/app/config/zenoh_config.json5`

**Rover:**
- `models/.cache/yolo` → `/models/yolo` (YOLO model)
- `models/.cache/reid` → `/models/reid` (OSNet model)
- `models/.cache/sherpa-onnx` → `/models/sherpa-onnx` (TTS model)
- `rover-kiwi/zenoh_bridge/zenoh_config.json5` → `/app/config/zenoh_config.json5`
- `rover-kiwi/config` → `/app/config/rover` (arm config, simulation config)

## Model Export

### YOLO Model

The YOLO model requires PyTorch for export:

```bash
cd models/scripts
python3 -m venv venv
source venv/bin/activate
pip install ultralytics
python3 export_yolo_to_onnx.py
```

This creates `models/.cache/yolo/yolo12n.onnx` (~6 MB).

### OSNet ReID Model

```bash
cd models/scripts
./download_osnet_model.sh
```

This downloads and exports `models/.cache/reid/osnet_x0_25.onnx` (~6 MB).

## Environment Variables

### Orchestra

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_USERNAME` | `admin` | Web UI username |
| `AUTH_PASSWORD` | `password` | Web UI password |
| `SELECTED_ENTITY_ID` | `rover-kiwi` | Default selected rover |
| `FLEET_ROSTER` | `rover-kiwi` | Comma-separated rover IDs |
| `ZENOH_MODE` | `peer` | Zenoh network mode |
| `SOCKET_IO_PORT` | `3030` | Socket.IO server port |

### Rover-Kiwi

| Variable | Default | Description |
|----------|---------|-------------|
| `ENTITY_ID` | `rover-kiwi` | Rover's unique ID |
| `SOURCE_TYPE` | `webcam` | Camera type (webcam/rtsp) |
| `SOURCE_URI` | `/dev/video0` | Camera device path |
| `YOLO_CONFIDENCE` | `0.5` | YOLO detection threshold |
| `ZENOH_MODE` | `peer` | Zenoh network mode |

Example:
```bash
AUTH_PASSWORD=mysecret make up-orchestra
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
    ├── download-models.sh        # Download ML models
    ├── entrypoint-orchestra.sh   # Orchestra startup script
    └── entrypoint-rover.sh       # Rover startup script
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make models` | Download ML models |
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
| `make status` | Check Dora node status |
| `make clean` | Remove containers and images |
| `make check-models` | Verify models are downloaded |

## Troubleshooting

### Orchestra

**Issue: Whisper model not found**
```
ERROR: Whisper model not found at /models/ggml/ggml-tiny.bin
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
# Export YOLO model
cd models/scripts
python3 -m venv venv
source venv/bin/activate
pip install ultralytics
python3 export_yolo_to_onnx.py

# Export OSNet model
./download_osnet_model.sh
```

**Issue: Audio playback fails with ALSA `snd_pcm_open` error**
```
ALSA lib pcm_dmix.c:999:(snd_pcm_dmix_open) unable to open slave
```

This means the container's process isn't in the host's `audio` group. The default `AUDIO_GID=29` (Debian standard) may not match your system.

**Solution:** Pass the correct audio GID when starting the rover:
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
export AUTH_PASSWORD=$(openssl rand -base64 32)
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
  echo "AUTH_PASSWORD=your-secret-password" > .env
  ```

- Use HTTPS/TLS for web UI in production with reverse proxy (nginx, traefik)

## Support

For issues or questions:
- Check [CLAUDE.md](../CLAUDE.md) for system architecture
- Review [README.md](../README.md) for feature documentation
- Open an issue on the repository
