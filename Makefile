# =============================================================================
# Robo-Fleet Docker Makefile
# =============================================================================
# Easy-to-use commands for building and running the robo-fleet system
# in Docker containers with two profiles: orchestra (workstation) and rover-kiwi

.PHONY: help models build-orchestra build-rover build-all up-orchestra up-rover \
        up-rover-direct down logs-orchestra logs-rover shell-orchestra shell-rover \
        status clean build-rover-cross

# Default target
.DEFAULT_GOAL := help

# Docker compose command with file path
# --ansi never: prevent ANSI escape codes that corrupt non-TTY output (e.g. fleet-control SSE stream)
COMPOSE := docker compose --ansi never -f docker/docker-compose.yml

# =============================================================================
# Help
# =============================================================================
help:
	@echo "╔════════════════════════════════════════════════════════════════════════╗"
	@echo "║              Robo-Fleet Docker Commands                                ║"
	@echo "╚════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup:"
	@echo "  make models          - Download required ML models"
	@echo ""
	@echo "Build Images:"
	@echo "  make build-orchestra - Build orchestra image (x86_64)"
	@echo "  make build-rover     - Build rover image (ARM64, native build)"
	@echo "  make build-rover-cross - Build rover image (ARM64, cross-compile from x86_64)"
	@echo "  make build-all       - Build both orchestra and rover images"
	@echo ""
	@echo "Run Containers:"
	@echo "  make up-orchestra    - Start orchestra container (workstation)"
	@echo "  make up-rover        - Start rover container (Raspberry Pi, zenoh mode)"
	@echo "  make up-rover-direct - Start rover in direct mode (web UI on rover, no Zenoh)"
	@echo "  make down            - Stop all containers"
	@echo ""
	@echo "Logs & Monitoring:"
	@echo "  make logs-orchestra  - View orchestra logs (follow mode)"
	@echo "  make logs-rover      - View rover logs (follow mode)"
	@echo "  make status          - Check Dora node status in containers"
	@echo ""
	@echo "Shell Access:"
	@echo "  make shell-orchestra - Open bash shell in orchestra container"
	@echo "  make shell-rover     - Open bash shell in rover container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           - Remove containers, images, and volumes"
	@echo ""
	@echo "Environment Variables:"
	@echo "  MONGODB_URI          - MongoDB connection string (default: mongodb://localhost:27017)"
	@echo "  MONGODB_DATABASE     - MongoDB database name (default: qm_hub)"
	@echo "  JWT_SECRET           - JWT signing secret (auto-generated if unset, warn-only)"
	@echo "  ALLOW_DEFAULT_CREDENTIALS - Bootstrap mode: allow default admin/password (default: false)"
	@echo "  SESSION_TTL_SECONDS  - Session expiry in seconds (default: 3600)"
	@echo "  ENTITY_ID            - Rover entity ID (default: rover-kiwi)"
	@echo "  ROVER_MODE           - Rover mode: zenoh (default) or direct"
	@echo "  SOURCE_URI           - Camera device (default: /dev/video0)"
	@echo "  AUDIO_GID            - Host audio group GID for /dev/snd access"
	@echo "                         (default: 29 for Debian/Ubuntu;"
	@echo "                          run: getent group audio | cut -d: -f3)"
	@echo ""
	@echo "Examples:"
	@echo "  make models && make build-orchestra && make up-orchestra"
	@echo "  MONGODB_URI=mongodb://mongo-host:27017 JWT_SECRET=\$$(openssl rand -base64 32) make up-orchestra"
	@echo "  SOURCE_URI=/dev/video2 make up-rover"
	@echo "  AUDIO_GID=\$$(getent group audio | cut -d: -f3) make up-rover"
	@echo ""

# =============================================================================
# Model Download
# =============================================================================
models:
	@echo "Downloading ML models..."
	@./docker/scripts/download-models.sh

# =============================================================================
# Build Images
# =============================================================================
build-orchestra:
	@echo "Building orchestra image (x86_64)..."
	$(COMPOSE) --profile orchestra build --progress plain

build-rover:
	@echo "Building rover image (ARM64, native build)..."
	$(COMPOSE) --profile rover-kiwi build --progress plain

build-rover-cross:
	@echo "Building rover image (ARM64, cross-compile from x86_64)..."
	@echo "Note: This requires Docker buildx and QEMU setup"
	docker buildx build --platform linux/arm64 \
		-f docker/Dockerfile.rover-kiwi \
		-t robo-fleet-dora-rs-rover-kiwi:latest \
		--load .

build-all: build-orchestra build-rover

# =============================================================================
# Run Containers
# =============================================================================
up-orchestra:
	@echo "Starting orchestra container..."
	$(COMPOSE) --profile orchestra up -d
	@echo ""
	@echo "Orchestra started! Access web UI at: http://localhost:3030"
	@echo "Auth: MongoDB at $${MONGODB_URI:-mongodb://localhost:27017} (db: $${MONGODB_DATABASE:-qm_hub})"
	@echo ""
	@echo "View logs with: make logs-orchestra"

up-rover:
	@echo "Starting rover container (zenoh mode)..."
	$(COMPOSE) --profile rover-kiwi up -d
	@echo ""
	@echo "Rover-Kiwi started!"
	@echo "View logs with: make logs-rover"

# @env: SOURCE_URI SOURCE_TYPE
up-rover-direct:  ## Start rover in direct-connect mode (web UI on rover, no Zenoh)
	@echo "Starting rover container (direct mode)..."
	ROVER_MODE=direct $(COMPOSE) --profile rover-kiwi up -d
	@echo ""
	@echo "Rover-Kiwi started in direct mode!"
	@echo "Web UI: http://<rover-ip>:3030"
	@echo "View logs with: make logs-rover"

down:
	@echo "Stopping all containers..."
	$(COMPOSE) --profile orchestra --profile rover-kiwi down

# =============================================================================
# Logs
# =============================================================================
logs-orchestra:
	$(COMPOSE) --profile orchestra logs -f

logs-rover:
	$(COMPOSE) --profile rover-kiwi logs -f

# =============================================================================
# Shell Access
# =============================================================================
shell-orchestra:
	@echo "Opening shell in orchestra container..."
	@docker exec -it robo-orchestra /bin/bash || echo "Error: Container not running. Start with 'make up-orchestra'"

shell-rover:
	@echo "Opening shell in rover container..."
	@docker exec -it robo-rover-kiwi /bin/bash || echo "Error: Container not running. Start with 'make up-rover'"

# =============================================================================
# Status & Health
# =============================================================================
status:
	@echo "╔════════════════════════════════════════════════════════════════════════╗"
	@echo "║                     Dora Node Status                                   ║"
	@echo "╚════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Orchestra Nodes:"
	@docker exec robo-orchestra dora list 2>/dev/null || echo "  Orchestra container not running"
	@echo ""
	@echo "Rover-Kiwi Nodes:"
	@docker exec robo-rover-kiwi dora list 2>/dev/null || echo "  Rover container not running"
	@echo ""

# =============================================================================
# Cleanup
# =============================================================================
clean:
	@echo "Removing all containers, images, and volumes..."
	$(COMPOSE) --profile orchestra --profile rover-kiwi down --rmi local -v
	@echo "Cleanup complete!"

# =============================================================================
# Advanced Commands
# =============================================================================
restart-orchestra:
	$(COMPOSE) --profile orchestra restart

restart-rover:
	$(COMPOSE) --profile rover-kiwi restart

ps:
	$(COMPOSE) --profile orchestra --profile rover-kiwi ps

# =============================================================================
# Development Helpers
# =============================================================================
validate-compose:
	@echo "Validating docker-compose.yml..."
	$(COMPOSE) config > /dev/null
	@echo "✓ docker-compose.yml is valid"

check-models:
	@echo "Checking for required models..."
	@ls -lh models/.cache/ggml/ggml-tiny.bin 2>/dev/null || echo "  ✗ Whisper model missing"
	@ls -lh models/.cache/yolo/yolo12n.onnx 2>/dev/null || echo "  ✗ YOLO model missing"
	@ls -lh models/.cache/reid/osnet_x0_25.onnx 2>/dev/null || echo "  ✗ OSNet model missing"
	@ls -lhd models/.cache/sherpa-onnx/vits-piper-en_US-lessac-medium 2>/dev/null || echo "  ✗ Sherpa-ONNX model missing"
	@echo ""
	@echo "Run 'make models' to download available models"
