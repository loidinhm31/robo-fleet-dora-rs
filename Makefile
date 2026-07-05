# =============================================================================
# Robo-Fleet Docker Makefile
# =============================================================================
# Easy-to-use commands for building and running the robo-fleet system
# in Docker containers with two profiles: orchestra (workstation) and rover-kiwi

.PHONY: help models models-reset check-models build-orchestra build-rover build-all up-orchestra up-rover \
        up-rover-direct up-mongodb down-mongodb logs-mongodb up-workstation down logs-orchestra \
        logs-rover shell-orchestra shell-rover status clean build-rover-cross format format-check \
        format-file validate-compose validate-workstation-compose

# Default target
.DEFAULT_GOAL := help

# Docker-compatible compose command with file path.
# Keep wrapper compatible with Podman's compose provider on Fedora hosts.
COMPOSE := docker compose -f docker/docker-compose.yml
WORKSTATION_COMPOSE := docker compose -f docker/docker-compose.yml -f docker/docker-compose.workstation.yml

# =============================================================================
# Help
# =============================================================================
help:
	@echo "╔════════════════════════════════════════════════════════════════════════╗"
	@echo "║              Robo-Fleet Docker Commands                                ║"
	@echo "╚════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup:"
	@echo "  make models          - Ensure the pinned repo-local model/runtime cache"
	@echo "  make models-reset    - Rebuild the repo-local model cache atomically"
	@echo "  make check-models    - Validate every required pinned model/runtime file"
	@echo ""
	@echo "Build Images:"
	@echo "  make build-orchestra - Build orchestra image (x86_64)"
	@echo "  make build-rover     - Build rover image (ARM64, native build)"
	@echo "  make build-rover-cross - Build rover image (ARM64, cross-compile from x86_64)"
	@echo "  make build-all       - Build both orchestra and rover images"
	@echo ""
	@echo "Run Containers:"
	@echo "  make up-mongodb      - Start loopback MongoDB for native or container testing"
	@echo "  make up-orchestra    - Start orchestra container (workstation)"
	@echo "  make up-rover        - Start rover container (Raspberry Pi, zenoh mode)"
	@echo "  make up-rover-direct - Start rover in direct mode (web UI on rover, no Zenoh)"
	@echo "  make up-workstation  - Start MongoDB + orchestra + rover with amd64 workstation override"
	@echo "  make down-mongodb    - Stop only the local MongoDB container"
	@echo "  make down            - Stop all containers"
	@echo ""
	@echo "Logs & Monitoring:"
	@echo "  make logs-mongodb    - View MongoDB logs (follow mode)"
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
	@echo "Rust formatting:"
	@echo "  make format          - Format the full Rust workspace"
	@echo "  make format-check    - Check formatting without modifying files"
	@echo "  make format-file FILE=path/to/file.rs - Format one Rust file safely"
	@echo "  Note: cargo fmt -- <paths> does not restrict formatting to those paths."
	@echo ""
	@echo "Environment Variables:"
	@echo "  MONGODB_URI          - MongoDB connection string (default: mongodb://127.0.0.1:27017)"
	@echo "  MONGODB_DATABASE     - MongoDB database name (default: gleanOak)"
	@echo "  MONGODB_PORT         - Loopback host port for the local MongoDB container (default: 27017)"
	@echo "  JWT_SECRET           - JWT signing secret (auto-generated if unset, warn-only)"
	@echo "  ALLOW_DEFAULT_CREDENTIALS - Bootstrap mode: allow default admin/password (default: false)"
	@echo "  SESSION_TTL_SECONDS  - Session expiry in seconds (default: 3600)"
	@echo "  ENTITY_ID            - Rover entity ID (default: rover-kiwi)"
	@echo "  ROVER_MODE           - Rover mode: zenoh (default) or direct"
	@echo "  SOURCE_URI           - Camera device (default: /dev/video0)"
	@echo "  AUDIO_GID            - Host audio group GID for /dev/snd access"
	@echo "                         (default: 29 for Debian/Ubuntu;"
	@echo "                          run: getent group audio | cut -d: -f3)"
	@echo "  STT_PROFILE          - en-vad-offline (default) or vi-vad-offline"
	@echo "  STT_MODEL_ROOT       - Sherpa ASR model root"
	@echo ""
	@echo "Examples:"
	@echo "  make up-mongodb && make build-orchestra && make up-orchestra"
	@echo "  MONGODB_URI=mongodb://127.0.0.1:27017 JWT_SECRET=\$$(openssl rand -base64 32) make up-orchestra"
	@echo "  SOURCE_URI=/dev/video2 make up-rover"
	@echo "  AUDIO_GID=\$$(getent group audio | cut -d: -f3) make up-rover"
	@echo ""

# =============================================================================
# Model Download
# =============================================================================
models:
	@echo "Ensuring repo-local models and ONNX Runtime..."
	@./models/scripts/setup-models.sh ensure

models-reset:
	@echo "Rebuilding repo-local model cache atomically..."
	@./models/scripts/setup-models.sh reset

# =============================================================================
# Build Images
# =============================================================================
build-orchestra:
	@echo "Building orchestra image (x86_64)..."
	$(COMPOSE) --profile orchestra build

build-rover:
	@echo "Building rover image (ARM64, native build)..."
	$(COMPOSE) --profile rover-kiwi build

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
	@echo "Auth: MongoDB at $${MONGODB_URI:-mongodb://127.0.0.1:27017} (db: $${MONGODB_DATABASE:-gleanOak})"
	@echo ""
	@echo "View logs with: make logs-orchestra"

up-rover:
	@echo "Starting rover container (zenoh mode)..."
	$(COMPOSE) --profile rover-kiwi up -d
	@echo ""
	@echo "Rover-Kiwi started!"
	@echo "View logs with: make logs-rover"

up-mongodb:
	@echo "Starting local MongoDB container..."
	$(COMPOSE) --profile mongodb up -d mongodb
	@echo ""
	@echo "MongoDB started on mongodb://127.0.0.1:$${MONGODB_PORT:-27017}"
	@echo "View logs with: make logs-mongodb"

# @env: SOURCE_URI SOURCE_TYPE
up-rover-direct:  ## Start rover in direct-connect mode (web UI on rover, no Zenoh)
	@echo "Starting rover container (direct mode)..."
	ROVER_MODE=direct $(COMPOSE) --profile rover-kiwi up -d
	@echo ""
	@echo "Rover-Kiwi started in direct mode!"
	@echo "Web UI: http://<rover-ip>:3030"
	@echo "View logs with: make logs-rover"

up-workstation:
	@echo "Starting workstation stack (MongoDB + orchestra + rover-kiwi)..."
	$(WORKSTATION_COMPOSE) --profile mongodb --profile orchestra --profile rover-kiwi up -d
	@echo ""
	@echo "Workstation stack started with amd64 overrides"

down:
	@echo "Stopping all containers..."
	$(WORKSTATION_COMPOSE) --profile mongodb --profile orchestra --profile rover-kiwi down

down-mongodb:
	@echo "Stopping local MongoDB container..."
	$(COMPOSE) --profile mongodb stop mongodb
	-docker rm -f robo-mongodb

# =============================================================================
# Logs
# =============================================================================
logs-mongodb:
	$(COMPOSE) --profile mongodb logs -f mongodb

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
	$(WORKSTATION_COMPOSE) --profile mongodb --profile orchestra --profile rover-kiwi down --rmi local -v
	-docker image rm -f localhost/robo-orchestra:latest localhost/robo-rover-kiwi:latest
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
format:
	cargo fmt --all

format-check:
	cargo fmt --all -- --check

# Use rustfmt directly for one file. `cargo fmt -- <paths>` forwards paths to
# rustfmt for every workspace target and does not limit Cargo's target selection.
format-file:
	@test -n "$(FILE)" || (echo "Usage: make format-file FILE=path/to/file.rs" && exit 2)
	rustfmt --edition 2021 "$(FILE)"

validate-compose:
	@echo "Validating docker-compose.yml..."
	$(COMPOSE) --profile mongodb --profile orchestra --profile rover-kiwi config > /dev/null
	@echo "✓ docker-compose.yml is valid"

validate-workstation-compose:
	@echo "Validating docker-compose.yml + docker-compose.workstation.yml..."
	$(WORKSTATION_COMPOSE) --profile mongodb --profile orchestra --profile rover-kiwi config > /dev/null
	@echo "✓ workstation compose is valid"

check-models:
	@echo "Validating repo-local model and runtime artifacts..."
	@./models/scripts/setup-models.sh check
