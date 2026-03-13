# Phase 03 — Update Rover Docker for Direct Mode

**Parent plan**: [plan.md](./plan.md)
**Depends on**: Phase 01 (web_bridge at root), Phase 02 (direct dataflow YAML)

## Overview

- **Date**: 2026-03-13
- **Description**: Extend rover Docker image to include `web_bridge`, `video_encoder`, `audio_converter` binaries; extend entrypoint to select dataflow via `ROVER_MODE` env var; add Makefile convenience target.
- **Priority**: P2
- **Implementation status**: done
- **Review status**: approved

## Key Insights

1. **No new Docker Compose profile needed** — the existing entrypoint already rewrites YAML at startup. Adding `ROVER_MODE` selection there keeps a single service definition.
2. **`docker/Cargo.rover.toml`** is the workspace used for rover builds — add the 3 new crates here (after Phase 01, web_bridge is at `"web_bridge"` not `"orchestra/web_bridge"`).
3. **`Dockerfile.rover-kiwi`** has 3 sections to update: Cargo.toml COPYs, dummy src creation, actual build/copy.
4. **`video_encoder`/`audio_converter`** are lightweight (image encode + PCM format convert); their build deps are already present in the rover builder image (pkg-config, libssl, etc.).
5. **Port 3030** should be documented in docker-compose.yml for direct mode (even with `network_mode: host`, self-documentation helps).
6. **`ROVER_MODE`** defaults to `zenoh` — existing deployments unaffected.

## Requirements

- Rover Docker image (same image) includes `web_bridge`, `video_encoder`, `audio_converter` binaries
- `ROVER_MODE=direct` selects `rover-kiwi-direct-dataflow.yml` at startup
- `ROVER_MODE=zenoh` (default) keeps existing behavior unchanged
- `make up-rover-direct` convenience target

## Related Code Files

- `docker/Cargo.rover.toml`
- `docker/Dockerfile.rover-kiwi`
- `docker/scripts/entrypoint-rover.sh`
- `docker/docker-compose.yml`
- `Makefile`

## Implementation Steps

### Step 1: `docker/Cargo.rover.toml` — add 3 new members

```toml
members = [
    "robo_rover_lib",
    # ... existing rover-kiwi/* members ...
    # Direct-connect mode additions
    "web_bridge",
    "orchestra/video_encoder",
    "orchestra/audio_converter",
]
```

### Step 2: `docker/Dockerfile.rover-kiwi` — Builder stage additions

After the existing COPY Cargo.toml lines, add:
```dockerfile
# Direct-connect mode additions
COPY web_bridge/Cargo.toml web_bridge/
COPY orchestra/video_encoder/Cargo.toml orchestra/video_encoder/
COPY orchestra/audio_converter/Cargo.toml orchestra/audio_converter/
```

In the dummy src creation RUN block, add:
```dockerfile
mkdir -p web_bridge/src && echo "fn main() {}" > web_bridge/src/main.rs && \
mkdir -p orchestra/video_encoder/src && echo "fn main() {}" > orchestra/video_encoder/src/main.rs && \
mkdir -p orchestra/audio_converter/src && echo "fn main() {}" > orchestra/audio_converter/src/main.rs
```

In the actual source COPY, add:
```dockerfile
COPY web_bridge web_bridge/
COPY orchestra/video_encoder orchestra/video_encoder/
COPY orchestra/audio_converter orchestra/audio_converter/
```

In the `cargo build --release` command, add:
```bash
-p web_bridge \
-p video_encoder \
-p audio_converter
```

### Step 3: `docker/Dockerfile.rover-kiwi` — Runtime stage additions

After existing COPY binary lines:
```dockerfile
COPY --from=builder /build/target/release/web_bridge ./bin/
COPY --from=builder /build/target/release/video_encoder ./bin/
COPY --from=builder /build/target/release/audio_converter ./bin/
```

Also copy the direct-connect dataflow YAML:
```dockerfile
COPY rover-kiwi/rover-kiwi-direct-dataflow.yml ./dataflow/rover-kiwi-direct-dataflow.yml
```

Expose port 3030 (for direct mode):
```dockerfile
EXPOSE 3030
```

### Step 4: `docker/scripts/entrypoint-rover.sh` — ROVER_MODE selection

Replace the final `exec dora run /tmp/rover-kiwi-dataflow.yml` block with:

```bash
# Select dataflow based on ROVER_MODE
ROVER_MODE="${ROVER_MODE:-zenoh}"

if [ "$ROVER_MODE" = "direct" ]; then
    echo "Mode: DIRECT (web_bridge on :${SOCKET_IO_PORT:-3030}, no Zenoh)"
    DATAFLOW_SRC="/app/dataflow/rover-kiwi-direct-dataflow.yml"
    DATAFLOW_TMP="/tmp/rover-kiwi-direct-dataflow.yml"
else
    echo "Mode: ZENOH (zenoh-bridge, orchestra connection)"
    DATAFLOW_SRC="/app/dataflow/rover-kiwi-dataflow.yml"
    DATAFLOW_TMP="/tmp/rover-kiwi-dataflow.yml"
fi

cp "$DATAFLOW_SRC" "$DATAFLOW_TMP"

# Apply same path substitutions (existing sed block — generalize to use $DATAFLOW_TMP)
sed -i 's|path: ../target/release/|path: /app/bin/|g' "$DATAFLOW_TMP"
# ... (rest of existing sed commands, changed to use $DATAFLOW_TMP) ...

echo ""
echo "Starting Rover-Kiwi dataflow (mode: $ROVER_MODE)..."
echo "==================================================================="

exec dora run "$DATAFLOW_TMP"
```

Also update the direct-mode YAML path substitutions (same as zenoh mode but add web_bridge config paths if needed).

### Step 5: `docker/docker-compose.yml` — document ROVER_MODE

In the `rover-kiwi` service environment section, add:
```yaml
# Rover operation mode:
#   zenoh  (default): connect to orchestra via Zenoh bridge
#   direct          : run web_bridge locally, no Zenoh required
ROVER_MODE: ${ROVER_MODE:-zenoh}

# Web bridge settings (used when ROVER_MODE=direct)
SOCKET_IO_PORT: "3030"
AUTH_USERNAME: ${AUTH_USERNAME:-admin}
AUTH_PASSWORD: ${AUTH_PASSWORD:-password}
```

### Step 6: `Makefile` — add `up-rover-direct` target

```makefile
up-rover-direct:  ## Start rover in direct-connect mode (web UI on rover, no Zenoh)
	ROVER_MODE=direct docker compose --profile rover-kiwi up -d
```

Also update `make status` or add `make logs-rover-direct` alias if needed (can reuse `logs-rover` since it's the same container).

## Todo

- [x] Update `docker/Cargo.rover.toml` (add 3 members)
- [x] Update `docker/Dockerfile.rover-kiwi` builder stage (3x: COPY Cargo.toml, dummy src, actual build)
- [x] Update `docker/Dockerfile.rover-kiwi` runtime stage (COPY 3 binaries + direct YAML)
- [x] Update `docker/scripts/entrypoint-rover.sh` (ROVER_MODE selection)
- [x] Update `docker/docker-compose.yml` (add ROVER_MODE + web bridge env vars)
- [x] Update `Makefile` (add `up-rover-direct` target)
- [ ] Test: `make build-rover && ROVER_MODE=direct make up-rover`
- [ ] Test: browser connects to `http://<rover-ip>:3030`

## Success Criteria

- `make build-rover` succeeds with new binaries included
- `make up-rover` (no ROVER_MODE set) behaves identically to current
- `make up-rover-direct` starts rover, web UI accessible on port 3030
- No zenoh process running in direct mode

## Risk Assessment

- **Build time**: Adding 3 crates increases rover build time. `web_bridge` depends on `socketioxide`, `axum`, `tokio` — significant but one-time cost due to Docker layer caching.
- **ARM64 cross-compilation**: `web_bridge` has no native OS deps (no GStreamer/ALSA) — should cross-compile cleanly.
- **`axum`/`socketioxide` on ARM**: Standard Rust crates, no architecture issues expected.
- **Image size**: ~10-15 MB additional binaries — acceptable.

## Security Considerations

- Direct mode exposes port 3030 on the rover — ensure `AUTH_USERNAME`/`AUTH_PASSWORD` are changed from defaults in production.
- `ALLOWED_ORIGINS: "*"` default in direct mode YAML should be restricted in production deployments.
- `network_mode: host` means port 3030 is accessible on the LAN — document this.

## Next Steps

After all 3 phases complete:
1. Integration test: start rover in direct mode, connect from browser
2. Integration test: start rover in zenoh mode, verify orchestra still works
3. Update `README.md` and `ARCHITECTURE.md` with direct-connect mode docs
