# Phase 01 — Create `common/` Package & Move web_bridge

**Parent plan**: [plan.md](./plan.md)

## Overview

- **Date**: 2026-03-13
- **Description**: Introduce `common/` directory as the home for nodes shared across targets. Move `orchestra/web_bridge/` → `common/web_bridge/`.
- **Priority**: P2
- **Implementation status**: done
- **Review status**: approved

## Key Insights

1. **Semantic contract**: `common/` = nodes designed to run on *any* target (orchestra or rover). `orchestra/` = workstation-only. `rover-kiwi/` = rover-only. Clear and enforced by directory placement.
2. **`video_encoder`/`audio_converter` stay in `orchestra/`** — they are simple encoding utilities that rover-direct happens to reuse, but they are not architecturally designed to be cross-target. They get added to `docker/Cargo.rover.toml` without moving.
3. **Binary output unchanged** — `target/release/web_bridge` stays the same regardless of source location. `orchestra-dataflow.yml` needs NO path changes.
4. **`docker/Cargo.orchestra.toml`** has its own member list — path must be updated there too.
5. **`Dockerfile.orchestra`** has hardcoded `COPY orchestra/web_bridge/Cargo.toml` — must be updated to `common/web_bridge/Cargo.toml`.
6. **`MODE` env var** is in docker-compose.yml but not used in source — no code change needed.
7. **Future nodes**: any node that becomes multi-target moves to `common/`. The directory sets a clear precedent.

## Requirements

- `cargo build -p web_bridge` succeeds after move
- All orchestra crates still build and run
- `orchestra-dataflow.yml` unchanged
- New structure: `common/web_bridge/`

## Final Directory Structure

```
robo_rover_lib/     ← shared library (types, utils) — unchanged
common/
  web_bridge/       ← Socket.IO server (orchestra + rover-direct)
orchestra/          ← workstation-only nodes (speech, NLU, zenoh, TTS, video_encoder, audio_converter)
rover-kiwi/         ← rover-only nodes (camera, ML pipeline, servos, zenoh, audio)
```

## Related Code Files

- `orchestra/web_bridge/` (source, to be moved)
- `Cargo.toml` (root workspace)
- `docker/Cargo.orchestra.toml`
- `docker/Dockerfile.orchestra`

## Implementation Steps

### Step 1: Create directory and move
```bash
mkdir common
git mv orchestra/web_bridge common/web_bridge
```

### Step 2: Update root `Cargo.toml`
```toml
# Before:
"orchestra/web_bridge",
# After:
"common/web_bridge",
```

### Step 3: Update `docker/Cargo.orchestra.toml`
```toml
# Before:
"orchestra/web_bridge",
# After:
"common/web_bridge",
```

### Step 4: Update `docker/Dockerfile.orchestra`

Find and replace all references to `orchestra/web_bridge`:
```dockerfile
# Before:
COPY orchestra/web_bridge/Cargo.toml orchestra/web_bridge/
# (in dummy src creation):
mkdir -p orchestra/web_bridge/src && echo "fn main() {}" > orchestra/web_bridge/src/main.rs
# (in actual source copy):
COPY orchestra/web_bridge orchestra/web_bridge/

# After:
COPY common/web_bridge/Cargo.toml common/web_bridge/
# (in dummy src creation):
mkdir -p common/web_bridge/src && echo "fn main() {}" > common/web_bridge/src/main.rs
# (in actual source copy):
COPY common web_bridge/   # or: COPY common/web_bridge common/web_bridge/
```

### Step 5: Verify
```bash
cargo build -p web_bridge
cargo build --release  # Full workspace
```

## Todo

- [x] `mkdir common && git mv orchestra/web_bridge common/web_bridge`
- [x] Update root `Cargo.toml` member path
- [x] Update `docker/Cargo.orchestra.toml` member path
- [x] Update `docker/Dockerfile.orchestra` (all 3 occurrence types: COPY Cargo.toml, dummy src, actual COPY)
- [x] `cargo build -p web_bridge` passes
- [ ] `orchestra-dataflow.yml` still starts correctly

## Success Criteria

- `common/web_bridge/` exists as a workspace member
- `cargo build` succeeds for full workspace
- `orchestra/` contains no reference to `web_bridge` anymore
- Docker orchestra build succeeds

## Risk Assessment

- **Low**: pure directory move, zero code changes
- grep `orchestra/web_bridge` across all files before moving to catch any missed references

## Security Considerations

None — no behavioral change.

## Next Steps

→ Phase 02: Create `rover-kiwi/rover-kiwi-direct-dataflow.yml`
