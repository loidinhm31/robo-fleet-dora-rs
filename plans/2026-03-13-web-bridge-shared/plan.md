---
title: "Shared web_bridge + Rover Direct-Connect Mode"
description: "Move web_bridge to root workspace, enable rover-kiwi standalone operation via web_bridge without zenoh"
status: done
priority: P2
effort: 4h
branch: main
tags: [web_bridge, rover, docker, dataflow, dora, refactor]
created: 2026-03-13
---

# Plan: Shared web_bridge + Rover Direct-Connect Mode

## Summary

Introduce a `common/` directory for nodes shared across targets. Move `orchestra/web_bridge/` → `common/web_bridge/`. Then create `rover-kiwi-direct-dataflow.yml` enabling the rover to run standalone (web UI directly on rover, no zenoh needed).

**Rule**: `common/` = multi-target nodes. `orchestra/` = workstation-only. `rover-kiwi/` = rover-only.

Mode selection via `ROVER_MODE` env var in the entrypoint — **no new Docker Compose profile needed**.

## Phases

| # | Phase | Effort | Status |
|---|-------|--------|--------|
| 01 | [Move web_bridge to root](./phase-01-move-web-bridge.md) | 30m | DONE |
| 02 | [Create rover-kiwi-direct-dataflow.yml](./phase-02-rover-direct-dataflow.md) | 1.5h | DONE |
| 03 | [Update rover Docker for direct mode](./phase-03-docker-infrastructure.md) | 2h | DONE |

## Architecture Decision

**Dora YAMLs have no conditional/profile support** — nodes are always included or must be manually commented. Two YAML files is unavoidable.

**Docker Compose profile not needed** — the rover entrypoint already rewrites the YAML at startup. Extending it to select YAML by `ROVER_MODE` env var keeps a single `rover-kiwi` service.

```
ROVER_MODE=zenoh   (default) → rover-kiwi-dataflow.yml (current behavior)
ROVER_MODE=direct            → rover-kiwi-direct-dataflow.yml (web_bridge, no zenoh)
```

## Key Files Changed

- `Cargo.toml` — workspace member path
- `docker/Cargo.orchestra.toml` — web_bridge path
- `docker/Cargo.rover.toml` — add web_bridge, video_encoder, audio_converter
- `docker/Dockerfile.rover-kiwi` — build + copy new binaries
- `docker/scripts/entrypoint-rover.sh` — ROVER_MODE selection
- `docker/docker-compose.yml` — document ROVER_MODE option
- `Makefile` — add `up-rover-direct` target
- `rover-kiwi/rover-kiwi-direct-dataflow.yml` — new dataflow
- `orchestra/web_bridge/` → `common/web_bridge/` (moved)
