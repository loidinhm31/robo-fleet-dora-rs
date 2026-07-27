# Phase 07 — Docker Cleanup and Local MongoDB

## Context Links

- [Parent plan](./plan.md)
- [Repository scout](./scout/01-repository-integration-surface.md)
- [Phase 02 models](./phase-02-model-cache-reset-and-bootstrap.md)
- [Phase 05 transport](./phase-05-fleet-transport-and-runtime-authority.md)
- Depends on: Phases 02–05

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Remove retired TTS build/runtime dependencies, add local MongoDB, and make amd64 compose authoritative for this machine. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Approved (2026-07-05 follow-up verification) |
| Recommended model | GPT-5.4; GPT-5.4-mini for bounded build logs |
| Estimated effort | 6h |

## Key Insights

- Orchestra `cmake`/Clang packages trace to retired Kokoro native dependencies.
- Rover `cmake`/Clang and the post-build `binutils` collector trace to retired `sherpa-rs`.
- Rover still needs ALSA, GStreamer, SSL, and dynamic ONNX Runtime for vision.
- Static Sherpa in `edge_voice` prevents ABI collision with vision ONNX Runtime.
- Current workstation compose duplicates a full rover service instead of acting as a minimal override.

## Requirements

### Functional

- Both Dockerfiles build renamed binaries and current dataflows.
- Local MongoDB 8.0 is available for native and Docker web bridge testing.
- Workstation override runs both services as linux/amd64.
- Container entrypoints validate Supertonic files and current model paths.
- Add Make targets for MongoDB and workstation stack operations.

### Non-functional

- MongoDB binds only `127.0.0.1:27017` and uses a named dev volume.
- Do not bake model weights into images; preserve read-only mounts.
- Remove only dependencies proven exclusive to retired engines.
- Docker-compatible Podman is accepted; do not require Docker Engine.

## Architecture

```text
host/native Dora ── mongodb://127.0.0.1:27017 ── mongo:8.0 container

amd64 compose:
  mongodb (loopback port, named volume)
  orchestra (host network, ASR mount)
  rover-kiwi (host network, vision/TTS mounts, amd64 override)
```

Dependency decision:

- Remove Orchestra builder `cmake`, `clang`, `libclang-dev`; remove runtime `libasound2` after `ldd` verification.
- Remove Rover builder `cmake`, `clang`, `libclang-dev`; remove `binutils`/sherpa shared-library discovery/copy.
- Retain builder `pkg-config`, `libasound2-dev`, GStreamer dev, `libssl-dev`.
- Retain rover runtime ALSA/Pulse/GStreamer/SSL packages.
- Retain ONNX Runtime downloader but pin `1.16.3` for vision.

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Dockerfile.orchestra` | Remove Kokoro/redundant packages |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Dockerfile.rover-kiwi` | Edge voice build and cleanup |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.yml` | MongoDB and current env/mounts |
| Replace | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.workstation.yml` | Minimal amd64 override |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Cargo.orchestra.toml` | Remove Kokoro member |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Cargo.rover.toml` | Rename edge voice member |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/scripts/entrypoint-rover.sh` | Supertonic checks/path rewrite |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/Makefile` | Local DB/workstation targets |

## Implementation Steps

1. Update Docker workspace manifests, Cargo scaffold copies, dummy sources, package build list, and binary copies.
2. Delete Kokoro and `sherpa-rs` diagnostic/copy layers.
3. Remove proven-unused build/runtime packages; keep required audio/vision packages.
4. Pin external vision ONNX Runtime downloader to 1.16.3 for amd64.
5. Update model directories, environment variables, entrypoint validation, and logs to Supertonic path.
6. Add `mongo:8.0` under profile `mongodb`, loopback port, healthcheck, named volume.
7. Add `make up-mongodb`, `down-mongodb`, `logs-mongodb`, and `up-workstation` targets.
8. Reduce workstation compose to amd64 build/platform overrides and host-specific resource limits only.
9. Run compose config validation using main plus workstation files.
10. Build each image; delegate at most one bounded build log to GPT-5.4-mini.
11. Run `ldd`/`readelf` inside images; restore a package only with evidence of a missing runtime library.

## Todo List

- [x] Docker manifests renamed
- [x] Retired dependencies/layers removed
- [x] Vision ORT pinned to 1.16.3
- [x] Supertonic entrypoint checks added
- [x] MongoDB service added
- [x] Workstation override simplified
- [x] Make targets added
- [x] Compose config valid
- [x] Both images build
- [x] Runtime dependency audit passes

## Success Criteria

- Dockerfiles contain no Kokoro, Piper, `sherpa-rs`, or sherpa shared-copy logic.
- MongoDB becomes healthy and is reachable from native host and host-network containers.
- Both images build for amd64 through Docker-compatible Podman.
- `ldd` reports no missing libraries for every copied binary.
- Images do not contain model weights or obsolete TTS assets.

## Risk Assessment

- Risk: package removal breaks transitive native build. Mitigation: isolated rebuild and evidence-based restore.
- Risk: Podman compose merge differs from Docker Compose. Mitigation: validate rendered config and actual run.
- Risk: Mongo port collision. Mitigation: preflight check and configurable loopback port.
- Risk: host network service collision. Mitigation: stop native dataflows before full Docker run.
- Note: `down-mongodb` portability nit remains low-severity and non-blocking.

## Security Considerations

- Mongo service is development-only, loopback-bound, and excluded from production claims.
- Never commit `docker/.env` or credentials.
- Generate test JWT secret at runtime; do not print it.
- Do not log credential-bearing MongoDB URI; log redacted host/database only.

## Next Steps

- Hand off to [Phase 08](./phase-08-native-x86-integration-and-benchmark.md).
