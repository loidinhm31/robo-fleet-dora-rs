# Phase 04: Orchestra Container and Host-Path Deployment

## Context links

- [Parent plan](./plan.md)
- [Phase 02 recorder](./phase-02-media-recorder-ffmpeg-and-storage.md)
- [Phase 03 backend](./phase-03-backend-control-catalog-and-playback.md)
- [Docker guide](../../docker/README.md)
- Depends on: recorder config/binary from Phase 02 and ports from Phase 03. Blocks Phase 06 container acceptance.

## Overview

- Date: 2026-07-17
- Description: Package FFmpeg and recorder, and mount a safe writable recording directory below host `/home`.
- Priority: P1 deployability
- Implementation status: Done (2026-07-18)
- Review status: Approved with warning (2026-07-18)
- Effort: 5h

## Key Insights

- Current Orchestra image builds no recorder and contains no FFmpeg runtime or persistent recording volume.
- The host path and container path differ: deployment config points below `/home`; processes use fixed `/recordings` in-container.
- Fedora Docker-compatible checks may use Podman. SELinux labeling and `XDG_RUNTIME_DIR` are part of acceptance.

## Requirements

- Build/copy `media_recorder` in the Orchestra image and install a pinned/reproducible FFmpeg runtime with H.264/AAC encode support plus ffprobe.
- Require `HOST_RECORDING_PATH` to be a dedicated existing directory such as `/home/<host-user>/robo-fleet-recordings`; never mount `/home` broadly.
- Bind it to `/recordings` with appropriate SELinux relabeling and set `RECORDING_ROOT=/recordings` for recorder and web bridge.
- Run as the existing non-root container user with stable host ownership; no silent fallback to the container layer.
- Health/readiness verifies recorder process, FFmpeg/ffprobe, root writability/free-space, and dataflow process presence.

## Architecture

- Native: deployment supplies an absolute `RECORDING_ROOT` below `/home`.
- Container: `${HOST_RECORDING_PATH}:/recordings:Z`; both recorder and web playback resolve the same fixed container root.
- Media remains host-persistent across container replacement. No volume or secrets exist on rover containers.

## Related code files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Cargo.orchestra.toml` — recorder workspace member.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Dockerfile.orchestra` — manifest cache, build/copy binary, FFmpeg/ffprobe, directory ownership.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.yml` — env, bind mount, process/readiness checks.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.workstation.yml` if host-specific mount/SELinux override is required.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/README.md` — `/home` setup, ownership, Fedora/Podman commands, recovery.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Makefile` only if an existing Orchestra build/smoke target must include the recorder.

## Implementation Steps

1. Add recorder manifests to Docker dependency-cache stages, release build, final copy, and process health list.
2. Install FFmpeg from the image's pinned package source; assert required encoders/muxer with build/smoke checks.
3. Document creation/chown of a narrow host directory below `/home`; make Compose fail when `HOST_RECORDING_PATH` is absent/invalid.
4. Mount at `/recordings` with SELinux-compatible labeling; verify non-root create, fsync, rename, read, and range playback.
5. Propagate duration/concurrency/free-space/finalization limits as bounded env values with safe defaults.
6. Add semantic startup failure for missing encoder or unwritable root; other Orchestra nodes may remain diagnosable, but recording cannot claim ready.
7. Exercise native and Docker-compatible Podman workflows with `XDG_RUNTIME_DIR=/run/user/$(id -u)`.

## Todo list

- [x] Build/copy recorder and install FFmpeg/ffprobe.
- [x] Add required host bind mount and env.
- [x] Set non-root ownership and SELinux guidance.
- [x] Extend health/readiness.
- [x] Add native/container smoke checks.

## Success Criteria

- `docker info` and a real Podman-compatible smoke run succeed with exported `XDG_RUNTIME_DIR`.
- Orchestra image reports required FFmpeg codecs and starts recorder as non-root.
- Recording survives container replacement at the configured host path.
- No clip is written to an anonymous container layer or outside the dedicated root.
- Two concurrent synthetic recordings stay within configured CPU/memory/disk gates.

## Risk Assessment

- Risk: SELinux denies the mount. Mitigation: documented `:Z`, ownership probe, explicit readiness error.
- Risk: codec missing in base repo. Mitigation: assert encoders during image build and pin chosen package/source.
- Risk: host UID mismatch. Mitigation: define one supported ownership setup and validate writes before dataflow readiness.

## Security Considerations

- Never mount `/home`, workspace root, Docker socket, or a world-writable broad directory.
- Do not run the recorder/FFmpeg as root or add elevated container capabilities.
- Treat host path values as deployment config, never Socket.IO input or loggable browser data.

## Next steps

- Phase 05 can integrate against a container or native backend with the same Socket/HTTP contract.
- Phase 06 runs final Podman/Docker acceptance.
- Remaining follow-up is verification in the live deployment environment only if any host-specific mount or SELinux regression shows up.

The approved operational warning is that raw `docker compose` does not perform
the host-side `realpath` containment check; use the documented
`validate-recording-path.sh` preflight (or the Make targets that invoke it)
before starting Orchestra.

## Unresolved questions

- None.
