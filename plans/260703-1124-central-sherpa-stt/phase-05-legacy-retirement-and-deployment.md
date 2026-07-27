# Phase 05 — Legacy Retirement and Deployment

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: passing [Phase 04](./phase-04-system-validation-gate.md)
- Docker: `docker/`
- Models: `models/`
- Edge placeholder: `rover-kiwi/edge_speech_recognizer/`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Remove obsolete Whisper/edge paths, finalize deployment assets, and align documentation after validation. |
| Priority | P2 |
| Implementation status | Complete |
| Review status | Approved 2026-07-04 |
| Effort | 6h |

## Key Insights

- Central crate already removed its normal Whisper dependency, but GGML/model/Docker/docs remnants remain.
- Edge recognizer is disabled but still a workspace/build/container artifact.
- Static Sherpa linkage still requires a real container runtime smoke test.
- Manual rover TTS must remain; only automatic parser feedback is removed.

## Requirements

- Start only after Phase 04 explicitly passes.
- Preserve a known-good commit/image rollback artifact through observation window.
- Remove all active Whisper/GGML central runtime wiring and disabled edge STT build/runtime references.
- Preserve rover Sherpa TTS assets and document echo risk.
- Build/test native and Docker-compatible Podman images after cleanup.

## Architecture

```text
Orchestra: Sherpa VAD/offline STT -> deterministic parser -> validated target bridge
Rover: microphone/capture/Zenoh/controllers + optional manual Sherpa TTS
```

## Related Code Files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml`; regenerate `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.lock`.
- Delete `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_speech_recognizer/`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Dockerfile.orchestra`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Dockerfile.rover-kiwi`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Cargo.rover.toml` and compose/entrypoint/model scripts.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/{Makefile,README.md,ARCHITECTURE.md,SETUP_ENVIRONMENT.md}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/README.md` and model scripts.
- Modify both rover dataflow files to remove commented edge nodes.

## Implementation Steps

1. Verify passing Phase 04 report and record rollback commit/image identifiers.
2. Remove edge recognizer workspace and Docker workspace members, dependency scaffolding, binary build/copy steps, commented nodes, and docs.
3. Delete the edge recognizer directory.
4. Remove Whisper/GGML model downloads, checks, mounts, environment variables, entrypoint rewriting, and active documentation.
5. Regenerate lockfile via Cargo; verify no `whisper-rs` or native Whisper packages remain.
6. Keep only Sherpa ASR model root/profile/thread/VAD/queue configuration for central STT.
7. Validate selected profile and Silero files in container startup without leaking absolute paths/environment.
8. Preserve all rover `sherpa_tts` runtime/model paths.
9. Update docs for final-only startup profiles, browser privacy, rover source routing, manual TTS echo debt, and rollback.
10. Run full Rust tests, lint/build gates, UI checks, and repository searches for obsolete active references.
11. Export `XDG_RUNTIME_DIR=/run/user/$(id -u)` before Docker-compatible Podman checks; verify `docker info` and a real smoke container.
12. Build Orchestra and rover images; run both Sherpa profiles with read-only model mounts.
13. Inspect runtime linkage; static central binary must not require a Sherpa shared object.
14. Deploy Orchestra first, then rover. Monitor status, latency, validation/reset counts, queue drops, command targets, CPU, and RSS.
15. Retain rollback artifact until the observation window completes without regression.

## Todo List

- [x] Confirm passing gate and rollback artifact.
- [x] Remove edge crate and references.
- [x] Remove Whisper/GGML wiring.
- [x] Regenerate and inspect lockfile.
- [x] Finalize container model/config handling.
- [x] Update root/model/deployment/UI docs.
- [x] Run native and UI checks.
- [x] Build and smoke real containers/profiles.
- [x] Deploy in required order and monitor.

## Rollback Artifacts

- Baseline rollback commit retained before cleanup: `c815b3d73973ade3110804a1cb334e780bc97838` (`fix(stt): harden central session validation gate`)
- Orchestra image smoke-tested after cleanup: `localhost/robo-fleet-phase05-orchestra:test` -> `sha256:30a99446aaa3e3cb1bc005d3d442f196850f2666d44b7b98d5e4ec8c1ff1af3f`
- Rover image smoke-tested after cleanup: `localhost/robo-fleet-phase05-rover:test` -> `sha256:8a51ef3398817b7120c64174eee277c1c24bc5df0d922840fe6175ad5b2da1d3`

## Success Criteria

- No active Whisper or edge recognizer build/runtime reference remains.
- Lockfile contains no Whisper dependency.
- Both profiles start from read-only Sherpa mounts in native and container runs.
- Rover image no longer builds/copies edge recognizer.
- Static runtime needs no Sherpa shared object.
- Architecture/setup docs match deployed behavior.
- Rollback and observation evidence are recorded.

## Risk Assessment

- Static artifact may fail in runtime image. Require real image execution, not compile-only evidence.
- Broad cleanup can delete rover TTS assets. Scope searches and review diffs by ASR/TTS ownership.
- Two repositories can drift. Deploy matching contract commits together.

## Security Considerations

- Mount models read-only and preserve non-root runtime posture where hardware access permits.
- Do not add credentials or secrets to compose defaults/logs.
- Confirm cleanup does not bypass authentication, rate limiting, or target validation.

## Next Steps

After observation passes, archive both the superseded and residual plans. Track rover playback suppression/AEC and optional online STT as separate backlog work.

## Unresolved Questions

None.
