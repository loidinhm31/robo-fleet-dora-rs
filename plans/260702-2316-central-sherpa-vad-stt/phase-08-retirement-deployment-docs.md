# Phase 08 — Whisper/Edge Retirement, Deployment, and Docs

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: passing [Phase 07](./phase-07-system-validation-gate.md)
- Docker: `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/`
- Models: `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-02 |
| Description | Remove obsolete Whisper/edge paths, finalize containers/configuration, and align documentation. |
| Priority | P2 |
| Implementation status | Pending |
| Review status | Pending |
| Effort | 6h |

## Key Insights

- Whisper assumptions exist in Cargo, Makefile, model scripts, Dockerfile, compose, entrypoint, README, and architecture.
- Edge recognizer is only a disabled placeholder but is still built and copied into the rover image.
- Static Sherpa should not require a runtime shared library in Orchestra image.
- Retirement before validation would remove the fastest rollback path.

## Requirements

- Start only after Phase 07 explicitly passes both profiles.
- Remove all active Whisper runtime/build/model wiring.
- Remove edge placeholder from workspace, Docker, dataflows, and docs.
- Keep rover `sherpa_tts` unchanged except documentation of manual-TTS echo risk.
- Build and smoke native plus container profiles after cleanup.
- Keep previous known-good image/commit as rollback during rollout.

## Architecture

Final deployment contains:

```text
Orchestra: Sherpa VAD/offline STT + command parser + web/Zenoh bridges
Rover: capture/converter/Zenoh + controllers + optional manual Sherpa TTS
```

No Whisper engine, GGML mount, edge STT binary, runtime profile selector, or parser TTS feedback remains.

## Related Code Files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml` and regenerate `Cargo.lock`.
- Delete `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_speech_recognizer/`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/{Dockerfile.orchestra,Dockerfile.rover-kiwi,Cargo.rover.toml,docker-compose.yml}` and entrypoints/scripts.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/{Makefile,README.md,ARCHITECTURE.md,SETUP_ENVIRONMENT.md}` where present.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/README.md` and model download/check scripts.

## Implementation Steps

1. Confirm Phase 07 report identifies both profiles as passed and record rollback image/commit.
2. Remove `edge_speech_recognizer` workspace member, Docker-specific workspace member, dependency prebuild scaffolding, build target, runtime copy, and commented rover dataflow nodes.
3. Delete edge placeholder directory.
4. Remove Whisper dependency remnants, `WHISPER_MODEL_PATH`, GGML model checks/downloads/mounts, GGML runtime directory, compose settings, and entrypoint path rewriting.
5. Regenerate `Cargo.lock` through normal Cargo resolution; verify no `whisper-rs`/Whisper native packages remain.
6. Update Orchestra Docker dependency cache and runtime stages for static Sherpa central binary.
7. Mount `/models/sherpa-onnx/asr` read-only and expose `STT_PROFILE`, model root, threads, VAD tuning, and queue capacity.
8. Make entrypoint validate only the selected profile plus Silero files and print profile/language without leaking full environment.
9. Update `make models`, `make check-models`, Docker docs, model docs, setup docs, README, and architecture from Whisper to Sherpa.
10. Document that automatic parser TTS is removed; manual TTS remains and may be transcribed until playback suppression/AEC backlog is implemented.
11. Update UI docs for browser-private versus fleet-rover transcript behavior and startup-only profile status.
12. Run full Rust test/build set after lockfile cleanup.
13. Run UI tests, type-check, lint, and build against final contracts.
14. Build Orchestra and rover containers using Docker-compatible Podman if applicable.
15. Smoke Orchestra container with English profile, then restart with Vietnamese profile; verify status, browser capture, rover transcript, and command target.
16. Verify `docker run`/Podman runtime does not require a Sherpa shared object when static linkage was selected.
17. Deploy Orchestra first, then rover. Monitor status, decode latency, validation/reset counts, queue drops, command targets, CPU, and RSS.
18. Retain rollback artifact until the agreed observation window completes without regression.

## Todo List

- [ ] Confirm validation gate and rollback artifact.
- [ ] Remove edge crate and all references.
- [ ] Remove Whisper/GGML wiring.
- [ ] Regenerate and inspect lockfile.
- [ ] Finalize Docker model mounts/env/entrypoint.
- [ ] Update root/model/Docker/UI docs.
- [ ] Run final Rust and UI checks.
- [ ] Build and smoke both containers/profiles.
- [ ] Deploy in correct order and monitor.

## Success Criteria

- Repository search finds no active Whisper or edge-recognizer build/runtime reference.
- `Cargo.lock` contains no Whisper dependency.
- Orchestra native and container runs report correct Sherpa profile/status.
- Both profiles work with read-only model mount.
- Rover image no longer builds or copies edge recognizer.
- Final architecture and setup docs match deployed dataflow.
- Rollback instructions and observation metrics are recorded.

## Risk Assessment

- Risk: Static native artifact is not portable to runtime image. Mitigation: build and run real image, inspect linkage, do not rely on compile-only success.
- Risk: Cleanup removes shared Sherpa TTS assets. Mitigation: scope ASR cleanup to GGML/central paths and preserve rover TTS model directory.
- Risk: Documentation drift across two repositories. Mitigation: final architecture diff review after container smoke.

## Security Considerations

- Keep model mounts read-only.
- Do not add secrets or credentials to compose defaults or logs.
- Preserve non-root runtime posture where compatible with Dora and required devices.
- Verify removed fallback does not weaken authentication, rate limiting, or target validation.

## Next Steps

After observation passes, archive the plan with completion reports. Track TTS playback-state suppression/AEC and optional binary browser audio as separate backlog work.
