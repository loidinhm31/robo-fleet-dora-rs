# Phase 10 — amd64 Docker Verification and Documentation

## Context Links

- [Parent plan](./plan.md)
- [Phase 07 Docker changes](./phase-07-docker-cleanup-and-local-mongodb.md)
- [Phase 09 live E2E](./phase-09-live-web-e2e.md)
- [Development rules](/home/loidinh/.claude/workflows/development-rules.md)
- Depends on: Phases 01–09

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Run complete amd64 container stack, repeat E2E, audit logs/dependencies, and align documentation. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Completed |
| Recommended model | GPT-5.4 orchestrator; GPT-5.4-mini for long logs; GPT-5.5 only for cross-layer failure |
| Estimated effort | 5h |

## Key Insights

- Docker CLI is Podman-compatible on this Fedora host; that is acceptable.
- A successful image build does not prove Dora nodes/models/audio start.
- Same Playwright suite must validate native and Docker stacks.
- Long raw logs are the correct cost-saving target for GPT-5.4-mini, not architecture or completion review.

## Requirements

### Functional

- Validate container runtime with real run, render compose, build, start, inspect, and run E2E.
- Verify MongoDB, Orchestra, Rover, Dora nodes, Supertonic status, model mounts, and runtime libraries.
- Scan bounded logs for panic, missing model/library, repeated restart, and failed healthcheck.
- Update all active documentation and remove stale engine/model/Pi claims for this workflow.

### Non-functional

- Use amd64 override; do not claim ARM/Pi validation.
- Keep raw logs outside repository; store only concise report if needed.
- Main GPT-5.4 session owns final evidence and completion claim.
- Final review compares implementation against architecture and plan gates.

## Architecture

Primary commands:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
docker info
docker run --rm hello-world

docker compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.workstation.yml \
  --profile mongodb --profile orchestra --profile rover-kiwi \
  up -d --build
```

Then run the live edge-voice Playwright suite against the Docker web bridge.

Mini-agent log task must include environment:

```text
CWD: /mnt/data/ws/sharing/robo-fleet-dora-rs
OS: Fedora 44 x86_64
Runtime: Podman Docker compatibility
Target: amd64 workstation compose
Plan: plans/260704-1318-edge-voice-supertonic-x86
Read only the provided bounded log file/range.
Return the structured log contract from reports/01-locked-decisions-and-model-routing.md.
```

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/README.md` | Current native/Docker voice workflow |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/README.md` | Current model lifecycle |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/README.md` | amd64 compose/Mongo/troubleshooting |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/SETUP_ENVIRONMENT.md` | Local environment and safety behavior |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/CLAUDE.md` | Current architecture/model guidance |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md` | Convert target section to implemented current state |

## Implementation Steps

1. Stop native Dora dataflows and confirm ports/devices are free.
2. Validate Podman/Docker compatibility with `docker info` and `hello-world`.
3. Render merged compose config; verify amd64, mounts, profiles, env, healthchecks, and loopback Mongo.
4. Build both images with plain progress; save bounded logs outside repository.
5. Start full stack and wait for all healthchecks.
6. Inspect running Orchestra/Rover processes in-container; assert expected node set including `edge-voice`.
7. Inspect `edge_voice` status/model load and every binary with `ldd`/`readelf` as applicable.
8. Run live edge-voice and existing stream Playwright suites against Docker stack.
9. Collect bounded service logs. Delegate raw repetitive reading to one GPT-5.4-mini subagent.
10. Main GPT-5.4 validates all mini findings, exit codes, health, and E2E reports.
11. If failure crosses FFI/audio/bridge boundaries after evidence collection, escalate the focused diagnosis to GPT-5.5.
12. Update docs, remove stale claims, and perform architecture post-implementation gate.
13. Run full Rust tests, UI type/lint/build/tests, compose config, and final dirty-diff review.

## Todo List

- [x] Native stack stopped cleanly
- [x] Docker/Podman real smoke passes
- [x] Compose render reviewed
- [x] Images build
- [x] Services healthy
- [x] Dora node sets correct
- [x] Supertonic ready in container
- [x] Runtime libraries complete
- [x] Edge voice E2E passes in Docker
- [x] Existing stream E2E passes
- [x] Mini log report verified by main model
- [x] Documentation aligned
- [x] Full tests/review pass

## Success Criteria

- Full amd64 stack starts without restart loops, missing files, missing libraries, or node errors.
- Docker edge voice E2E passes using same assertions as native stack.
- Docker image contains no Kokoro/Piper binaries or exclusive dependencies.
- `make models`, `models-reset`, and `check-models` docs match actual behavior.
- Active docs contain no production Kokoro/Piper/GGML or Pi-validation claims.
- Architecture matches implementation and all global completion gates are evidenced.

## Risk Assessment

- Risk: raw audio device differs inside rootless container. Mitigation: verify Pulse/ALSA mapping separately from synthesis readiness.
- Risk: long logs consume expensive context. Mitigation: bounded file/range plus GPT-5.4-mini structured extraction.
- Risk: mini agent misses causal context. Mitigation: main model reviews surrounding evidence and command exit code.
- Risk: docs drift during final fixes. Mitigation: final architecture/code diff gate after all fixes.

## Security Considerations

- Confirm no secret-bearing `.env`, logs, traces, model URLs with tokens, or credentials enter Git diff.
- MongoDB remains loopback-only and explicitly development-only.
- Container logs redact Mongo credentials and JWTs.
- Review Supertonic OpenRAIL-M notice inclusion before distributing images/models.

## Next Steps

- Mark plan complete only after evidence for every global gate is recorded.
- Archive the plan through the repository plan workflow after implementation review.

## Completion Evidence

- Verified on 2026-07-05 on Fedora x86_64 using Podman Docker compatibility and the workstation override compose file.
- `docker info` and `docker run --rm hello-world` passed with `XDG_RUNTIME_DIR=/run/user/$(id -u)`.
- Rendered merged compose confirmed amd64 services, loopback Mongo, loopback Zenoh endpoints, and workstation audio override wiring.
- `robo-mongodb`, `robo-orchestra`, and `robo-rover-kiwi` stayed healthy.
- In-container process inspection confirmed the expected Orchestra and Rover binaries, including `edge_voice`, `audio_capture`, `audio_playback`, `central_speech_recognizer`, `command_parser`, `web_bridge`, and both Zenoh bridges.
- Dynamic library scans reported no missing runtime libraries for `/app/bin/*` in either container.
- Docker Playwright verification passed with explicit Chromium path:
  - `test:e2e:edge-voice-live`: `1 passed (9.2s)`
  - `test:e2e:stream-live`: `2 passed (29.1s)`
- Audio/STT runtime evidence showed rover capture, rover Zenoh publish, orchestra Zenoh receive, and central STT frame counters advancing without restart loops.
