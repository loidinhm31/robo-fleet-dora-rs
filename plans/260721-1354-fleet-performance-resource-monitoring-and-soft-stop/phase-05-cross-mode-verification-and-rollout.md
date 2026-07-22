# Phase 05 — Cross-Mode Verification and Rollout

## Context links

- [Parent plan](./plan.md)
- [Architecture](../../ARCHITECTURE.md#planned-resource-monitoring-and-soft-stop-lifecycle)
- Dataflows: `orchestra/orchestra-dataflow.yml`, both Rover dataflows

## Overview

- Date: 2026-07-21
- Priority: P1
- Implementation status: Verification complete on the current x86_64 workstation. Rollout is not authorized.
- Review status: Approved after follow-up review and user approval (2026-07-22T20:00:36+07:00)
- Description: prove safety, resource behavior, recovery, and coordinated deployment.

## Key Insights

- Unit success cannot prove process identity, cgroup scope, device release, or distributed recovery.
- Soft-stop guarantees ownership release and quiescence, not exact RSS reduction.
- Direct mode bypasses both Zenoh bridges and needs explicit parity tests.

## Requirements

- Validate native Orchestra+Rover, direct Rover, and workstation amd64 Podman/Docker-compatible stack.
- Exercise loss, duplicate, reordering, expiry, stale revision, disconnect, restart, and partial failure.
- Verify emergency/Stop responsiveness and no old command replay.
- Collect before/paused/resumed CPU, RSS, device/model state, and resume latency evidence.
- Verify node-specific cancellation/finalization terminal results and scheduled wake-lease races.
- Treat the current x86_64 workstation as the acceptance device for this phase.

## Architecture

Release order: shared Rust/backend contract → bridges/managers/node adapters → UI → remove temporary old-contract read path. Full dataflow restart is rollback and returns runtime lifecycle state to Running. Never roll back only one bridge when new lifecycle topics are active.

## Related code files

- Add Rust unit/integration tests beside types, managers, bridges, monitors, and adapters.
- Add UI Vitest and Playwright fake/live Socket tests.
- Modify Docker compose/healthchecks only as needed for new always-on nodes and scope mounts.
- Add concise deployment/acceptance evidence under this plan's `reports/`.

## Implementation Steps

1. Run Rust format/check/test for affected workspace packages and contract fixtures.
2. Run UI type-check, lint, build, Vitest, and Playwright resource/lifecycle scenarios.
3. Native test one Orchestra-local and one Rover-remote node through pause/resume.
4. Direct-mode test identical Socket contract without Zenoh.
5. Container test cgroup labels/limits and exact process aggregation.
6. Fault-inject lost status, duplicate command, stale revision, expiry, and rover disconnect.
7. Verify recorder finalizes or explicitly fails, TTS/playback reports `interrupted_by_lifecycle`, and STT closes without a partial transcript before Quiesced.
8. Test schedule revalidation on every occurrence, overlap refcount, duplicate/stale acquire/release, disable/delete, wake failure, disconnect/restart/epoch change, user Pause or Resume during auto-wake, and final reconciliation to latest desired state.
9. Verify controllers/emergency path remain live and queued commands never replay.
10. Record CPU/RSS/PSS and resume latency; gate on agreed CPU reduction plus deterministic device/model/buffer release, treating RSS/PSS as supporting evidence only.

## Todo list

- [x] Rust and cross-language contracts pass
- [x] UI checks/tests pass
- [x] Native Orchestra/Rover pass
- [x] Direct mode pass
- [x] Container scope pass
- [x] Fault/safety matrix pass
- [x] Cancellation and schedule wake-lease matrix pass
- [x] Rollout/rollback evidence recorded

## Current verification status

- Focused Rust lifecycle, web bridge, bridge-routing, audio-playback, and edge voice suites pass; the web UI lint and type check pass. See [workstation verification evidence](./reports/phase-05-workstation-verification-2026-07-22.md).
- Direct-mode acceptance passed on the current x86_64 workstation: a 12-node Rover flow used both available cameras and USB audio, active TTS was interrupted by lifecycle pause, the system quiesced, and resume reached ready state in 401 ms. Three paused CPU samples met the required 50% reduction gate.
- Expired commands now receive an immediate terminal `expired` result; native TTS shutdown now allows the observed synchronous generation time without falsely reporting lifecycle failure.
- Final reviewed Orchestra and Rover `linux/amd64` images built successfully; MongoDB, Orchestra, and Rover became healthy. Docker, native Orchestra/Rover, and direct Rover Socket.IO lifecycle smoke tests pass using the web app's installed client dependency. The Docker timeout was fixed by preserving complete lifecycle-status snapshots across Dora queue boundaries.
- Fresh Orchestra/Rover epoch mismatches now surface as explicit `superseded`/`stale_epoch`, rather than a false remote `running` state; delayed reports cannot overwrite active or completed authority transitions. Native disconnect/restart fault handling, stale/expired admission, duplicate/reordered manager and scheduler safety tests, wake-lease reconciliation, and targeted cancellation handling have been exercised. Follow-up code review approved the remediation. No commit or rollout approval has been made.
- Final offline-Rover startup verification also produced an explicit `superseded` result with `stale_epoch` at revision 0; this is the expected baseline when Orchestra starts without a matching Rover epoch.
- Non-blocking follow-up verification (2026-07-22): lifecycle-status Dora queues are sized to 64 entries. Orchestra now rejects initial or runtime activation beyond 15 Rovers, the maximum that fits a complete status tick (15 × 4 remote safe-node reports + 4 local Orchestra reports); this retains the four-Rover acceptance scale with explicit headroom. Focused Rust coverage confirms four-Rover capability expansion, `audio_playback` and `edge_voice` ignore sibling-target lifecycle commands without changing gate state, and the Rover Zenoh bridge rejects malformed or validation-invalid lifecycle commands before its lifecycle-manager relay. `cargo test -p lifecycle_manager -p audio_playback -p edge_voice -p rover_zenoh_bridge` was run as focused package suites. This evidence does not authorize a commit, rollout, or ARM/Raspberry Pi acceptance.

## Success Criteria

- Paused workloads meet the agreed CPU gate over three collection intervals and owned devices/models/buffers deterministically report released; RSS/PSS is evidence, not a hard gate.
- Resource monitoring/control spine remains live; UI shows authoritative/stale states correctly.
- Every accepted active operation emits exactly one terminal result before Quiesced; timeout/partial teardown never reports successful pause.
- Resume reaches readiness within agreed limit without command replay or unsafe actuator output.
- Final code contains no old Fleet Performance synthetic contract/control.

## Risk Assessment

- Device-dependent behavior is accepted on the current x86_64 workstation; record any unavailable host device explicitly.
- RSS may not fall after model drop: document; do not add unsupervised process kill.
- Coordinated release mismatch: staged compatibility window and explicit version logs.

## Security Considerations

- Test unauthorized, expired-token, rate-limit, inactive-rover, and arbitrary-node attempts.
- Confirm audit logs identify actor/request/target/result without credentials or paths.

## Next steps

- If measured unload misses an approved memory target, plan separate supervised hard-stop/restart.

## Unresolved Questions

- CPU gate: at least 50% reduction across three paused samples; resume readiness within 30 seconds.
