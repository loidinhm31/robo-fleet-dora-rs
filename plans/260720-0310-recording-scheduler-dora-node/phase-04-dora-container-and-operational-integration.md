# Phase 04 — Dora, Container, and Operational Integration

## Context links

- [Parent plan](./plan.md)
- [Phase 02](./phase-02-scheduler-core-recurrence-and-persistence.md)
- [Phase 03](./phase-03-web-bridge-coordinator-and-recorder-reconciliation.md)
- Dataflow: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`
- Compose: `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.yml`

## Overview

- Date: 2026-07-20
- Description: Build/package scheduler, wire bounded Dora ports, configure Mongo/runtime, health, observability, feature flag, and rollback.
- Priority: P2
- Implementation status: Completed (2026-07-20)
- Review status: Completed — implementation and follow-up review accepted
- Effort: 7h

## Key Insights

- A dataflow node also needs workspace/image build and binary copy changes.
- Fedora Podman Docker compatibility is accepted; runtime environment and real smoke run matter.
- Mongo already gates Orchestra startup; scheduler needs readiness after indexes/reconciliation.
- Manual recording must remain available when scheduling is disabled/degraded.

## Requirements

- Bounded command, feedback, status, intent, and reconciliation queues with documented overflow.
- Scheduler Mongo URI/database, exact 1/2/4/8/16/30-second retry cadence, and fixed 90-day terminal retention config.
- `RECORDING_SCHEDULER_ENABLED` rollout flag; disabled rejects/hides schedule API only.
- Logs/metrics for due/start/stop/retry/missed/conflict/failure, schedules/groups, queue depth, reconciliation.
- Health fails when the scheduler process is missing. An alive process blocked on Mongo/index/reconciliation publishes scheduler-degraded readiness while manual rover control and manual recording stay healthy.
- Retry alert delivery is not added in v1; logs/status/metrics remain the observability boundary.
- Container remains non-root; no new recording-root permissions.

## Architecture

- Dora web -> scheduler: command/query, coordinator feedback, reconciliation snapshot.
- Scheduler -> web: result/snapshot/status, scheduled intent, reconciliation request.
- Web -> recorder: active-session query; recorder -> web: active-session snapshot.
- Startup: Mongo healthy -> nodes -> scheduler indexes/load -> snapshot handshake -> ready.
- Runtime degradation: scheduler status becomes degraded and rejects/pauses schedule actions; container/manual path stays healthy unless the scheduler process exits.
- Rollback: disable flag/UI; gracefully stop/reconcile owned group before removing node; retain Mongo history.

## Related code files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml` — node, ports, queues, env.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml` — scheduler member.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Cargo.orchestra.toml` — container workspace member.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Dockerfile.orchestra` — manifest, build, binary copy.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.yml` — scheduler env/process health.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.workstation.yml` — workstation overrides.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/scripts/entrypoint-orchestra.sh` — only if readiness/flag templating needs it.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Makefile` — help/validation/status.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/README.md` — config/readiness/rollback/retention.

## Implementation Steps

1. Add exact dataflow ports/queue sizes and overflow comments.
2. Add native/container workspace and binary copy points.
3. Add tick/reconcile/horizon/limit/feature config plus frozen retry cadence and 90-day terminal retention.
4. Add ready/degraded scheduler state that preserves manual controls during Mongo/reconciliation outage.
5. Extend process health probe so a missing enabled scheduler fails health, without treating dependency degradation as full-container failure.
6. Validate non-root Mongo networking and recording bind mount.
7. Add keyed logs/metrics/shutdown summary without secrets.
8. Document enable, disable, graceful rollback, backup, index migration.
9. Validate native dataflow and Compose configs before live start.

## Todo list

- [x] Binary builds and appears in image as the non-root `dora` user.
- [x] Dora IDs/queues match contracts and dataflow validation passes.
- [x] Disabled flag preserves manual recorder; only literal `true` enables scheduling.
- [x] Mongo health/startup/retry configuration is wired and validated in Compose/dataflow configuration.
- [x] Missing enabled scheduler fails process health; Mongo/reconciliation degradation reports scheduler unavailable while the manual path remains available.
- [x] Runbook documents enable/disable, rollback, retention, and operational status.

## Completion record

- Delivered bounded Dora ports, scheduler readiness (`initializing`/`ready`/`degraded`), feature-gated schedule API rejection, retry/reconciliation lifecycle logs, queue-depth diagnostics, and operational Docker/Compose wiring.
- Verified with focused Rust tests, `dora graph`, both Compose validation targets, image build/inspection for the scheduler binary and non-root user, and `git diff --check`.
- A live dependency-fault matrix remains the explicit scope of [Phase 06](./phase-06-end-to-end-fault-and-rollout-verification.md); it was not represented as a completed live smoke test here.

## Success Criteria

- `cargo build -p recording_scheduler -p web_bridge -p media_recorder` succeeds.
- Dora dataflow validates and starts with scheduler ports.
- `make validate-compose` and `make validate-workstation-compose` succeed.
- With `XDG_RUNTIME_DIR=/run/user/$(id -u)`, `docker info` and `docker run --rm hello-world` succeed; Podman compatibility is acceptable.
- Image runs scheduler/non-root and manual rollback remains functional.
- Scheduler dependency outage pauses scheduled work, exposes degraded status, and automatically reconciles after recovery.

## Risk Assessment

- Image drift: validate root and container workspace membership.
- Health flapping: process liveness controls container health; dependency readiness controls scheduler-degraded status with bounded recovery.
- Queue loss: durable intent retries; snapshots repair status.
- Rollback active session: graceful disable reconciles/stops owned group.

## Security Considerations

- Keep Mongo credentials/JWT out of logs, image layers, plan fixtures, and health output.
- Keep Mongo private and scheduler DB credentials least-privilege.
- Preserve non-root UID, path allowlist, no-follow protections, and private playback.

## Next steps

- UI connects after snapshots in [Phase 05](./phase-05-scheduler-ui-and-client-state.md).
- Full live fault matrix in [Phase 06](./phase-06-end-to-end-fault-and-rollout-verification.md).

## Unresolved questions

1. Production Mongo credentials/topology/backup owner?
2. Grace period before forced rollback of active scheduled group?
