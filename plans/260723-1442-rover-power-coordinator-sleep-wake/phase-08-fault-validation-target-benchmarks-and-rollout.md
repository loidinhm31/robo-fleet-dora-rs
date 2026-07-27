# Phase 08 — Fault Validation, Target Benchmarks, and Rollout

## Context links

- Parent: [plan.md](./plan.md)
- Design gates: [Acceptance Targets and Invariants](../../docs/power-coordinator-architecture.md#acceptance-targets)
- Inputs: [authority tests](./research/researcher-01-power-authority-journal.md), [voice benchmarks](./research/researcher-02-voice-resource-ui.md)
- Dependencies: Phases 01–07 complete and individually green.

## Overview

- Date: 2026-07-24
- Description: prove safety, durability, power benefit, p95 wake targets, topology parity, and reversible rollout.
- Priority: P1
- Implementation status: Pending
- Review status: Pending

## Key Insights

- Workstation `linux/amd64` Docker verifies packaging/topology, not physical ARM camera/audio/KWS acceptance.
- CPU/RSS claims require target Rover baselines; RSS is evidence, not a hard reclamation promise.
- Fault injection must cover restart, partition, duplicate/reorder, deadline, disk, Mongo, and lifecycle partial failure.

## Requirements

- Gates: WakeAck <1.5 s p95; NormalRover Ready <5 s p95; measured ScheduledCapture p95 + margin; zero stale replay.
- Auto requires demand-free five minutes and fresh consecutive below-threshold samples for every affected domain.
- Benchmark `Awake`, `NormalRover`, `IdleListening/KWS`, `ScheduledCapture`, `Dormant`; report CPU/RSS/thermal/power proxy/device/model release.
- Validate split Orchestra/Rover, direct Rover, workstation Docker/Podman, and physical target Rover.
- Validate 90-day TTL/query, current projection monotonicity, local journal recovery/full behavior, auth/entity/rate gates.
- Roll out observe-only → Awake-only → scheduler/manual Sleep → Auto → KWS; every stage has flag-based rollback.

## Architecture

One deterministic fault harness drives contract fixtures and fake clocks in-process, then Dora/Zenoh topology tests, workstation containers, and physical Rover acoustic/resource runs. Release evidence records build SHA, model/phrase checksums, config, topology, hardware, raw metrics, p50/p95/p99, and pass/fail.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/test-power-coordinator-faults.sh`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/benchmark-rover-power-profiles.sh`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/benchmark-rover-kws.sh`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/test-data/power/{noise-corpus-manifest.json,fault-scenarios.json}`; keep raw audio outside Git if licensing/privacy requires.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Makefile` — contract, fault, Docker, benchmark targets.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/{Dockerfile.orchestra,Dockerfile.rover-kiwi,docker-compose.yml,docker-compose.workstation.yml}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docs/{power-coordinator-architecture.md,codebase-summary.md}` and root `ARCHITECTURE.md` only for verified implementation drift/results.
- Add external UI Playwright/Vitest coverage under `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.

## Implementation Steps

1. Run Rust unit/contract/property/integration suites: ledger commutativity, authority order, barriers, journal crash points, Mongo indexes/TTL/projection, scheduler restart, KWS safety.
2. Run TypeScript type/lint/build/Vitest and fake-Socket.IO/Playwright: auth, exact entity, reconnect, live-over-cold, stale/reorder, Wake→Auto, history pagination.
3. Validate workstation Docker-compatible stack. Export `XDG_RUNTIME_DIR=/run/user/$(id -u)`; require `docker info`, real build/run smoke, full compose profiles, health/log/top checks.
4. Validate direct dataflow and split dataflows with identical contracts; restart each coordinator/manager/bridge/scheduler/projector independently.
5. Inject Zenoh bridge outage/reconnect, delayed/reordered/duplicate messages, transition timeout, missing target, Mongo outage, journal torn/full, resource staleness, reservation invalidation.
6. On physical Rover, run ≥30–60 minute profile/acoustic trials. Measure domain CPU/RSS, thermals/power proxy, release evidence, KWS latency, false accepts/hour, false rejects by distance/noise.
7. Derive per-domain CPU thresholds, consecutive fresh-sample count, minimum awake hold, journal cap, takeover timeout, bootstrap p95, and safety margin from evidence; rerun five-minute Auto tests.
8. Roll out flags `POWER_COORDINATOR_ENABLED`, `POWER_OBSERVE_ONLY`, `POWER_AUTO_ENABLED`, `POWER_KWS_ENABLED`, `POWER_HISTORY_ENABLED`; advance only on stage gates and document rollback.
9. Update architecture/docs to verified behavior, preserve raw benchmark report, and remove legacy lease/scheduler fallback only after stable release window.

## Todo list

- [ ] Pass Rust/Mongo/browser contract gates.
- [ ] Pass workstation Docker and direct/split topology gates.
- [ ] Pass restart/partition/reorder/disk/database fault matrix.
- [ ] Pass physical Rover power/KWS/p95 acceptance.
- [ ] Freeze deployment thresholds/timeouts/capacities.
- [ ] Complete staged rollout and rollback drill.

## Success Criteria

- WakeAck and NormalRover p95 SLAs pass on physical Rover; scheduler has no readiness-start violation.
- Auto sleeps only after five demand-free minutes plus fresh low CPU per domain; stale sample blocks it.
- No stale actuator/media/power replay across restart or partition; all injected partial failures avoid false Ready/Dormant.
- Offline KWS wake, status-first Orchestra takeover, local journal replay, and Mongo non-regression pass.
- Docker workstation and direct/split modes pass; report explicitly distinguishes amd64 packaging from ARM hardware acceptance.
- Rollback flags restore prior manual lifecycle/recording behavior without data loss.

## Risk Assessment

- Benchmarks vary by heat/noise: control corpus, ambient conditions, warm-up, repeated percentile runs.
- Docker compatibility assumptions fail on Fedora: use Podman socket/runtime guidance and real smoke, not CLI presence alone.
- Staged dual path diverges: keep migration window short and compare observe-only decisions.

## Security Considerations

- Run negative auth/rate/entity/source tests and verify audit redaction.
- Test forged epoch/source, oversized payload/history range, malicious detail, journal permission, and model checksum failure.
- Acoustic corpus consent/licensing documented; raw audio never enters Mongo events.

## Next steps

After all gates, mark plan completed and publish evidence links. Unresolved until hardware run: exact phrase/model, false-wake limits, CPU thresholds/sample count, minimum awake hold, journal size, and takeover/prewarm timings.
