# Phase 08 — Fault Gates, Target Evidence, and Rollout

## Context links

- Parent: [plan.md](./plan.md); design: [acceptance targets](../../docs/power-coordinator-architecture.md#acceptance-targets).
- Inputs: both research reports and scouts in this plan directory.
- Dependencies: Phases 01–07.

## Overview

- Date: 2026-07-26; priority: P1; implementation/review: In progress (status review 2026-07-28).
- Prove authority safety, data durability, measured savings, acoustic behavior, and reversible release sequencing.

## Status note — 2026-07-28

Approved implementation automation is complete for the currently runnable
gates: automated validation passed, Mongo integration passed, and Podman/Docker
workstation preflight passed. Phase 08 remains **not complete**. Release is
blocked on physical ARM KWS/profile evidence, live direct and split-topology
evidence, an exclusive-stack smoke run, and staged rollout plus rollback drill.

## Key Insights

- amd64 Docker/Podman proves packaging/topology, not physical ARM camera/audio/KWS performance.
- CPU/RSS, p95 wake, snapshot staleness, retry allowlist, and journal capacity are target-derived operational values, not plan guesses.

## Requirements

- Pass `WakeAck <1.5 s p95`, `NormalRover Ready <5 s p95`, measured ScheduledCapture lead, five-minute/fresh-sample Auto, no stale replay, and journal-before-apply.
- Validate split/direct parity, Docker-compatible workstation, physical Rover, restart/partition/reorder, journal tail/full, Mongo outage, lifecycle partial failure, and schedule mutation during prewarm.
- Roll out observe-only → Awake-only → scheduler/manual Sleep → Auto → KWS with independent rollback flags.

## Architecture

One harness drives fake-clock/contract faults, then Dora/Zenoh topology, Docker-compatible stack, and physical Rover acoustic/resource runs. Evidence records SHA, hardware, topology, config, phrase/model checksum, p50/p95/p99, and outcome.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/{test-power-coordinator-faults.sh,benchmark-rover-power-profiles.sh,benchmark-rover-kws.sh}`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/test-data/power/{fault-scenarios.json,noise-corpus-manifest.json}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/{Makefile,docker/Dockerfile.orchestra,docker/Dockerfile.rover-kiwi,docker/docker-compose.yml,docker/docker-compose.workstation.yml}`.
- Modify verified behavior only in `/mnt/data/ws/sharing/robo-fleet-dora-rs/docs/power-coordinator-architecture.md`, `ARCHITECTURE.md`, and `docs/codebase-summary.md`; add UI Vitest/Playwright coverage externally.

## Implementation Steps

1. Run contract/unit/integration tests for ledger, authority unknown, lifecycle barriers, journal, projector, scheduler, KWS safety, and UI reducer.
2. Inject stale/duplicate/reordered commands, bridge and coordinator restarts, absent snapshot, partition, timeout, missing node, Mongo outage, torn/full journal, and edit/delete/supersession during prewarm.
3. Validate Docker-compatible stack with exported `XDG_RUNTIME_DIR`, `docker info`, real build/run smoke, compose profiles, health, logs, and process checks.
4. Run direct and split dataflows; independently restart bridges, managers, coordinator, scheduler, and projector.
5. On physical Rover run repeated 30–60 minute profile/noise trials; capture KWS errors, latency, thermals, power proxy, CPU/RSS, and release evidence.
6. Derive deployment thresholds, retry reason allowlist, journal reserve, snapshot retry/staleness, p95 bootstrap/margin; rerun gates.
7. Advance feature flags only after stage gates and a rollback drill; remove lease fallback after a stable release window.

## Todo list

- [ ] Pass contracts, browser, projector, Docker/direct/split gates (automated,
      Mongo, and Podman preflight portions passed; live topology evidence remains).
- [ ] Pass partition/restart/disk/database schedule fault matrix.
- [ ] Collect physical Rover KWS/power/profile evidence and freeze values.
- [ ] Run exclusive-stack smoke and complete staged rollout/rollback drill.

## Success Criteria

- No forced takeover or stale replay under injected faults.
- Every reservation lifecycle releases correctly; only approved transient faults retry before its window closes.
- ARM target meets wake/KWS evidence gates; workstation result is labelled packaging-only.
- Rollback returns to current lifecycle/recording behavior without data loss.

## Risk Assessment

- Noise/heat variance requires controlled corpus and repeated percentiles.
- Podman Docker compatibility requires a real smoke test, not CLI detection alone.

## Security Considerations

- Include forged epoch/source/entity, oversized payload/range, malformed detail, journal permission, and model checksum negative tests.

## Next steps

Mark complete only with evidence links and updated architecture-to-code review. Unresolved values remain explicit release blockers.
