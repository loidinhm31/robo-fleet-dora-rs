---
title: "Fleet resources and safe node soft-stop"
description: "Replace synthetic fleet performance data with resource-only monitoring and add authenticated node quiesce/unload control across Orchestra and Rover."
status: in_progress
priority: P1
effort: 36h
branch: main
tags: [feature, frontend, backend, observability, lifecycle, safety]
created: 2026-07-21
---

# Fleet Resources and Safe Node Soft-Stop

## Overview

Remove FPS, inferred latency, queue, and drop values from fleet monitoring. Measure only scoped CPU and memory for Orchestra/Rover systems and configured processes. Replace global `performance_control` with authenticated, target-pinned, acknowledged node pause/resume. Soft-stop quiesces work and unloads supported resources; it does not use `SIGSTOP` or guarantee OS RSS reclamation.

## Decisions

- Rename wire contracts/events to `resource_*`; no synthetic-field compatibility.
- Keep CameraViewer stream FPS: separate local stream diagnostic, not Fleet Resources.
- Run one lightweight resource monitor on Orchestra and one per Rover.
- Expose individual node control only from server capabilities/allowlist; unsafe nodes locked.
- Always on: bridges, web bridge, lifecycle managers, resource monitors, scheduler, actuator controllers, emergency path.
- Pause cancels active work through node-specific terminal/finalization contracts; after a global 30 s deadline it force-tears down, emits explicit operation failure/loss, and reports Quiesced only after actual release.
- Each schedule occurrence gets one recorder-only wake attempt; upstream media uses existing demand, failure is not retried, user Pause wins immediately, and final lease release reconciles latest user intent.
- Any signed-in user may control lifecycle; session validation, rate limit, exact target validation, and audit remain mandatory.
- Soft-stop state is runtime-only; full restart defaults Running.
- Monitoring stays active while workloads pause.

## Phases

| # | Phase | Status | Effort | Link |
|---|---|---|---:|---|
| 1 | Resource-only contracts and collectors | Done (2026-07-21T22:35:04+07:00; review approved) | 6h | [phase-01](./phase-01-resource-only-contracts-and-collectors.md) |
| 2 | Lifecycle control plane and routing | Done (2026-07-22T00:04:24+07:00; review approved) | 8h | [phase-02](./phase-02-lifecycle-control-plane-and-routing.md) |
| 3 | Safe node adapters and unload/resume | Pending | 12h | [phase-03](./phase-03-safe-node-adapters-and-unload-resume.md) |
| 4 | Fleet Resources UI and authoritative state | Pending | 6h | [phase-04](./phase-04-fleet-resources-ui-and-state.md) |
| 5 | Cross-mode verification and rollout | Pending | 4h | [phase-05](./phase-05-cross-mode-verification-and-rollout.md) |

## Architecture and Research

- [Architecture](../../ARCHITECTURE.md#planned-resource-monitoring-and-soft-stop-lifecycle)
- [Lifecycle research](./research/researcher-01-lifecycle-control.md)
- [Resource/UI research](./research/researcher-02-resource-metrics-ui.md)
- [Original assessment](../reports/ask-260721-1354-fleet-performance-monitoring-assessment.md)

## Dependencies

- Coordinated Rust and separate `robo-control-app` checkout release.
- Existing Socket.IO auth/rate limiting, fleet target validation, Dora, and Zenoh bridges.
- Explicit process manifest for native and container deployments.

## Validation Summary

**Validated and revalidated:** 2026-07-21
**Questions asked:** initial 8 prompts/6 decisions; revalidation 8 prompts/6 decisions

### Confirmed Decisions

- FPS removal: Fleet Resources/FleetSelector only; keep CameraViewer measured diagnostics.
- Control granularity: individual server-allowlisted safe nodes with automatic dependency sequencing.
- Savings proof: require CPU reduction and released model/device/buffer ownership; RSS/PSS is evidence, not a hard gate.
- Busy policy: cancel active work through explicit terminal contracts, then pause.
- Schedule policy: auto-resume paused media per occurrence, revalidate every run, restore pause after completion.
- Authorization: any authenticated user, with rate limit, exact-target validation, and audit.
- Failure policy: one global 30 s transition deadline, then force teardown with explicit failed/lost work status.
- CPU gate: benchmark target hardware first, then freeze the numeric reduction threshold; deterministic release remains mandatory.
- Schedule recovery/scope: recorder-only lease, existing upstream media demand, one attempt per occurrence, no retry.
- Precedence: explicit user Pause revokes the active automation lease and cancels/finalizes immediately.

### Action Items From Revalidation

- [ ] Revise Phase 03 from hold-Degraded-on-timeout to explicit force teardown after 30 s without false Quiesced.
- [ ] Specify recorder-only lease versus upstream media-demand ownership and cleanup in Phases 02–03.
- [ ] Add benchmark procedure/output gate to Phase 05; freeze the numeric CPU threshold from evidence before release acceptance.
