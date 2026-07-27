---
title: "Revalidated Rover Power Coordinator"
description: "Revise workload sleep/wake delivery around snapshot-safe authority, IdleListening, scheduled prewarm, local KWS, and durable history."
status: in-progress
priority: P1
effort: 254h
branch: main
tags: [feature, backend, frontend, database, infra, critical]
created: 2026-07-26
---

# Revalidated Rover Power Coordinator

## Overview

Implement the power coordinator specified in [the revalidated architecture](../../docs/power-coordinator-architecture.md). It remains workload-level only: no OS suspend, container/process kill, or voice-to-actuator command path.

## Revalidated decisions

- Auto rests in `IdleListening` after its five-minute demand-free gate; explicit Sleep reaches `Dormant`.
- UI activity creates a two-minute `NormalRover` demand. Any authenticated user may set Awake/Sleep; server-side target pinning, rate limits, expiry, and audit remain mandatory.
- Orchestra enters `AuthorityUnknown` and waits for a fresh Rover snapshot. It never force-takes authority or sends a profile command before reconciliation.
- Input KWS is `Hey Kiwi`; output-only bundled WakeAck is `I am on`. KWS never emits actuator, tracking, media, or recording commands.
- Scheduler delete/edit/supersession releases reservations immediately. Only allowlisted transient recorder/storage faults retry inside the occurrence window.
- Local journal intent precedes apply; Mongo stores 90-day events and a non-TTL monotonic current-state projection.

## Phases

| # | Phase | Status | Progress | Effort | Dependency |
|---|---|---|---:|---:|---|
| 1 | [Contracts, snapshot gate, lifecycle hardening](./phase-01-contracts-snapshot-gate-and-lifecycle.md) | Done — accepted 2026-07-27 | 100% accepted | 32h | — |
| 2 | [Coordinator profiles and Auto reducer](./phase-02-coordinator-profiles-and-auto-reducer.md) | Pending — ready after Phase 01 | 0% accepted | 40h | 1 |
| 3 | [Local journal and Mongo projection](./phase-03-local-journal-and-mongo-projection.md) | Blocked — partial carryover | 0% accepted | 34h | 1–2 |
| 4 | [Zenoh authority and direct-mode routing](./phase-04-zenoh-authority-and-direct-routing.md) | Pending | 0% | 34h | 1–3 |
| 5 | [Scheduler reservations and bounded prewarm](./phase-05-scheduler-reservations-and-bounded-prewarm.md) | Pending | 0% | 34h | 1–4 |
| 6 | [Rover KWS and WakeAck](./phase-06-rover-kws-and-wake-ack.md) | Pending | 0% | 30h | 1–4 |
| 7 | [Authenticated power API and UI](./phase-07-authenticated-power-api-and-ui.md) | Pending | 0% | 28h | 1–5 |
| 8 | [Fault gates, target evidence, rollout](./phase-08-fault-gates-target-evidence-and-rollout.md) | Pending | 0% | 22h | 1–7 |

Progress is revalidated acceptance progress, not estimated code volume. Existing
code is retained as a partial implementation baseline, but no carried phase is
accepted until its revised success criteria pass.

## Cutover audit

- Cut over on 2026-07-26 from the superseded plan after commits `ff6624e`,
  `a9ba1c4`, and `a1cbc38`.
- Phase 01 reacceptance completed 2026-07-27 (118 focused tests plus one
  doctest passing). Snapshot gating, immutable replay, deadline fencing, and
  reservation tombstones are accepted at the contract/state-machine boundary.
- Phase 02 is no longer blocked by Phase 01, but remains unaccepted and must
  address its own carryover gaps: protected-work input, valid Orchestra Sleep
  profile, and complete profile/source bounds.
- Phase 03 remains blocked by Phases 01–02. Wake-causing commands cannot use
  reserved journal capacity, event history loses command/demand context, and
  Mongo/outage coverage is not an enforced test gate.
- Phase 04 must not start until Phases 01–03 are reaccepted.

Phase 01 owns the snapshot gate contract and coordinator state-machine
behavior. Phase 04 owns Zenoh/direct ports, snapshot request/reply, and proof
that no production command leaves Orchestra before reconciliation. This split
avoids a Phase 01 ↔ Phase 04 dependency cycle.

## Release gates

- `WakeAck <1.5 s p95`; `NormalRover Ready <5 s p95`; scheduled lead uses measured p95 plus margin.
- Old, duplicate, cross-entity, expired, or reordered inputs never regress authority or effective state.
- Every applied transition has a prior synced local journal intent; Mongo/network outage never blocks safety or local wake.
- Split mode, direct mode, Docker-compatible workstation, partition/restart faults, and physical Rover trials pass.

## Inputs

- [Phase 01–03 cutoff audit](../reports/reconciliation-260726-0613-rover-power-plan-cutover.md) — current implementation baseline.
- [Prior validated plan](../260723-1442-rover-power-coordinator-sleep-wake/plan.md)
- [Control/scheduler research](./research/researcher-01-control-scheduler-revision.md) — historical pre-baseline evidence.
- [Voice/UI/history research](./research/researcher-02-voice-ui-history-revision.md) — historical pre-baseline evidence.
- [Power-stack scout](./scout/scout-01-power-stack.md) — historical pre-baseline inventory.

Before executing any phase, revalidate linked historical claims against
`HEAD`; the cutoff audit and current source override pre-implementation
inventory statements.

## Unresolved questions

- Pin a checksum-verified KWS model and target-hardware false-accept/false-reject limits for `Hey Kiwi`.
- Derive CPU thresholds, sample count, minimum awake hold, journal capacity, and snapshot staleness from Phase 8 evidence.
- Freeze the allowlist of recorder/storage reason codes that qualifies for bounded retry.
- Decide the valid Orchestra low-power profile/mapping for explicit Sleep.
