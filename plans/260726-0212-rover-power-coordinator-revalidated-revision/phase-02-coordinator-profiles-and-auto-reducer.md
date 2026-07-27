---
title: "Coordinator Profiles and Auto Reducer"
description: "Deliver role-scoped power profiles, bounded demand accounting, protected-work vetoes, readiness barriers, and the five-minute Auto reducer gate."
status: completed
priority: P1
effort: 40h
branch: main
tags: [feature, backend, power-coordinator, rover, orchestra]
created: 2026-07-26
---

# Phase 02 — Coordinator Profiles and Auto Reducer

## Context links

- Parent: [plan.md](./plan.md); design: [policy and profiles](../../docs/power-coordinator-architecture.md#policy-and-effective-state).
- Evidence: [control research](./research/researcher-01-control-scheduler-revision.md).
- Dependency: Phase 01 contracts/fencing.

## Overview

- Date: 2026-07-27; priority: P1; implementation/review: DONE — accepted 2026-07-27T13:32:39+07:00.
- Add deterministic deployment-local policy, demand ledger, reviewed profiles, readiness barriers, and Auto hysteresis.

## Carryover audit

The coordinator, demand ledger, resource-domain evidence, transition planner,
barriers, and most reducer tests exist. The phase is not accepted because:

- no runtime input owns `protected_operation`;
- Orchestra explicit Sleep selects `Dormant`, which the role contract rejects;
- configuration can reduce the five-minute Auto gate below five minutes;
- `ScheduledCapture` and the always-on inventory omit required
  recorder/encoder/control owners;
- demand TTL/capacity is global rather than bounded per source.

## Key Insights

- Policy, requested profile, and effective profile are distinct. Resource CPU confirms semantic idleness; it never creates it.
- Auto must settle in `IdleListening`; only explicit Sleep reaches `Dormant` and disables KWS.

## Requirements

- Create one `common/power-coordinator` instance per deployment role with fresh `Awake` startup and transient policy/demand reset.
- Profiles are role-scoped. Rover uses `Dormant`, `IdleListening`,
  `ScheduledCapture`, and `NormalRover`; Orchestra uses `OrchestraSpeech` plus
  a contract-valid low-power mapping decided in this phase. Controllers,
  bridges, scheduler, monitors, coordinator, lifecycle manager, web bridge,
  watchdog, and emergency path are always on.
- Auto requires no demand/protected work, five demand-free minutes, and consecutive fresh low-CPU samples per affected domain. Missing/stale evidence blocks sleep.
- One transition per domain; new demand/high or stale CPU cancels idle. Use bounded retry/backoff and minimum awake hold.

## Architecture

`DemandLedger + policy + protected work + freshness -> pure reducer -> requested profile -> static dependency transition planner -> aggregate authoritative readiness`. Sleep closes dependents first; wake opens prerequisites first.

## Related code files

- Modify `common/power-coordinator/src/{config.rs,demand_ledger.rs,profiles.rs,state_machine.rs,transition_planner.rs,readiness.rs,main.rs}` and focused tests.
- Modify shared resource contracts and `common/resource_monitor/src/{config.rs,resource_sampler.rs,process_resolver.rs,main.rs}`.
- Modify lifecycle/dataflow adapters only where required for protected-work and
  role-valid profile behavior.

## Implementation Steps

1. Complete role-scoped profile catalogs and validate acyclic dependencies,
   always-on vetoes, and every emitted profile/status against the role contract.
2. Finish bounded per-source acquire/renew/release/expiry behavior keyed by entity and demand ID.
3. Wire protected-work truth and correct the pure reducer: Rover Auto reaches
   `IdleListening`, Rover explicit Sleep reaches `Dormant`, and Orchestra Sleep
   uses the newly frozen contract-valid mapping.
4. Apply phase barriers through lifecycle manager and publish requested/effective/failed aggregate state only from authoritative terminal status.
5. Retain configured domain mapping/freshness/sequence/CPU/RSS and enforce a
   five-minute minimum Auto grace with monotonic fake-clock tests.

## Todo list

- [x] Add coordinator crate, baseline profile validation, and reducer tests.
- [x] Add domain CPU freshness evidence and staged lifecycle barriers.
- [x] Wire protected-work ownership into the runtime contract/dataflow.
- [x] Make Orchestra Sleep contract-valid and enforce the five-minute floor.
- [x] Complete profile and always-on ownership inventories.
- [x] Add per-source TTL/capacity and missing retry/restart/profile tests.

## Success Criteria

- Auto never reaches Dormant; explicit Sleep never leaves KWS active.
- Every requested/effective profile and status validates for its deployment
  role, including Orchestra explicit Sleep.
- A demand/protected operation cannot coexist with automatic quiesce of its required domain.
- Stale/missing/high resource evidence blocks Auto; partial transitions never claim Ready/Dormant.

## Risk Assessment

- Incomplete ownership inventory can quiesce a safety dependency; profile validation and tests must reject it.
- Incorrect CPU mapping must block, not imply zero usage.

## Security Considerations

- Source allowlists, per-source TTL/capacity, and static always-on constraints are enforced by coordinator, not UI.

## Next steps

Phase 03 persists intent before enabling real transitions. Threshold values remain target-hardware decisions.

## Completion status

All Phase 02 carryover gaps and success criteria are accepted: role-valid
profiles (including Orchestra Sleep), protected-work ownership, bounded
per-source demand TTL/capacity, complete always-on inventories, readiness
barriers, and the five-minute Auto grace are implemented and covered by tests.
