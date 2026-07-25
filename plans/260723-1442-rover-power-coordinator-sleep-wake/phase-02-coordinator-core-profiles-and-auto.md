# Phase 02 — Coordinator Core, Profiles, and Auto

## Context links

- Parent: [plan.md](./plan.md)
- Design: [power coordinator architecture](../../docs/power-coordinator-architecture.md)
- Inputs: [authority research](./research/researcher-01-power-authority-journal.md), [voice/resource research](./research/researcher-02-voice-resource-ui.md)
- Dependency: Phase 01 contracts and lifecycle guarantees.

## Overview

- Date: 2026-07-26 (finalized)
- Description: implement deterministic demand ledger, static profile planner, aggregate readiness, and demand-first Auto policy.
- Priority: P1
- Implementation status: Complete — 100%
- Review status: Complete — final review 9/10; 0 critical/high findings.

## Key Insights

- Coordinator chooses policy/profile; lifecycle manager only executes exact targets.
- Static reviewed dependency phases are safer and simpler than runtime graph discovery.
- CPU confirms semantic idleness only; stale resource data must block automatic sleep.

## Requirements

- Run one common coordinator per deployment role with fresh `Awake` startup and transient demand reset.
- States: `Active`, `IdlePending`, `Quiescing`, `IdleListening`, `Dormant`, `Prewarming`, `Waking`, `Degraded`, `Failed`.
- Profiles: `Dormant`, `IdleListening`, `ScheduledCapture`, `NormalRover`, `OrchestraSpeech`.
- `Awake` holds normal; `Auto` selects least profile satisfying demands; `Sleep` quiesces normal work but permits UI Wake/scheduled reservation.
- Five-minute demand-free grace plus fresh consecutive per-domain low-CPU samples; memory is telemetry only.
- Serialize one transition per domain; cancel idle on demand/protected work/stale or high CPU; bounded retry/backoff and minimum awake hold.

## Architecture

`DemandLedger` reduces active bounded demands to required domains. `ProfileCatalog` expands domains into ordered lifecycle phases. `StateMachine` journals intent in Phase 03, closes admission, applies reverse dependency order for sleep/forward order for wake, and publishes Ready/Dormant only after every target terminal acknowledgment.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/power-coordinator/Cargo.toml`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/power-coordinator/src/{lib.rs,main.rs,config.rs,demand-ledger.rs,profiles.rs,state-machine.rs,transition-planner.rs,readiness.rs}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml` — workspace member/dependencies.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/resource_types.rs` and `resource_contract_tests.rs` — domain usage/freshness.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/resource_monitor/src/{config.rs,resource_sampler.rs,main.rs}` — static process-to-domain aggregation.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/lifecycle_manager/src/main.rs` — coordinator command/status ports.

## Implementation Steps

1. Define role-specific immutable profile catalogs and always-on spine; validate acyclic dependencies and forbid pausing controllers, watchdog, bridges, monitors, scheduler, or coordinators.
2. Build bounded demand ledger keyed by `(entity_id,demand_id)` with acquire/renew/release/expiry and immutable-payload idempotency.
3. Implement pure reducer from policy + demands + protected operations + resource freshness to requested profile/state.
4. Implement transition planner: close admission/safe-stop, quiesce dependents first; wake prerequisites first; barrier each phase on authoritative lifecycle status.
5. Implement timeout-to-Degraded, bounded retry, cancellation, and final profile reconciliation.
6. Extend resource snapshots with domain CPU/RSS, sample interval/sequence/freshness; map only configured process identities.
7. Implement Auto timers using monotonic time: five minutes demand-free and low CPU for configured consecutive fresh samples in every affected domain.
8. Add deterministic tests with fake clock for expiry, churn, stale samples, partial barriers, Sleep scheduled exception, and fresh Awake restart.

## Final validation (2026-07-26)

- `cargo test -p power_coordinator`: 11 passed, 0 failed.
- Targeted validation covers duplicate/reordered TTL handling, release tombstones, profile dependency ordering, fresh Awake restart, Auto grace/freshness gates, stale/high-CPU cancellation, partial readiness barriers, lifecycle revision fencing, and lifecycle-result supersession.
- Final independent review: 9/10; 0 critical/high findings.

## Todo list

- [x] Create coordinator crate and pure state reducer.
- [x] Add ledger bounds/idempotency/expiry.
- [x] Freeze profile dependency tables and always-on vetoes.
- [x] Add per-domain resource evidence.
- [x] Add transition barriers, retries, hysteresis, fake-clock tests.

## Success Criteria

- No demand/protected operation can coexist with automatic quiesce of its required domain.
- Auto cannot leave `IdlePending` before five minutes or with stale/high/missing CPU.
- New demand cancels idle immediately; duplicate/reordered demand cannot extend TTL.
- Partial failure reports Degraded/Failed, never false Ready/Dormant.
- Restart forgets transient policy/demands and reaches fresh `Awake` in safe dependency order.

## Risk Assessment

- Hidden device ownership can deadlock profiles: inventory ownership and test every profile barrier.
- CPU mapping may be incomplete: missing domain evidence blocks Auto rather than guessing zero.
- Repeated reload churn harms latency: minimum awake hold and bounded backoff.

## Security Considerations

- Only allowlisted producer/source pairs may create each demand class.
- Enforce per-source and per-entity capacity/TTL; expired input cannot revive state.
- Always-on safety veto is compile-time/profile validation plus runtime assertion.

## Next steps

Lifecycle-result-based supersession is complete: the coordinator records accepted
lifecycle revisions and reissues the replacement transition immediately when a
new demand arrives; an in-flight quiesce cannot win after demand arrival. The
regression test is `late_quiesce_acceptance_reissues_the_superseding_wake_without_timeout`.
Phase 03 adds write-ahead durability before enabling real transitions.
Benchmark-derived thresholds, consecutive sample count, and minimum awake hold
remain deployment values to freeze in Phase 08.
