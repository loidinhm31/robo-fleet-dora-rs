# Phase 01 — Contracts, Snapshot Gate, and Lifecycle Hardening

## Context links

- Parent: [plan.md](./plan.md); design: [power coordinator architecture](../../docs/power-coordinator-architecture.md).
- Evidence: [control research](./research/researcher-01-control-scheduler-revision.md), [power-stack scout](./scout/scout-01-power-stack.md).
- Dependency: none. Blocks every later phase.

## Overview

- Date: 2026-07-26; priority: P1; implementation/review: In progress — remediation.
- Freeze V1 shared contracts and expose a safe `AuthorityUnknown` boundary above existing lifecycle fencing.

## Carryover audit

The superseded Phase 01 commit is a useful baseline, not accepted completion.
Shared contracts, validation, lifecycle transition IDs, terminal fencing, and
lease tombstones exist. Revalidated acceptance still requires:

- integrate `PowerSnapshotGate` into the coordinator state-machine boundary so
  missing or stale snapshots suppress remote profile-command decisions;
- add immutable `command_id` replay caching: exact duplicate returns the same
  result and changed reuse fails closed;
- prevent reconciliation from refreshing an already-started lifecycle
  transition deadline;
- preserve reservation release/expiry tombstones and expand the fixture/test
  matrix to reservations and changed duplicates.

## Key Insights

- `LifecycleManager` already owns exact-target epoch/revision validation, timeout fencing, and late-status rejection; it must not become a global policy engine.
- Power contracts and the coordinator crate now exist as a partial baseline.
  Existing wake leases remain compatibility-only until scheduler cutover.

## Requirements

- Define policy, profile, demand/reservation, authority snapshot, transition, status, event, reason, and bounded-detail contracts in `robo_rover_lib`.
- Exact duplicate immutable request succeeds idempotently; changed duplicate, expiry, wrong entity/role, stale epoch/sequence, and unknown enum fail closed.
- The coordinator state machine exposes a snapshot gate: missing/stale remote
  authority becomes `AuthorityUnknown`, and its command-effect reducer emits no
  remote power/profile command. Phase 04 owns production transport wiring.
- Expose a transport-agnostic typed decision (for example,
  `ObserveOnly | CommandAllowed`) from the authority reducer. Phase 01 tests
  that decision; Phase 04 maps it to bridge/direct outputs.
- The decision is one-shot and fail-closed: missing, expired, future-captured,
  wrong-role/entity, stale/equal-authority, reordered, or already-consumed
  snapshots return `ObserveOnly`; only a fresh matching snapshot plus a
  strictly newer proposed authority returns `CommandAllowed`, after which a new
  snapshot is required.
- `PowerSnapshotGate::observe` may retain typed validation errors for invalid
  input. The coordinator authority adapter must map every such error to
  `ObserveOnly` plus bounded reason telemetry; invalid input must not terminate
  the coordinator or produce a command effect.
- Lifecycle accepted is not applied; timeout begins once and late/foreign status cannot clear it.

## Architecture

`PowerCommand/Demand -> coordinator -> fenced LifecycleCommand -> authoritative status -> PowerStatus`. Authority compares `(epoch, sequence)`; `AuthorityUnknown` is an observable no-command state, not Dormant or Running.

## Related code files

- Modify `robo_rover_lib/src/types/power_types/{command.rs,demand.rs,status.rs,validation.rs}` and `power_contract_tests.rs`.
- Modify lifecycle contracts under `robo_rover_lib/src/types/lifecycle_types/`.
- Modify `common/power-coordinator/src/{state_machine.rs,demand_ledger.rs,main.rs}`.
- Modify `common/lifecycle_manager/src/{manager.rs,main.rs,lib.rs}`.
- Expand `test-data/contracts/power-v1.json`.

## Implementation Steps

1. Audit and finish JSON/Rust fixture fields, bounds, ID semantics, actor-free wire shape, reason codes, and capability/source matrix.
2. Add immutable command replay results plus reservation tombstones and expanded golden/negative fixtures.
3. Integrate the existing snapshot primitive into the coordinator reducer/API
   as a typed `ObserveOnly | CommandAllowed` decision (final name frozen with
   the contract); prove missing/stale state produces `AuthorityUnknown` and
   `ObserveOnly` without requiring Phase 04 transport. Map primitive validation
   errors to `ObserveOnly` and bounded telemetry at this adapter boundary.
4. Extend `test-data/contracts/power-v1.json` with table-driven authority-gate
   cases for missing, expired, future, wrong-target, stale/equal, reordered,
   consumed, and fresh-strictly-newer snapshots. Rust tests must consume the
   fixture rather than duplicate case data.
5. Retain coordinator-origin lifecycle fencing without changing user semantics.
6. Preserve revoked lease tombstones and prove lease-triggered reconciliation
   cannot refresh an active deadline or revive a released lease.

## Todo list

- [x] Land the baseline shared fixture and validators.
- [x] Add lifecycle transition IDs, terminal fencing, and lease tombstones.
- [ ] Integrate the snapshot gate into the coordinator state-machine boundary;
  leave production transport wiring to Phase 04.
- [ ] Freeze one-shot `ObserveOnly | CommandAllowed` semantics in the shared
  fixture and table-driven Rust tests.
- [ ] Add immutable command replay results and changed-duplicate rejection.
- [ ] Keep active lifecycle deadlines fixed across reconciliation.
- [ ] Add reservation tombstones plus expanded fixture/property-style tests.

## Success Criteria

- Cross-entity/stale/changed-duplicate power inputs fail closed.
- The coordinator authority harness returns `ObserveOnly` while its snapshot is
  absent/stale and `CommandAllowed` only after fresh reconciliation; Phase 04
  proves the mapping at production bridge edges.
- Every authority-gate fixture case passes, including one-shot consumption and
  strictly newer `(epoch, sequence)` ordering.
- Invalid snapshot validation never exits the coordinator and never produces a
  command effect.
- Timeout and terminal status remain revision-fenced; existing lifecycle tests remain green.

## Risk Assessment

- Contract churn fans out into Rust, Zenoh, Socket.IO, Mongo, and TypeScript; fixture review is a hard gate.
- Treating missing snapshot as either Running or Dormant is unsafe; preserve explicit unknown.

## Security Considerations

- Browser data never supplies actor privilege, source, authority epoch, or target fallback.
- Enforce bounded TTL, queue capacity, detail length, and source/profile combinations.

## Next steps

Start Phase 02 only after fixture and lifecycle review. Freeze snapshot retry/staleness values later from fault evidence.
