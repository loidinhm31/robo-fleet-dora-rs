# Rover power plan cutoff audit

- Date: 2026-07-26
- Old plan: `plans/260723-1442-rover-power-coordinator-sleep-wake`
- Active plan: `plans/260726-0212-rover-power-coordinator-revalidated-revision`

## Outcome

Stop the old plan after its claimed Phase 03 and do not begin old Phase 04.
Retain commits `ff6624e`, `a9ba1c4`, and `a1cbc38` as a partial implementation
baseline. Revalidated Phase 01 is the next executable phase; Phases 02–03 are
blocked carryover until their dependencies are reaccepted.

Passing focused tests demonstrate useful code, but do not prove the missing
acceptance paths.

## Phase 01 blockers

- `PowerSnapshotGate` is only defined/tested; the coordinator reducer does not
  consume it yet. Phase 01 owns that state-machine boundary, while Phase 04
  owns production transport (`robo_rover_lib/src/types/power_types/status.rs`).
- `PowerCoordinator` has no immutable command replay cache. A successful
  command bumps authority, so exact replay becomes stale; changed command-ID
  reuse is not fenced (`common/power-coordinator/src/state_machine.rs`).
- A new lease-triggered reconciliation while a target is already
  resuming/quiescing can assign a new deadline; ordinary component progress is
  already covered by a fixed-deadline regression test
  (`common/lifecycle_manager/src/manager.rs`).
- Reservation release/expiry lacks terminal tombstones and fixture coverage is
  incomplete (`common/power-coordinator/src/demand_ledger.rs`,
  `test-data/contracts/power-v1.json`).

## Phase 02 blockers

- `set_protected_operation` has no runtime caller or input.
- Orchestra Sleep selects `Dormant`, while the shared role contract allows
  only `OrchestraSpeech`.
- The five-minute Auto rule is a configurable default, not a validated floor.
- Scheduled-capture and always-on catalogs omit required workload/control
  owners.
- TTL/capacity is global rather than bounded per demand source.

## Phase 03 blockers

- Wake-causing commands use normal journal capacity, so the reserved slice is
  unreachable at the point command admission needs it
  (`common/power-coordinator/src/event-outbox.rs`).
- Command history drops status/action/demand/source/target context, and Mongo
  lacks demand/source indexes and filters.
- Projector write failure exits instead of bounded retry; Rover record/ack
  transport remains Phase 04 work.
- The Mongo integration test returns success when its URI is absent, so normal
  test output does not prove Mongo, TTL, outage, or cold-start behavior.
- Physical ENOSPC, full crash-point, and outage/reconnect gates are missing.

## Verification evidence

Fresh audit run:

`cargo test -p power_coordinator -p power_event_projector -p robo_rover_lib -p lifecycle_manager -p resource_monitor`

Result: 118 unit/integration tests plus one doctest passed; zero failures. The
Mongo test was skipped internally because `POWER_PROJECTOR_TEST_MONGODB_URI`
was unset.

## Next execution

Run the active plan from revalidated Phase 01. Phase 01 integrates the snapshot
gate into the coordinator state-machine boundary without transport. Reaccept
Phase 01, then Phase 02, then the expanded Phase 03 corrective checklist.
Phase 04 subsequently wires and proves the production Zenoh/direct snapshot
path; do not start it before those gates pass.

## Unresolved questions

- Is `Dormant` valid for Orchestra, or must Orchestra Sleep use a separate
  low-power profile?
- Which runtime component owns protected-operation truth?
- Which exact encoder/recorder nodes belong to each `ScheduledCapture`
  profile?
- Does full-disk acceptance require injected physical ENOSPC in addition to
  configured journal quota saturation?
