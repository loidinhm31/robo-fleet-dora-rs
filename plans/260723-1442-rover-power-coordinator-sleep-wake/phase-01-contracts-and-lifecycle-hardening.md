# Phase 01 — Contracts and Lifecycle Hardening

## Context links

- Parent: [plan.md](./plan.md)
- Design: [power coordinator architecture](../../docs/power-coordinator-architecture.md)
- Inputs: [brainstorm](../reports/brainstorm-260723-1442-rover-power-coordinator-sleep-wake.md), [authority research](./research/researcher-01-power-authority-journal.md)
- Dependency: none; blocks every later phase.

## Overview

- Date: 2026-07-26
- Description: freeze Rust/wire contracts and make lifecycle execution safe for coordinator control.
- Priority: P1
- Implementation status: Complete
- Review status: Complete

## Completion update

Completed 2026-07-26. Version-1 power contracts, validators, canonical JSON fixture, and negative/idempotency/stale-order tests are in place. Lifecycle coordinator fencing now enforces exact-target transition IDs, fixed deadlines, terminal timeout protection, and replay tombstones; legacy lease wiring remains compatibility-only for the Phase 05 cutover.

## Key Insights

- Existing epoch/revision CAS, exact targets, admission/applied split, and late-status fencing are reusable.
- Wake leases are compatibility code, not the new public power contract; scheduler/UI must migrate to demands.
- Contract fixtures must precede Rust, Zenoh, Socket.IO, Mongo, and TypeScript integrations.

## Requirements

- Define version-1 policy, profile, demand/reservation, authority, transition, status, event, and bounded reason-code types.
- Validate exact entity/role, UUIDs, timestamp order, TTL/capacity, renew sequence, profile/source combinations, and sanitized detail.
- Duplicate ID + identical immutable payload = idempotent; changed payload = conflict.
- Lifecycle accepted never means applied; timeout begins once and late/foreign status cannot clear it.
- Retain legacy wake lease only behind compatibility wiring until Phase 05 cutover.

## Architecture

`PowerCommand/PowerDemand -> power coordinator -> fenced LifecycleCommand -> LifecycleStatus -> aggregate PowerStatus`. Authority tuple is `(epoch, sequence)`; status additionally carries transition ID and requested/effective profile. Cross-language JSON fixtures are canonical.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/power_types.rs` — protocol enums/envelopes and validation.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/power_contract_tests.rs` — fixtures, bounds, stale/idempotency cases.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/mod.rs` — exports.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/lifecycle_types/{command.rs,lease.rs,status.rs,validation.rs}` — coordinator origin/fencing and compatibility deprecation.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/lifecycle_manager/src/{manager.rs,main.rs,lib.rs}` — transition emission, fixed deadlines, tombstones, status fanout tests.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/test-data/contracts/power-v1.json` — Rust/TypeScript golden fixture.

## Implementation Steps

1. Write contract matrix: producer, target, authority fields, mutable renew fields, expiry, terminal result, max lengths/capacities.
2. Implement `PowerPolicy`, `PowerProfile`, `PowerDemandAction/Source/Priority`, `PowerState`, `PowerReasonCode`, command/result/status/reservation/event envelopes.
3. Add validators: protocol=1, bounded ASCII identifiers, UUIDs, exact role/entity, `issued <= not_before < expires`, renew monotonicity, 256-byte sanitized detail.
4. Add serde golden/negative/property-style tests for duplicate, reorder, expiry, wrong entity, wrong epoch, and unknown enum values.
5. Harden lifecycle manager so effective changes always emit one authorized exact-target command; terminal/timeout state is revision-fenced.
6. Keep revoked lease tombstones until their generation/expiry fence; prevent delayed acquire revival and deadline refresh during reconciliation.
7. Mark lease APIs deprecated and add tests proving coordinator contracts do not serialize as lifecycle leases.

## Todo list

- [x] Freeze v1 field names and bounds.
- [x] Add shared types, validators, fixtures.
- [x] Harden lifecycle transition/deadline behavior.
- [x] Add stale/reorder/idempotency tests.
- [x] Review always-on target allowlist and legacy lease removal gate.

## Success Criteria

- Rust golden fixture round-trips exactly; malformed/cross-entity inputs fail closed.
- Same command/request returns same outcome; changed duplicate and lower epoch/revision are rejected.
- Timeout cannot be refreshed by reconcile; late success cannot overwrite timeout.
- Quiesce/release/resume emits at most one exact-target transition per revision.
- Existing lifecycle and recording tests remain green.

## Risk Assessment

- Contract churn multiplies downstream work: freeze before Phase 02.
- Removing leases early may break scheduled recording: preserve compatibility until cutover.
- More lifecycle statuses may overflow queues: size from configured target count and test saturation.

## Security Considerations

- Actor identity never comes from browser payload.
- Reject unknown source/profile combinations, excessive TTL, unbounded detail, and non-active exact targets.
- Power messages are separate from actuator/media commands; no payload embeds executable commands.

## Next steps

Phase 01 is complete; proceed with Phase 02 coordinator core and profile/Auto behavior. Legacy lease removal remains gated on the Phase 05 scheduler cutover.
