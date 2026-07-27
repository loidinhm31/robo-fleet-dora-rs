# Phase 05 — Scheduler Reservations and Measured Prewarm

## Context links

- Parent: [plan.md](./plan.md)
- Design: [Scheduled Wake Sequence](../../docs/power-coordinator-architecture.md#scheduled-wake-sequence)
- Input: [brainstorm scheduler decision](../reports/brainstorm-260723-1442-rover-power-coordinator-sleep-wake.md)
- Dependencies: Phases 01–04 contracts, durability, routing/readiness.

## Overview

- Date: 2026-07-24
- Description: make scheduler a deterministic future-demand producer and gate recording start on aggregate Ready.
- Priority: P1
- Implementation status: Pending
- Review status: Pending

## Key Insights

- Scheduler remains sole schedule/occurrence authority; it never chooses nodes or starts lifecycle transitions.
- Reservation accepted is not profile ready.
- Prewarm derives from measured profile-ready p95 plus margin, not fixed 30 seconds.

## Requirements

- Deterministic reservation ID from occurrence/group generation; replay/renew/release idempotent.
- Sleep permits accepted scheduled reservation and wakes only `ScheduledCapture`.
- Coordinator prewarm time = `planned_start - max(bootstrap, rolling p95) - safety margin`; measured p95 replaces bootstrap after minimum sample count.
- Scheduler starts only after Ready and final revision/window/rover/storage/media revalidation.
- Terminal recorder feedback or invalidated occurrence releases reservation; readiness miss ends with bounded reason.
- Restart: scheduler rebuilds future reservations from durable occurrence/outbox state; coordinator does not persist transient reservation.

## Architecture

Scheduler materializes reservation alongside existing occurrence/outbox. Coordinator accepts now, waits until computed prewarm, transitions Orchestra/Rover prerequisites, reports `accepted|prewarming|ready|blocked|failed`. Existing scheduled recording coordinator receives acquire only after Ready.

## Related code files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/{domain.rs,state_machine.rs,runtime.rs,runtime_groups.rs}` — reservation lifecycle.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/{mongo_documents.rs,mongo_repository.rs,node_persistence.rs}` — durable rebuild source/outbox.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/{node_intents.rs,node_loop.rs,ports.rs,service_actions.rs}` — demand/status ports and Ready gate.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/{scheduled-recording-coordinator.rs,recording-schedule-gateway.rs,recording-schedule-feedback-spool.rs}` — post-Ready start/release feedback.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/tests/{runtime-reconciliation.rs,mongo-recovery.rs,mongo-integration.rs}`.

## Implementation Steps

1. Derive UUIDv5 reservation ID from entity, occurrence/group ID, and generation; store reservation state with occurrence, not a second schedule truth.
2. Extend scheduler outbox with register/renew/release intents and recover them before normal tick admission.
3. Register future reservation early; coordinator validates bounds/capability and returns accepted without waking.
4. Track per-profile successful readiness latency in a bounded rolling window; use p95 after configured minimum samples and explicit conservative bootstrap before it.
5. At computed prewarm, request `ScheduledCapture`; relay blocked/failed/ready with authority and transition IDs.
6. On Ready, revalidate schedule revision/window, group generation, active rover, recorder/storage, and media authority before emitting existing recording acquire.
7. Release on cancellation, supersession, manual suppression, terminal recorder feedback, or failed window. Reconcile to current `Awake|Auto|Sleep`.
8. Add fake-clock/recovery tests for overlap, late Ready, invalid-after-Ready, restart, duplicate status, Mongo outage, and final-owner release.

## Todo list

- [ ] Add reservation occurrence/outbox state.
- [ ] Add p95 estimator and bootstrap/margin config.
- [ ] Gate existing recording acquire on Ready.
- [ ] Release on every terminal/invalidation path.
- [ ] Add scheduler/coordinator recovery tests.

## Success Criteria

- No scheduled recording starts from accepted/prewarming/blocked status.
- Wake lead uses recorded p95 + margin; metrics expose estimate, sample count, actual latency, and misses.
- Duplicate/reordered reservation/status cannot create two sessions or regress occurrence.
- Sleep wakes minimal ScheduledCapture and returns to current policy after final release.
- Restart rebuilds future reservations and adopts/reconciles active recording without blind duplicate start.

## Risk Assessment

- Sparse latency data underestimates wake: require conservative reviewed bootstrap until sample floor.
- Ready arrives after occurrence invalidation: final revalidation and immediate release.
- Overlapping occurrences churn profiles: group-scoped demand and final-owner release.

## Security Considerations

- Only local authenticated scheduler route may use scheduler source.
- Reservation cannot carry paths, actuator commands, or arbitrary profile.
- Sanitize readiness failure detail; bound reservation horizon/TTL/count.

## Next steps

Phase 07 exposes occurrence power status to UI. Product must decide whether invalid-after-Ready is shown as cancelled or power-suppressed; implementation must always release first.
