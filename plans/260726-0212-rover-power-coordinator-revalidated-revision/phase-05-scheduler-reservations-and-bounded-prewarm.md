# Phase 05 — Scheduler Reservations and Bounded Prewarm

## Context links

- Parent: [plan.md](./plan.md); design: [scheduled wake sequence](../../docs/power-coordinator-architecture.md#scheduled-wake-sequence).
- Evidence: [control/scheduler research](./research/researcher-01-control-scheduler-revision.md), [scheduler scout](./scout/scout-02-scheduler-voice-ui.md).
- Dependencies: Phases 01–04.

## Overview

- Date: 2026-07-26; priority: P1; implementation/review: Accepted (2026-07-27).
- Keep recording scheduler as occurrence authority while making it a deterministic future-demand producer with aggregate-Ready gating.

## Key Insights

- Scheduler already has durable occurrence/group/outbox recovery, generation validation, manual suppression, and bounded retry patterns.
- Reservation accepted is never Ready. Power coordinator chooses nodes; scheduler never starts lifecycle work or FFmpeg directly.

## Requirements

- Derive deterministic reservation ID from occurrence/group generation; register, renew, release, and replay idempotently.
- Calculate prewarm as `planned_start - max(bootstrap, rolling profile p95) - safety_margin`; expose sample count, estimate, actual, and miss metrics.
- Final validation checks revision, window, entity, recorder, storage, media authority, and power readiness before existing recorder acquire.
- Delete/edit/supersession/manual suppression/terminal feedback/missed window release immediately. Only allowlisted transient recorder/storage errors retry with existing bounded delays before planned end.

## Architecture

Scheduler stores reservation state beside occurrence/outbox. Coordinator accepts future demand without waking, then reports `accepted|prewarming|ready|blocked|failed`; scheduler performs final validation and emits recorder intent only after `ready`.

## Related code files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/{domain.rs,state_machine.rs,runtime.rs,runtime_groups.rs,node_intents.rs,node_loop.rs,service_actions.rs}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/{mongo_documents.rs,mongo_repository.rs,node_persistence.rs,ports.rs}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/{scheduled-recording-coordinator.rs,recording-schedule-gateway.rs,recording-schedule-feedback-spool.rs,main.rs}`.
- Modify Orchestra dataflow and scheduler tests under `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/tests/`.

## Implementation Steps

1. Materialize reservation with occurrence generation; recover/register it through scheduler outbox before due admission.
2. Add power demand/status ports and conservative p95 estimator with a reviewed bootstrap until sample floor.
3. Request `ScheduledCapture` at computed prewarm; propagate authority/transition IDs and blocked/failed reasons.
4. Classify final validation failures: release permanent invalidation immediately; retry only explicit transient recorder/storage conditions within the current window.
5. Release exactly once on every terminal/final-owner path and reconcile the then-current Awake/Auto/Sleep policy.
6. Add fake-clock, overlap, restart, outbox, late Ready, invalid-after-Ready, and retry-classification tests.

## Todo list

- [x] Add reservation occurrence/outbox lifecycle.
- [x] Add readiness estimator and final validation gate.
- [x] Add release/retry classifier and recovery tests.

## Success Criteria

- Accepted/prewarming/blocked reservation never starts recording.
- Edit/delete/supersession cannot leak demand or revive an occurrence.
- Transient retry ends at window deadline; non-transient causes do not churn power transitions.

## Risk Assessment

- Sparse latency samples can under-prewarm; bootstrap is conservative until enough target evidence exists.
- Overlap/final-owner errors can leak power demand; group/generation-scoped ownership is mandatory.

## Security Considerations

- Scheduler source is local/allowlisted; reservation cannot contain paths, actuator commands, arbitrary profiles, or unbounded horizon.

## Next steps

Phase 07 exposes occurrence power status. Product wording may differ, but release behavior is fixed.

## Implementation status — 2026-07-27

An initial implementation added deterministic group reservation IDs, bounded
bootstrap/p95 estimation, aggregate-Ready recorder gating, and Dora wiring.
Focused validation passed (257 tests across scheduler, shared contract, power
coordinator, and web bridge), but review scored it 3/10 and rejected it.

Required before acceptance:

- durable register/release command outbox, exact result correlation, and replay;
- release tombstones that survive delete/edit/supersession until acknowledged;
- activation/prewarm-to-Ready rather than registration-to-Ready p95 samples,
  with sample/estimate/actual/miss observability;
- final validation before acquire, including current revision/window/entity,
  status freshness, and recorder/storage/media authority;
- an explicit recorder/storage transient retry allowlist and boundary fault tests.

The missing prerequisite is a signed, entity-scoped Rover-to-Orchestra
`PowerCommandResult` transport path. Current aggregate `PowerStatus` cannot
safely acknowledge an individual reservation command.

## Acceptance — 2026-07-27

Accepted after remediation and independent re-review (9/10, no critical
blockers). The completed work includes a durable, idempotent reservation-command
outbox; signed entity-scoped command-result transport; release tombstones;
reservation-scoped Ready evidence; activation-to-Ready p95 metrics with misses;
and final recorder, storage, and media-authority admission checks. Focused Rust
formatting, compilation, scheduler/shared/coordinator/web/Zenoh tests passed.
