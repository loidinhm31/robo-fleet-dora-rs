# Phase 01 — Contracts and Decision Freeze

## Context links

- [Parent plan](./plan.md)
- [Brainstorm](../reports/brainstorm-260720-0233-recording-scheduler-dora-node.md)
- [Backend research](./research/researcher-01-backend-scheduler-report.md)
- [Integration research](./research/researcher-02-integration-ui-deployment-report.md)
- [Architecture](../../ARCHITECTURE.md)
- Existing Rust contract: `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/recording_types.rs`
- Existing TypeScript contract: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording.ts`

## Overview

- Date: 2026-07-20
- Description: Freeze protocol, ownership, state machines, authentication, limits, and compatibility before runtime work.
- Priority: P1
- Implementation status: Done — 2026-07-20 13:40 +07 (UTC+0700)
- Review status: Approved — fixture path/type validation follow-up assigned to Phase 2/CI
- Effort: 7h

## Key Insights

- Current `recording_session_*` is manual session control; scheduling needs a separate family.
- Current recorder retains original request ID but generates random recording ID. Deterministic request identity enables adoption without changing clip identity.
- Browser fields are untrusted. Web bridge must validate the login session and inject the authenticated audit actor.
- Existing scheduler override drafts encode retired rover leases. They are migration clues only.

## Requirements

- Freeze schedule fields: UUID, revision, entity, title, enabled, recurrence, local start/date, IANA timezone, elapsed duration, safe relative-directory template, audit fields.
- Freeze occurrence fields: deterministic ID, schedule revision, planned bounds, DST resolution, lifecycle, retry metadata, group/start-request/recording IDs, suppression, errors.
- Recurrence: one-time, daily, weekly with non-empty ISO weekday set.
- State machine: `planned -> due -> start_pending -> active -> stop_pending -> completed`; terminal `suppressed|missed|failed|cancelled`.
- Schedule CAS: create revision 1; update/enable/disable/delete require expected revision; conflict returns authoritative schedule.
- Default limits still to approve: duration 1 minute–24 hours, title 1–128 chars, template <=240 chars, bounded future date, maximum enabled schedules per rover.
- Any authenticated user may query, create, update, enable, disable, and delete schedules in v1. Scheduler RBAC is out of scope.
- Manual start during a scheduled group suppresses current owners, finalizes the scheduled clip, then starts the requested manual session.
- Give every terminal occurrence/audit record a 90-day TTL; never TTL active or nonterminal state.
- Retry transient failures after 1, 2, 4, 8, and 16 seconds, then every 30 seconds until window end; no attempt cap. Retry alerting is future scope.
- One logical occurrence may reference multiple failed, partial, and recovered clip attempts.
- A missing scheduler process fails health; Mongo/reconciliation failure degrades scheduling only and preserves manual control.

## Architecture

- Socket.IO client -> server: `recording_schedule_command` and entity-scoped `recording_schedule_query`.
- Socket.IO server -> client: `recording_schedule_command_result`, `recording_schedule_snapshot`, `recording_occurrence_status`.
- Dora web -> scheduler: command/query, coordinator feedback, reconciliation snapshot.
- Dora scheduler -> web: result/snapshot/status, scheduled intent, reconciliation request.
- Recorder reconciliation: add internal active-session snapshot request/result; preserve manual start/stop shapes.
- IDs: `occurrence_id = UUIDv5(schedule_id, revision, planned_start_ms)`; stable group/start request UUIDv5; random recorder UUIDv4.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/recording_schedule_types.rs` — schedule/recurrence wire types.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/recording_occurrence_types.rs` — occurrence, intent, feedback, snapshots.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/recording_schedule_validation.rs` — focused validators; keep files near 200 LOC.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/mod.rs` — exports and tests.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/Cargo.toml` — UUID v5/timezone-safe dependencies only if shared validation needs them.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/tests/fixtures/recording-schedule-v1.json` — cross-language canonical fixtures.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording-schedule.ts` — exact TypeScript mirror.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts` — typed event names.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/index.ts` — exports.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md` — already seeded; finalize decisions after review.

## Implementation Steps

1. Encode the confirmed manual replacement, overlap, authentication, retention, retry, crash recovery, and health decisions in a contract table.
2. Define version-1 Rust types and validation without runtime behavior.
3. Define internal scheduled intent/reconciliation types separately from browser input.
4. Define bounded reason codes and error detail.
5. Mirror public contracts in TypeScript; deprecate obsolete lease draft only after import search.
6. Add accepted/rejected/conflict/snapshot fixtures in Rust and linked UI tests.
7. Review authority of every field: browser, web-injected, scheduler-generated, recorder-generated.
8. Record remaining numeric limits and operational questions before Phase 2.

## Todo list

- [x] Six revalidation decisions frozen and documented.
- [x] Rust types/validators compile and round-trip.
- [x] TypeScript types match fixtures.
- [x] Invalid protocol/UUID/timezone/path/revision rejected.
- [x] Audit actor absent from browser mutation input and derived from the authenticated session.
- [x] Existing manual fixtures unchanged.

## Success Criteria

- `cargo test -p robo_rover_lib recording_schedule` passes.
- Linked app shared-package type-check and fixture tests pass.
- Rust JSON equals TypeScript fixture JSON.
- Review confirms scheduler cannot directly express rover camera/audio control.
- Manual `recording_session_*` compatibility tests remain green.

## Risk Assessment

- Contract bloat: split public schedule, occurrence, and internal coordinator types.
- Hidden obsolete UI consumer: search imports before deprecating drafts.
- Premature recorder API changes: add only snapshot/reconciliation fields proven necessary.
- Manual replacement sequencing risk: require terminal scheduled status before admitting the manual recorder start.

## Security Considerations

- Never accept audit actor, recording ID, retry state, or occurrence state from browser.
- Require a valid login for every schedule query and mutation; no scheduler role branches in v1.
- Validate IDs, lengths, timezone, recurrence, duration, weekdays, path, protocol version.
- Bound error detail to 256 chars and sanitize storage/Mongo paths.
- Entity-scope every query and broadcast.

## Next steps

- Approved contract unlocks [Phase 02](./phase-02-scheduler-core-recurrence-and-persistence.md).
- UI fixture work can prepare [Phase 05](./phase-05-scheduler-ui-and-client-state.md).
- Phase 2/CI must add an automated check that resolves both canonical fixture paths and verifies the JSON fixture shape/type before dependent tests run.

## Unresolved questions

1. Approve duration, future-date, and per-rover schedule-count limits.
2. Should v1 show only the selected rover or all fleet rovers to a logged-in user?
