# Phase 03 — Web-Bridge Coordinator and Recorder Reconciliation

## Context links

- [Parent plan](./plan.md)
- [Phase 01](./phase-01-contracts-and-decision-freeze.md)
- [Phase 02](./phase-02-scheduler-core-recurrence-and-persistence.md)
- Existing registry: `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/media-demand-registry.rs`
- Existing recorder state: `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/recording-socket.rs`
- Existing session manager: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/session-manager.rs`

## Overview

- Date: 2026-07-20
- Description: Add authenticated schedule boundary, per-rover scheduled ownership coordinator, explicit recorder discovery, manual precedence, and restart adoption.
- Priority: P1
- Implementation status: Done (100%)
- Review status: Approved after fresh code review
- Effort: 14h

## Key Insights

- Web bridge already owns correct demand aggregation. Scheduled work must become another distinct consumer.
- Bridge caches and recorder active maps vanish on restart; explicit snapshot creates a reconciliation barrier.
- Matching deterministic start request ID plus entity distinguishes scheduled sessions; recording ID stays recorder-owned.
- Unknown active sessions are never stopped during recovery.

## Requirements

- Socket handlers require a valid login, rate-limit, validate, and inject the authenticated audit actor. All logged-in users receive the same scheduler CRUD permissions in v1.
- Schedule queues/status cache are bounded independently from manual recorder queues.
- Coordinator keys groups/consumers by entity/group/generation; duplicate intent is idempotent.
- Scheduled start acquires camera/JPEG/microphone through `MediaDemandRegistry`, then sends existing recorder start.
- Accepted result persists random recording ID through scheduler feedback; terminal status releases only scheduled consumer.
- Overlap intermediate owner changes never send start/stop or flap demand.
- Manual stop stops exact recording, suppresses current scheduled owners, and waits for next boundary.
- Manual start during a scheduled group suppresses current owners, sends stop for the exact scheduled recording, waits for terminal/finalized status, then starts the requested manual session.
- Manual replacement and scheduled OFF never release an unrelated manual/browser media hold.
- Reconnect pauses actions, requests desired/snapshot state, adopts/repairs, then resumes.
- Queue full, timeout, inactive rover, storage, startup, and encoder errors map to bounded feedback.
- Recorder crash feedback appends failed/partial/recovered attempts to the same logical occurrence.

## Architecture

- Add `RecordingScheduleGateway` for Socket.IO CRUD/query/result routing.
- Add pure `ScheduledRecordingCoordinator` reducer and `RecorderReconciler` comparison layer.
- Reconciliation:
  - Desired active + matching request/entity active: adopt ID and restore scheduled demand.
  - Desired active + no active: emit same deterministic start after barrier.
  - Desired inactive + known scheduled active: stop exact ID.
  - Unknown active: classify manual/foreign; never stop or adopt.
  - Multiple active per entity: fail closed and surface invariant violation.
- Recorder adds active-session snapshot request/result from `SessionManager::statuses()`; no filesystem guessing.
- Manual replacement is a serialized per-entity state: `scheduled_active -> suppressing -> finalizing -> manual_start_pending -> manual_active`.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/recording-schedule-gateway.rs` — CRUD/query routing.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/scheduled-recording-coordinator.rs` — ownership reducer.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/recording-reconciler.rs` — restart comparison.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/recording-schedule-queues.rs` — bounded queues/pending TTL.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs` — module wiring only; avoid further growth.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/security.rs` — reusable authenticated-claim/audit-actor helper; no scheduler role matrix.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/media-demand-registry.rs` — minimal restore/introspection helpers with tests.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/recording-socket.rs` — origin-safe shared transport.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/main.rs` — snapshot input/output.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/session-manager.rs` — read-only snapshot/idempotency tests.
- Extend `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/tests/recording-workflow.rs` — reconciliation workflow.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/tests/recording-scheduler-coordinator.rs` — ownership/restart matrix.

## Implementation Steps

1. Extract authenticated schedule gateway; reject missing/expired sessions and derive audit actor from verified claims.
2. Add bounded schedule queues/correlation separate from recorder pending map.
3. Implement pure coordinator transitions for start, join, leave, stop, feedback, timeout, override.
4. Use scheduled consumer `scheduled:<group_id>:<generation>` for all three resources.
5. On first owner acquire resources deterministically; rollback if command enqueue fails.
6. Route deterministic start through existing `RecordingSessionCommand::Start`.
7. Fan recorder result/status to manual UI and scheduler; keep internal result private.
8. Add explicit recorder snapshot contract/handler with request ID, recording ID, entity, state.
9. Implement reconciliation barrier and cases above; restore only adopted scheduled demand.
10. Implement serialized manual replacement: persist suppression, finalize the scheduled clip, release only scheduled demand, then admit manual start.
11. Route crash recovery attempt metadata to the same scheduler occurrence.
12. Add timeout/overflow/error classification and audit logs.
13. Test duplicate/reordered events and two entities.

## Todo list

- [x] CRUD requires authentication, is rate-limited/CAS-safe, and has no scheduler RBAC branches.
- [x] Actor cannot be spoofed.
- [x] First/last owner controls exactly one session/consumer.
- [x] Browser/manual holds survive scheduled stop/failure.
- [x] Snapshot barrier blocks blind start.
- [x] Matching session adopts random recording ID.
- [x] Unknown/manual session never stopped.
- [x] Queue overflow/timeout visible.
- [x] Manual start finalizes/suppresses scheduled group before starting manual session.
- [x] Crash recovery attaches multiple clip attempts to one occurrence.

## Success Criteria

- Unit tests cover overlap, duplicates, stale generations, partial enqueue rollback, two rovers.
- Restart matrix passes for scheduler, bridge, and recorder before/after acknowledgements.
- Existing manual recording/media-demand tests stay green.
- No scheduled `TargetedMediaControl` bypasses registry transitions.
- Terminal status releases only scheduled consumer.
- Manual replacement never overlaps two recorder sessions for one entity and honors the requested manual directory.

## Risk Assessment

- Dual authority: review searches scheduler for recorder/targeted-control outputs and rejects them.
- Manual/feedback race: serialize per-entity transitions and generation-check feedback.
- Finalization failure during replacement: surface failure, keep manual start pending/failed explicitly, and never start a second concurrent session.
- Crash demand leak: reconcile durable group plus recorder snapshot.
- Main-file growth: focused modules; main only adapts I/O.

## Security Considerations

- Authentication before admission; scheduler revalidates the authenticated actor envelope. Role-based scheduler authorization is deferred.
- Entity-scope snapshots/broadcasts; sanitize errors and audit without secrets.
- Bound queues, pending maps, snapshot size, and request TTL.

## Next steps

- Wire ports/processes in [Phase 04](./phase-04-dora-container-and-operational-integration.md).
- Expose snapshots to [Phase 05](./phase-05-scheduler-ui-and-client-state.md).

## Unresolved questions

1. Broadcast scope: all fleet rovers or selected rover for every logged-in user?
2. Does snapshot need producer-instance ID to reject stale pre-restart status?
