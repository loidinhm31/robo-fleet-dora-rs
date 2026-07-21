# Phase 02 — Lifecycle Control Plane and Routing

## Context links

- [Parent plan](./plan.md)
- [Lifecycle research](./research/researcher-01-lifecycle-control.md)
- [Architecture](../../ARCHITECTURE.md#planned-resource-monitoring-and-soft-stop-lifecycle)
- Existing routing: `common/web_bridge/src/main.rs`, both `zenoh_bridge/src/main.rs`

## Overview

- Date: 2026-07-21
- Priority: P1
- Implementation status: Done
- Review status: Approved
- Description: create authoritative, idempotent pause/resume routing and state reconciliation.

## Key Insights

- Existing `performance_control` is a global broadcast gate, not resource control.
- Publish success is not applied state. Resume requires an always-responsive control spine.
- Explicit entity target must be pinned when admitted; fleet selection cannot retarget it.

## Requirements

- Socket command/result/status with auth, command rate limit, request correlation, and target validation.
- Versioned `LifecycleCommand`, `LifecycleCommandResult`, `LifecycleStatus`, component capability/state.
- Command fields: request ID, manager epoch, explicit deployment role/entity/node, desired state, expected revision, issued/expiry time.
- Idempotent duplicate handling, revision CAS, TTL rejection, periodic status/query reconciliation.
- Any valid signed-in session may submit a lifecycle command; authorization does not depend on an operator/admin role.
- Preserve user desired state separately from temporary scheduled-recording wake leases and effective state.
- Normal Orchestra mode and direct Rover mode share the same browser contract.
- Never pause control-plane/safety nodes.

## Architecture

Add a small `common/lifecycle_manager` Dora node configured for Orchestra or Rover. Orchestra manager owns desired state and routes local commands or exact-rover commands through Orchestra bridge. Rover bridge routes `rover/{entity}/cmd/lifecycle/v1` to Rover manager. Applied status returns on `rover/{entity}/lifecycle/status/v1`. Web bridge caches current capabilities/status and replays them to authenticated reconnects.

Scheduled recording uses temporary, uniquely identified wake leases for an exact rover/resource set. Leases affect effective state without overwriting user desired state. Overlapping occurrences share/refcount the effective wake; releasing the final lease reconciles against the current desired revision, so a user resume made during recording is not undone.

State: `Running → Cancelling/Quiescing → Quiesced → Resuming → Running`, with `Degraded`, `Failed`, and `Superseded`. Admission `accepted/rejected` is separate from progress and terminal applied status. `Quiesced` requires all accepted work terminalized and all declared resources released; `Degraded` is truthful partial teardown, never successful pause.

## Related code files

- Create: `robo_rover_lib/src/types/lifecycle_types.rs` and validation/tests.
- Create: `common/lifecycle_manager/` with state, capability registry, routing, reconciliation.
- Modify: workspace Cargo, three dataflow YAMLs, both Zenoh bridges, `common/web_bridge/src/main.rs`.
- Modify: UI shared Socket/lifecycle types and test fixtures.
- Remove: old `performance_control` handler and process-global boolean.

## Implementation Steps

1. Freeze wire enums/reason codes and validation bounds; add JSON cross-language fixtures.
2. Implement manager epoch, user desired revision, lease-derived effective state, applied state, request cache, revision CAS, TTL, and deadlines. Reject reused request IDs with changed payload.
3. Add server-owned capability manifest; unsupported/always-on nodes cannot be targeted.
4. Add authenticated `node_lifecycle_command` for any valid session; snapshot deployment role/entity/node before Dora enqueue.
5. Wire local Orchestra routing, remote Zenoh routing/status, and direct Rover routing.
6. Cache/replay authoritative capabilities and statuses; add explicit status query on cache miss.
7. Add audit logs for actor/request/target/admission/applied result without sensitive data.
8. Add idempotent internal scheduled wake-lease acquire/release, overlap ownership, bounded expiry/restart reconciliation, and reconciliation to the latest user desired revision.

## Todo list

- [x] Lifecycle contracts and fixtures
- [x] Common manager
- [x] Socket admission/result/status
- [x] Dora/Zenoh normal routing
- [x] Direct-mode routing
- [x] Replay/reconciliation tests

## Validation

- Final review: 9/10, approved.
- Gates passed: lifecycle contract/manager tests, workspace formatting and compilation, authenticated Socket admission, exact-target Orchestra↔Zenoh and direct-Rover routing, periodic reconciliation, and stale/expired/CAS/idempotency checks.

## Success Criteria

- Same request ID never applies twice; stale/expired commands never apply.
- Pre-restart commands/statuses from an old manager epoch never apply to the new authority generation.
- Changing selected rover after submit cannot change the target.
- Concurrent user commands with a stale expected revision return conflict plus current status instead of silently overwriting state.
- Releasing the final schedule wake lease restores the current desired state, not a stale pre-wake snapshot.
- UI can distinguish accepted, applied, degraded, failed, stale, and unsupported.
- Disconnect/reconnect converges status without replaying workload data or actuator commands.

## Risk Assessment

- Lost status: periodic query/reconciliation and freshness timers.
- Resume deadlock: managers and bridges always on; transitions have deadlines.
- Split brain: one Orchestra manager owns revision; Rover reports applied revision only.

## Security Considerations

- Require a valid signed-in session, existing command limiter, active entity allowlist, exact server-advertised node, and revision CAS.
- Bound request cache, TTL, node IDs, and failure detail. Derive actor from the session; never trust browser actor/entity metadata.

## Next steps

- Phase 03 gives supported nodes safe lifecycle adapters.

## Unresolved Questions

- Define retry/backoff and terminal status when a scheduled wake cannot reach Running before its occurrence deadline.
