# Phase 04 — Authority, Zenoh, and Direct Routing

## Context links

- Parent: [plan.md](./plan.md)
- Design: [Ownership and Component Flow](../../docs/power-coordinator-architecture.md#ownership)
- Input: [authority research](./research/researcher-01-power-authority-journal.md)
- Dependencies: Phases 01–03 contracts, coordinator, journal.

## Overview

- Date: 2026-07-24
- Description: route power control independently of actuator topics and reconcile Rover-local wake with Orchestra authority.
- Priority: P1
- Implementation status: Pending
- Review status: Pending

## Key Insights

- Rover must wake locally during a Zenoh partition.
- Reconnect is status-first: Orchestra observes Rover epoch/state before issuing a strictly newer authority.
- Direct mode uses identical contracts without either Zenoh bridge.

## Requirements

- Dedicated topics: `rover/{entity_id}/power/v1/command`, `/status`, `/snapshot`, `/event`; never reuse `cmd/movement|arm|media`.
- Exact entity routing, bounded queues, expiry, epoch/sequence fencing, duplicate idempotency, and stale/reorder counters.
- Rover local KWS may create bounded local epoch/demand while disconnected; Orchestra wins only after snapshot reconciliation.
- No pre-sleep actuator/media message replay; fresh command required after wake.
- Direct web bridge routes to local Rover coordinator with same Socket.IO contract/status semantics.

## Architecture

Split mode: Orchestra coordinator → Orchestra bridge → Zenoh → Rover bridge → Rover coordinator. Return status/snapshot/event follows reverse path. Reconnect handshake is `snapshot request -> Rover snapshot -> Orchestra persist/compare -> next epoch command -> Rover applied status`. Direct mode connects web bridge/scheduler ports locally.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/power-coordinator/src/authority.rs` and `reconciliation.rs`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs` — dedicated targeted power topics and event relay.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/zenoh_bridge/src/main.rs` — local coordinator routing/status snapshot.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/rover-kiwi-dataflow.yml`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/rover-kiwi-direct-dataflow.yml`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/lifecycle_manager/src/lib.rs` — topology/queue invariant tests.

## Implementation Steps

1. Add authority reducer: compare epoch/sequence, exact role/entity, issued/expiry; exact duplicates ack, changed duplicates reject.
2. Add dedicated Dora ports for command/status/snapshot request/snapshot/event/replication ack.
3. Add dedicated Zenoh topics with bounded payload validation and per-entity subscriptions; never use mutable selected-rover fallback.
4. Implement link-state handshake. Orchestra blocks new remote power commands until fresh Rover snapshot or takeover timeout.
5. After snapshot, Orchestra persists `max(local,observed)+1`, reconciles effective profile, then opens normal admission; safety state always wins.
6. Clear cached pre-sleep actuator/media commands at quiesce epoch and reject any older command after wake.
7. Wire split and direct dataflows. Direct mode omits bridges but preserves target, epoch, status, journal, and event paths.
8. Test duplicate/reordered status, bridge restart, delayed old command, simultaneous local wake/reconnect, inactive rover, and queue saturation.

## Todo list

- [ ] Implement authority comparison and handshake.
- [ ] Add dedicated Dora/Zenoh routes.
- [ ] Wire Orchestra, Rover, and direct dataflows.
- [ ] Fence old actuator/media replay.
- [ ] Add partition/reconnect/topology tests.

## Success Criteria

- Rover local state remains safe/operational without Orchestra.
- Orchestra never sends takeover authority before consuming fresh Rover status, and its command epoch is strictly newer.
- Delayed pre-partition power or actuator command cannot apply after wake/reconnect.
- Split and direct modes emit schema-identical live status.
- Wrong/inactive entity and oversized/expired payloads fail closed.

## Risk Assessment

- Split brain from incomparable epochs: use observed maximum + durable increment, not wall clock.
- Snapshot loss can stall takeover: bounded retries, visible reconciling state, no unsafe guess.
- Bridge fanout pressure: separate bounded control/event queues; status cannot be starved by history replication.

## Security Considerations

- Bridge accepts coordinator messages only from local Dora routes; browser cannot publish Zenoh.
- Exact target and source capability checked at both bridge boundaries.
- Audit stale/rejected authority without logging secrets or raw payloads.

## Next steps

Phase 05 adds scheduler producer semantics. Authority takeover timeout remains a Phase 08 fault-derived deployment value.

