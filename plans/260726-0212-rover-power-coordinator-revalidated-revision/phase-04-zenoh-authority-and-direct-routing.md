# Phase 04 — Zenoh Authority and Direct-Mode Routing

## Context links

- Parent: [plan.md](./plan.md); design: [ownership](../../docs/power-coordinator-architecture.md#ownership).
- Evidence: [current cutoff audit](../reports/reconciliation-260726-0613-rover-power-plan-cutover.md);
  [power-stack scout](./scout/scout-01-power-stack.md) and
  [voice/UI research](./research/researcher-02-voice-ui-history-revision.md)
  are historical pre-baseline inputs and must be rechecked against `HEAD`.
- Dependencies: Phases 01–03.

## Overview

- Date: 2026-07-26; priority: P1; implementation/review: Pending.
- Add dedicated power transport and snapshot-first reconciliation without overloading movement/media or lifecycle-lease topics.

## Key Insights

- Current bridges carry lifecycle status, commands, leases, and queries only. Dedicated power V1 topics and direct-mode ports do not exist.
- Rover-local KWS may wake while partitioned; Orchestra must observe fresh state before a strictly newer authority command.

## Requirements

- Use `rover/{entity_id}/power/v1/{command,status,snapshot,event}`; bound payloads/queues and validate entity, expiry, epoch, sequence, and duplicate semantics at both bridge edges.
- Wire Phase 01's coordinator snapshot gate to production transport. On
  reconnect request snapshot, mark `AuthorityUnknown` until fresh authoritative
  state arrives, persist observed high-water, then issue next epoch if needed.
- Clear pre-sleep actuator/media replay cache at quiesce epoch; require fresh command after wake.
- Direct mode omits bridges only; contracts, journal, authority, and Socket.IO semantics stay identical.

## Architecture

Split: Orchestra coordinator → Orchestra bridge → Zenoh → Rover bridge → Rover coordinator, with reverse status/snapshot/event flow. Direct: local web bridge/scheduler connect to Rover coordinator through identical Dora contracts.

## Related code files

- Modify `orchestra/zenoh_bridge/src/main.rs` and `rover-kiwi/zenoh_bridge/src/main.rs`.
- Modify `orchestra/orchestra-dataflow.yml`,
  `rover-kiwi/rover-kiwi-dataflow.yml`, and
  `rover-kiwi/rover-kiwi-direct-dataflow.yml`.
- Modify `common/power-coordinator/src/{state_machine.rs,main.rs,lib.rs}` and
  shared power snapshot tests. Introduce focused `authority.rs` or
  `reconciliation.rs` modules only if the implementation needs that separation.

## Implementation Steps

1. Add dedicated topic helpers, Dora ports, bounded serialization, and per-entity routing tests.
2. Implement snapshot request/reply/reconcile reducer; do not expose a timeout force-takeover operation.
3. Persist `max(local, observed)+1` only after snapshot; retain safety veto and stale/reordered counters.
4. Fence pre-sleep command caches and wire equivalent direct dataflows.
5. Test partition, bridge restart, snapshot loss, duplicate/reordered commands, inactive rover, and queue saturation.

## Todo list

- [ ] Add power topics and ports.
- [ ] Add `AuthorityUnknown` reconnect gate.
- [ ] Wire split/direct parity and replay fence tests.

## Success Criteria

- No profile command leaves Orchestra before fresh Rover snapshot.
- Partitioned Rover KWS wake remains safe; reconnect never regresses effective state.
- Split and direct status schemas are identical.

## Risk Assessment

- Incomparable epochs and cached commands can create split brain; snapshot-first reconciliation plus no-replay is mandatory.

## Security Considerations

- Browsers cannot publish Zenoh. Bridges accept only local coordinator routes and exact active entities.

## Next steps

Phase 05 makes scheduler a reservation producer; Phase 06 adds local KWS demand.
