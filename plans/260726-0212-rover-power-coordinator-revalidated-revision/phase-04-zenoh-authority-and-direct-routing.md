# Phase 04 — Zenoh Authority and Direct-Mode Routing

## Context links

- Parent: [plan.md](./plan.md); design: [ownership](../../docs/power-coordinator-architecture.md#ownership).
- Evidence: [current cutoff audit](../reports/reconciliation-260726-0613-rover-power-plan-cutover.md);
  [power-stack scout](./scout/scout-01-power-stack.md) and
  [voice/UI research](./research/researcher-02-voice-ui-history-revision.md)
  are historical pre-baseline inputs and must be rechecked against `HEAD`.
- Dependencies: Phases 01–03.

## Overview

- Date: 2026-07-26; priority: P1; implementation: Complete; review: Passed, accepted 2026-07-27.
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

1. Complete — dedicated topics/ports, per-entity routing, signed power-command envelope, and bounded bridge ingress.
2. Complete — snapshot request/reply/reconcile path; no timeout force-takeover operation.
3. Complete — observed authority epoch high-water persists before later restart; stale/reordered snapshots remain gated.
4. Complete — quiesce fences replay cache; direct flow includes the same durable projector acknowledgement path.
5. Complete — exact authority reconciliation, signed snapshot/ACK transport, and control-plane isolation are covered by focused tests. All three Dora graphs validate.

## Todo list

- [x] Add power topics and ports.
- [x] Add `AuthorityUnknown` reconnect gate and snapshot-request trigger.
- [x] Wire split/direct durable journal acknowledgement parity and replay fence tests.
- [x] Authenticate Rover authority snapshots before Orchestra accepts them for reconciliation.
- [x] Authenticate journal acknowledgements before Rover compacts its durable event journal.
- [x] Isolate bounded power-control/event ingress from high-rate video/audio traffic.
- [x] Reconcile and document an exact-successor authority rule that preserves current command semantics.

## Current status / review blockers (2026-07-27)

Implemented changes remain uncommitted. An ordinary command must carry the
Rover's current authority and advances it on apply; reconnect is the sole
epoch transition and must be exactly `{snapshot.epoch + 1, sequence: 1}`.
This keeps the existing command/result semantics while rejecting all gaps and
reordering. Orchestra accepts only signed, fresh Rover snapshots before it
forwards their raw payload to the coordinator or stores observed epoch state.
Remote ACKs are similarly signed over target, role, version, deployment ID,
event ID, and lifetime before the Rover bridge releases raw ACK data locally.

Rover control and media Dora ingress now use independent bounded queues,
control is selected first, and media is sent through a separate bounded lossy
publisher task so a slow Zenoh media write cannot stall control. Status/snapshot
updates coalesce to their newest value. Pending remote journal records remain
deduplicated by event ID.

Validation passed: 36 `power_coordinator` tests, 78 `robo_rover_lib` tests,
24 Orchestra bridge tests, 10 Rover bridge tests, projector offline checks,
`cargo check` for all affected packages, `cargo fmt --check`, `git diff --check`,
and all three Dora graphs. Mongo duplicate/reorder integration remains
intentionally skipped without `POWER_PROJECTOR_TEST_MONGODB_URI`.

Fresh code review passed with no actionable findings. The final review also
verified that Rover video/audio/playback metrics count only accepted bounded
media handoffs and record rejected handoffs as drops.

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

## Unresolved questions

- Run the opt-in Mongo duplicate/reorder integration when
  `POWER_PROJECTOR_TEST_MONGODB_URI` is available.
