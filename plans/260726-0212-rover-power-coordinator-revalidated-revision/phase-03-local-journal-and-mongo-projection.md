# Phase 03 — Local Journal and Mongo Projection

## Context links

- Parent: [plan.md](./plan.md); design: [event journal and history](../../docs/power-coordinator-architecture.md#event-journal-mongo-projection-and-ui-history).
- Evidence: [voice/history research](./research/researcher-02-voice-ui-history-revision.md).
- Dependencies: Phases 01–02.

## Overview

- Date: 2026-07-26; priority: P1; implementation/review: Blocked by Phases 01–02 remediation.
- Make power intent locally durable, replicate events idempotently, and expose a non-regressing 90-day history/current projection.

## Carryover audit

Commit `a1cbc38` provides CRC-framed local storage, tail recovery, authority
high-water metadata, ack-gated compaction, configured capacity limits, Mongo
event upsert, conditional current projection, TTL, and a clamped cursor query.
It is not accepted completion:

- wake-causing policy/demand/reservation commands are journaled as normal
  records, so they cannot use reserved wake-to-safer capacity;
- command events remove status and omit action, demand/source, and exact-target
  context, leaving history/current projection semantically incomplete;
- projector failure exits the node; reconnect/backoff is not validated;
- demand/source indexes and filters are missing;
- the Mongo integration test silently passes without Mongo, and crash/outage,
  retention, cold-start, and physical disk-pressure gates are incomplete;
- Rover record/ack transport belongs to Phase 04 and must remain an explicit
  release dependency rather than an implied Phase 03 success.

## Key Insights

- Required order is sync intent, apply, live status, replicate, conditionally project. Mongo cannot gate Rover wake or safety.
- Restart keeps authority high-water but discards transient policy/demands and boots `Awake`.

## Requirements

- Journal survives corrupt tail, kill/restart, full disk, and partition without silently dropping unacknowledged transition intent.
- Wake-causing commands and transitions use reserved wake-to-safer capacity;
  deeper sleep remains inhibited when normal capacity is unsafe.
- `power_lifecycle_events` is append-only, unique, and TTL-limited to 90 days; `power_current_state` is one non-TTL row per role/entity.
- Projection advances only for newer epoch/sequence; UI queries enforce retention even while Mongo TTL cleanup lags.
- Durable events retain sanitized command/action, policy/profile,
  demand/reservation source, and exact lifecycle target/revision context when
  applicable.

## Architecture

Each coordinator owns framed journal/outbox plus high-water metadata. Orchestra projector idempotently upserts events and conditionally advances current state; replication acknowledgement permits compaction only after durability.

## Related code files

- Modify `common/power-coordinator/src/{event-journal.rs,event-outbox.rs,outbox-event.rs,journal-record.rs,journal-storage.rs,journal-capacity.rs}` and journal tests.
- Modify `robo_rover_lib/src/types/power_types/` for bounded event context.
- Modify `orchestra/power-event-projector/src/{main.rs,config.rs,mongo-documents.rs,mongo-repository.rs,projector.rs}` and projector tests.
- Modify `docker/docker-compose.yml` and Orchestra/Rover dataflows for retry,
  health, and explicit Phase 04 transport boundaries.

## Implementation Steps

1. Retain checksummed length-delimited records, fsync-before-apply, tail recovery, high-water checkpoint, and safe compaction.
2. Classify wake-causing command intents before admission and preserve reserved capacity under configured saturation and physical disk pressure.
3. Extend the durable event contract with bounded command/action, policy/profile, demand/reservation source, and target/revision context; append a status-bearing applied event after successful command application.
4. Add event, time, demand/source, transition/target, reason, TTL, and current-state unique indexes and matching bounded history filters.
5. Keep idempotent event upsert and conditionally advance projection by epoch/sequence; retry Mongo outages with bounded backoff and acknowledge only after durability.
6. Add crash-point, kill/restart, saturation/ENOSPC, outage/reconnect, reordered batch, TTL/query, cold-start, and interrupted-compaction tests.
7. Make the Mongo-backed test an explicit runnable gate. Phase 04 then wires Rover record/ack transport without changing journal semantics.

## Todo list

- [x] Add baseline journal/outbox, capacity health, Mongo projector, TTL, and monotonic current projection.
- [ ] Make reserved wake capacity reachable by wake-causing commands.
- [ ] Preserve full bounded event context and status-bearing applied events.
- [ ] Add demand/source/target indexes, filters, and bounded projector retry.
- [ ] Validate projector retry/health behavior through Docker/dataflow tests,
  not only crate-level mocks.
- [ ] Complete enforced recovery, disk-pressure, Mongo, retention, and non-regression gates.
- [ ] Hand Rover record/ack transport to Phase 04 as an explicit dependency.

## Success Criteria

- Every applied transition has a preceding synced intent.
- Local and Orchestra offline events replicate once logically; Phase 04 proves
  the same property across the Rover transport.
- Mongo outage leaves safety/local wake functional and history never returns events older than 90 days.
- Saturated normal journal capacity still admits a bounded wake-to-safer
  command, and history retains the context required to explain each change.

## Risk Assessment

- Per-transition fsync affects latency; never batch away transition intent.
- Full journal must surface degraded observability before it risks an unsafe loss.

## Security Considerations

- Journal path is deployment-owned. Events exclude raw audio, tokens, paths, and native error dumps.

## Next steps

Resume this phase only after Phases 01–02 are reaccepted. Phase 04 then
transports snapshots/events and Rover acknowledgements. Derive final capacity
from Rover disk evidence in Phase 08.
