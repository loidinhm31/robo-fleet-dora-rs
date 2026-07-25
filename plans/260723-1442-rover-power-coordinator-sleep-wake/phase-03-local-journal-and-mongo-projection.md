# Phase 03 — Local Journal and Mongo Projection

## Context links

- Parent: [plan.md](./plan.md)
- Design: [Event Journal, Mongo Projection, and UI History](../../docs/power-coordinator-architecture.md#event-journal-mongo-projection-and-ui-history)
- Input: [authority/journal research](./research/researcher-01-power-authority-journal.md)
- Dependencies: Phases 01–02 event/state contracts.

## Overview

- Date: 2026-07-26 (completed)
- Description: make transition intent locally durable, replicate history idempotently, and maintain non-regressing current projection.
- Priority: P1
- Implementation status: Complete — 100%
- Review status: Complete

## Completion update

Completed 2026-07-26T05:43:45+07:00. Local journal durability/recovery, authority high-water persistence, bounded capacity handling, idempotent Mongo projection, conditional current-state advancement, 90-day history clamping, and outage/recovery ordering validation are complete.

## Key Insights

- MongoDB cannot sit on Rover wake/safety path.
- Journal replay republishes unacknowledged events and authority high-water marks; it does not restore transient policy/demands.
- Live status is current-state authority. Mongo projection is cold/stale fallback only.

## Requirements

- Order: sync local intent → apply → live status → idempotent replication → conditional projection advance.
- Persist authority epoch high-water across restart; restart still chooses fresh `Awake`.
- Bounded append journal survives torn final record, process kill, disk full, and long partition without silent loss.
- `power_lifecycle_events` retains append-only history for 90 days; `power_current_state` has no TTL.
- Reordered/duplicate replication cannot regress projection; TTL asynchrony never expands query retention.
- Event detail excludes audio, secrets, absolute paths, tokens, and unbounded native errors.

## Architecture

Each coordinator owns a local framed journal/outbox. Orchestra `power-event-projector` consumes local and Rover events, upserts stable event IDs, conditionally advances `(authority_epoch, sequence)`, and acknowledges replication. Journal compaction removes only acknowledged records; high-water metadata remains.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/power-coordinator/src/{event-journal.rs,journal-record.rs,event-outbox.rs}`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/power-coordinator/tests/{journal-recovery.rs,event-ordering.rs}`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/power-event-projector/Cargo.toml`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/power-event-projector/src/{main.rs,config.rs,mongo-documents.rs,mongo-repository.rs,projector.rs}`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/power-event-projector/tests/mongo-integration.rs`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml` — projector workspace member.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.yml` — journal volumes/config.

## Implementation Steps

1. Implement versioned length-delimited journal records with checksum, monotonic sequence, stable event ID, fsync-before-apply, tail-truncation recovery, and atomic compacted checkpoint.
2. Persist authority high-water separately from runtime policy; on boot set epoch above recovered/observed epoch, clear demands, journal fresh Awake intent.
3. Add byte/record caps and high-water behavior: never drop unacknowledged records; disable Auto sleep and reject new non-safety Sleep when capacity is unsafe; preserve reserved wake-to-safer capacity.
4. Create Mongo collections/indexes:
   - unique events `{deployment_id:1, entity_id:1, event_id:1}`;
   - history `{deployment_id:1, entity_id:1, occurred_at:-1}`;
   - transition/demand/kind/reason plus time indexes;
   - TTL `{expires_at:1}`, `expireAfterSeconds:0`, where app sets `expires_at = occurred_at + 90d`;
   - unique current `{deployment_id:1, entity_id:1, role:1}`, no TTL.
5. Upsert events idempotently; advance current state only when epoch is newer or same epoch sequence is greater.
6. Make history queries clamp `from/to` to 90 days despite asynchronous TTL cleanup; paginate with stable `(occurred_at,event_id)` cursor.
7. Test duplicate/reordered batches, crash between journal/apply/status/ack, corrupt tail, Mongo outage/reconnect, compaction interruption, and projection cold start.

## Todo list

- [ ] Implement local journal/outbox and capacity status.
- [ ] Add authority high-water persistence/fresh Awake boot.
- [ ] Add projector collections and exact indexes.
- [ ] Add 90-day filtered cursor history query.
- [ ] Add recovery, ordering, TTL, outage tests.

## Success Criteria

- Every applied transition has a prior synced intent under crash injection.
- Offline events replicate once logically by stable ID after reconnect.
- Old event cannot regress `power_current_state`; TTL never touches current state.
- Query returns no event older than 90 days even before TTL cleanup.
- Journal corruption/full state is visible; local safety and wake-to-safer behavior remain available.

## Risk Assessment

- Per-event fsync can add latency: batch non-transition telemetry, never transition intents.
- Full journal during long outage: expose degraded observability and inhibit deeper sleep.
- Mongo index creation on populated deployment can block: create empty collections before rollout and monitor.

## Security Considerations

- Journal directory owner/mode is deployment-controlled; no browser-selected path.
- Store stable actor/audit ID only, never session token or raw wake audio.
- Sanitize reason detail before both local append and Mongo replication.

## Next steps

Phase 04 transports status/events and implements status-first authority takeover. Freeze journal byte cap and reserved wake capacity from Rover disk budget during Phase 08.
