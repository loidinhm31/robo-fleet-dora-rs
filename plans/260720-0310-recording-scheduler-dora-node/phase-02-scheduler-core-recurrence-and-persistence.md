# Phase 02 — Scheduler Core, Recurrence, and Persistence

## Context links

- [Parent plan](./plan.md)
- [Phase 01](./phase-01-contracts-and-decision-freeze.md)
- [Backend research](./research/researcher-01-backend-scheduler-report.md)
- Historical references only: commits `92c2fb9`, `caa47c7`; no wholesale cherry-pick.

## Overview

- Date: 2026-07-20
- Description: Build durable scheduler node, recurrence engine, occurrence/group state, Mongo repository, and fake-clock tests.
- Priority: P1
- Implementation status: Done — Phase 02 hardening complete
- Review status: Approved after hardening; follow-up warnings are tracked below
- Effort: 14h

## Key Insights

- Recurring rules preserve local wall-clock intent; storing only UTC loses DST behavior.
- Single-writer transitions plus unique indexes/outbox avoid requiring Mongo transactions on standalone Compose.
- Persist intent before emitting Dora work; duplicate feedback must be harmless.
- Mongo outage fails closed for new tracked work; active sessions reconcile after storage returns.

## Requirements

- New `recording_scheduler` Dora-RS crate; no FFmpeg, rover, or Zenoh dependency.
- Inject `Clock`: wall time for recurrence, monotonic time only for sleeps/backoff.
- Materialize bounded horizon covering at least the next weekly occurrence; extend continuously.
- DST: fold earlier, gap first valid instant, elapsed duration from resolved start; persist resolution.
- Coalesce active occurrences per entity into durable group owner set.
- Earliest planned start then occurrence ID selects directory; group never moves directory.
- Retry transient feedback after 1, 2, 4, 8, and 16 seconds, then every 30 seconds until planned end; no attempt cap. Alerting is future scope.
- Set `expire_at = terminal_at + 90 days` only when an occurrence/audit record becomes terminal.
- Preserve an ordered list of failed, partial, and recovered clip attempts on one logical occurrence.
- Ignore duplicate, stale-generation, illegal, or regressive feedback.
- Startup indexes/loads/materializes/rebuilds, pauses emission, requests reconciliation, then resumes.

## Architecture

- Collections: `recording_schedules`, `recording_occurrences`, `recording_scheduler_groups`, optional `recording_scheduler_meta`.
- Indexes: unique schedule ID; `(entity_id, enabled, next_occurrence_ms)`; unique occurrence and `(schedule_id, revision, planned_start_ms)`; `(entity_id, state, planned_end_ms)`; unique active request/group; terminal-only 90-day `expire_at` TTL.
- Occurrence documents keep bounded clip-attempt entries with recording ID, state, timestamps, reason, and recovery sequence.
- Loop: reload due work -> persist transition/outbox -> emit -> apply feedback CAS -> schedule next wake.
- Startup: indexes -> load -> materialize -> rebuild groups -> request snapshot -> reconcile -> ready.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/Cargo.toml` — focused Dora/Mongo/timezone dependencies.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/main.rs` — Dora adapter only.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/config.rs` — env limits.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/clock.rs` — real/fake clock.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/recurrence.rs` — calendar/DST.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/domain.rs` — domain records.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/state-machine.rs` — legal transitions.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/mongo-documents.rs` — BSON conversion.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/mongo-repository.rs` — indexes/CAS.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/runtime.rs` — evaluation/retry/reconcile.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/src/ports.rs` — Dora serialization.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/tests/recurrence.rs` — timezone/property cases.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/tests/mongo-recovery.rs` — restart/CAS.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml` and `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.lock` — workspace member/lock.

## Implementation Steps

1. Implement bounded config and fake/real clock.
2. Implement one-time/daily/weekly candidate generation with `chrono_tz::Tz`.
3. Prove stable UUIDv5 occurrence/group/request IDs across restart.
4. Implement legal transitions, transient/permanent feedback classification, and exact 1–30 second retry schedule bounded by window end.
5. Implement signed BSON `i64` milliseconds; reject invalid values at recorder boundary.
6. Create idempotent indexes; remain not-ready on incompatible duplicates.
7. Implement CRUD CAS; deletion cancels future work but preserves terminal history for 90 days.
8. Implement unique occurrence upsert and bounded horizon.
9. Implement group owner-set transitions and persisted pending intent.
10. Persist multiple clip attempts under one occurrence and implement startup barrier/periodic reconciliation.
11. Add keyed logs/counters and wall-clock jump/restart tests; do not add retry alert delivery.

## Todo list

- [x] Fake clock has no real sleeps.
- [x] DST gap/fold/month/year/leap tested.
- [x] Duplicate materialization and stale CAS tested.
- [x] Group `0 -> 1`, intermediate, `1 -> 0` durable.
- [x] No action before reconciliation.
- [x] Terminal-only 90-day TTL verified; nonterminal rows have no `expire_at`.
- [x] Exact retry cadence continues without attempt cap until window end.
- [x] Failed/partial/recovered clips stay associated with one occurrence.
- [x] Empty/no-enabled schedule is responsive.

## Success Criteria

- `cargo test -p recording_scheduler` deterministic.
- Property tests show ordered, unique occurrences.
- Kill/restart after intent persistence replays same request ID logically once.
- Mongo unavailable rejects mutations and creates no untracked authoritative state.
- Transient failure tests retry at 1/2/4/8/16/30-second intervals and stop exactly at window end.
- Crash recovery retains every clip attempt without creating another logical occurrence.
- Two rover groups interleave without leakage.
- Modules stay focused; extract near 200 LOC.

## Risk Assessment

- Partial multi-document write: group outbox recovery rebuilds from nonterminal occurrences.
- Clock jump: recompute from wall clock and unique materialization.
- TTL misuse: assign the fixed 90-day `expire_at` only after terminal persistence.
- Unbounded attempt history: cap attempt metadata by recorder/window limits while retaining every actual attempt allowed in the window.
- Historical lease contamination: reuse tests/algorithms only.

## Security Considerations

- Revalidate authenticated actor and all command fields; do not add scheduler RBAC.
- Do not log Mongo URI, credentials, JWT, absolute storage paths, or raw browser payload.
- Bound list queries, schedule count, title/template, and horizon.
- Fail readiness on schema/index conflicts; never silently delete duplicates.

## Next steps

- Feed stable intents/snapshots into [Phase 03](./phase-03-web-bridge-coordinator-and-recorder-reconciliation.md).
- Expose config/process in [Phase 04](./phase-04-dora-container-and-operational-integration.md).

## Finalization record

- **2026-07-20 17:00 +07 (UTC+0700):** Phase 02 hardening is complete and review-approved. Targeted fault-boundary, bridge-overlap, durable-outbox replay, and update/delete-race tests passed, alongside `cargo test -p recording_scheduler`, live standalone Mongo verification at `mongodb://127.0.0.1:27017`, and Clippy.
- **Review follow-up warnings:** Phase 03 must validate reconciliation ordering against recorder snapshots and preserve the scheduler's deterministic group-directory selection. Phase 04/operations must define production Mongo persistence topology, credential ownership, backup, and restore procedures.

## Unresolved questions

1. Is standalone Mongo guaranteed, confirming no transaction requirement?
2. Exact materialization horizon and maximum schedules per entity?
