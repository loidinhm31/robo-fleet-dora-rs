# Backend research: Orchestra recording scheduler

Research time: 2026-07-20 03:10 Asia/Saigon  
Scope: Rust contracts, scheduler correctness, Mongo, web-bridge coordinator, recorder, Dora wiring. Codebase only; no external web.

## Executive recommendation

Build `orchestra/recording_scheduler` as single writer for schedules/occurrences and recurrence evaluation. Keep `common/web_bridge` as sole media-demand + recording-session coordinator. Scheduler emits durable desired occurrence ownership; bridge converts aggregate per-rover owner changes into current `TargetedMediaControl` and `RecordingSessionCommand`. Never let scheduler send rover controls or recorder commands directly.

Model overlap as a per-rover recording group: owner set `occurrence_id -> planned_end_ms`. `0 -> 1` acquires camera/JPEG/microphone and starts one recorder session; intermediate owner changes do nothing; `1 -> 0` stops that exact `recording_id`, then releases scheduled demand. Persist group/occurrence intent before Dora emission; replay until acknowledged.

Current protocol is close but cannot prove crash-safe idempotency alone. Correlate starts with deterministic request/occurrence IDs and `RecordingSessionStatus.request_id`; add an explicit recorder active-status snapshot/reconcile request or guaranteed startup snapshot. Do not rely on web-bridge's in-memory maps after restart.

## Codebase evidence

- `robo_rover_lib/src/types/recording_types.rs`: protocol v1; UUID request/recording IDs; start has entity + relative directory, stop has recording ID; status contains original request ID, entity, state, timestamps/bytes/reason. No origin, occurrence, owner, revision, or status query.
- `common/web_bridge/src/media-demand-registry.rs`: reference-counted `(entity, consumer, resource)` set. Emits ON only on first demand and OFF only after last release. Duplicate acquire/release is idempotent; consumer rename supports request-ID to recording-ID handoff. This is the safe aggregation boundary.
- `common/web_bridge/src/recording-socket.rs`: bounded in-memory command queues, 15s pending TTL, status cache, and `recording_id -> entity_id` active map. All disappear on process restart.
- `common/web_bridge/src/main.rs:780`: authenticated manual recording handler. Start pre-acquires media demand under `recording:<request_id>`; rejects inactive/actively recording rover; command result/status later renames/releases demand.
- `common/web_bridge/src/main.rs:1940`: Dora output queues drain from the web process. Scheduler traffic should use equivalent bounded queues, typed validation, TTL, and origin-aware routing—not browser socket ownership.
- `common/web_bridge/src/main.rs:3435`: recorder result/status processing is the coordinator insertion point. Scheduled result/status must route to scheduler and UI while manual results remain socket-routed.
- `common/web_bridge/src/main.rs:4200`: recording-demand helpers acquire/release Camera, JPEG, Microphone through `MediaDemandRegistry` and enqueue aggregate transitions.
- `orchestra/media_recorder/src/session-manager.rs:186`: only one active session per entity; recording ID is random UUID; active state is memory-only. Stop is idempotent for IDs in the process-local `finished` set, not across restart.
- `orchestra/media_recorder/src/main.rs:220`: validates existing command, starts/stops manager, emits result/status. Reuse it; no second FFmpeg/session engine.
- `orchestra/media_recorder/tests/recording-workflow.rs`: established workflow/error/backpressure/status test harness; extend rather than duplicate.
- `orchestra/orchestra-dataflow.yml:147`: recorder consumes bridge commands + Orchestra media fan-out and returns result/status to bridge.
- `orchestra/orchestra-dataflow.yml:182`: web bridge currently owns all recorder commands and Mongo auth configuration.
- `Cargo.toml:3`: add scheduler workspace member; shared UUID currently enables only `v4` and needs `v5` (or another stable ID derivation) for deterministic occurrence IDs.
- `common/web_bridge/src/security.rs`: Mongo user/auth helpers exist, but scheduler persistence belongs in its own focused repository module and collections.

## Recommended contracts

Add shared Rust modules under `robo_rover_lib/src/types/` (split validation from types if >200 LOC):

- `RecordingSchedule`: `schedule_id`, `revision`, `entity_id`, title, enabled, recurrence, local start/date, IANA timezone, duration_ms, relative-directory template, created/updated actor + epoch ms.
- Recurrence: tagged `one_time | daily | weekly`; weekly has non-empty ISO weekday set. Persist local wall-clock intent for recurring rules, epoch milliseconds for every materialized occurrence.
- `RecordingOccurrence`: deterministic `occurrence_id`, schedule ID/revision, entity, planned start/end epoch ms, state, retry count/next retry, group/request/recording IDs, override disposition, last error, timestamps.
- CRUD command/result/snapshot family: protocol version, request ID, actor context, expected revision. Update/delete require CAS; conflicts return current authoritative object.
- `ScheduledRecordingIntent`: occurrence up/down, entity, planned bounds, directory, generation. Bridge feedback includes accepted/applied/retryable, group ID, recording ID, recorder status.
- Keep manual `recording_session_*` wire compatible. Add optional internal origin/owner envelope in bridge, not user-controlled Socket.IO fields.

Use deterministic UUIDv5 over `schedule_id + schedule_revision + planned_start_ms`. Retry the same occurrence/request ID. Validate all IDs, lengths, durations, timezone names, future bounds, directory template expansion, and entity IDs at scheduler boundary.

## Scheduler state machine and correctness

Occurrence states: `planned -> due -> start_pending -> active -> stop_pending -> completed`; terminal alternatives `suppressed`, `missed`, `failed`, `cancelled`. Persist transition + durable intent before emission. Results may repeat/out-of-order; accept only legal monotonic transitions for matching occurrence generation.

Evaluation loop uses injected `Clock`; wake at earliest transition/retry plus bounded periodic reconciliation. On startup: load enabled schedules + nonterminal occurrences, materialize horizon, rebuild active desired owners, request bridge/recorder snapshot, then adopt matching `request_id`/entity sessions before starting. Do not issue a blind start during reconciliation grace.

Retry transient inactive-rover, bridge unavailable, recorder startup/resource errors with bounded exponential backoff capped by remaining window. Never retry after planned end. Invalid schedule/directory is permanent. Recorder crash may create a partial clip; restart may start a new physical clip for the remaining window, but must retain one logical occurrence and expose partial/recovered status.

Overlap grouping is per entity, not schedule. Stable group ID derives from entity + first uncovered union interval. Directory choice must be deterministic (recommended: earliest occurrence, then occurrence ID tie-break); record every owner-to-clip association.

Manual override belongs in bridge coordinator. Recommended manual stop: stop current scheduled group, mark all currently active owners `suppressed`, release only their scheduled consumers, leave schedules enabled. Recompute/start at the next schedule boundary, not on every retry tick. Manual start gets manual ownership and must not be stopped by a scheduled OFF. Exact collision semantics need product confirmation.

## Recurrence and DST policy

- Use a maintained Rust IANA timezone library (`chrono` + `chrono-tz` is simplest with current stack); never fixed offsets for recurring rules.
- Materialize each local calendar occurrence exactly once. Recommended gap policy: shift to first valid instant after gap. Fold policy: choose earlier instant. Persist resolution (`exact | gap_shifted | fold_earlier`) for audit.
- Compute `planned_end_ms = resolved_start_ms + duration_ms`; duration is elapsed time, so a DST transition does not silently change clip length.
- One-time schedules persist authoritative start epoch ms plus timezone for display. Weekly weekdays are evaluated in schedule timezone.
- Store scheduler instants as signed BSON/JSON-safe `i64` epoch ms internally; validate non-negative before converting to existing recorder `u64` timestamps.

## Mongo design

Scheduler alone writes `recording_schedules`, `recording_occurrences`, and optionally `recording_groups/outbox`. Web bridge authenticates/authorizes and forwards actor claims; scheduler revalidates command shape and allowed role.

Indexes: unique `schedule_id`; `(entity_id, enabled, next_occurrence_ms)`; unique `occurrence_id`; unique `(schedule_id, schedule_revision, planned_start_ms)`; `(entity_id, state, planned_end_ms)`; unique group/request identity. TTL only terminal history via explicit `expire_at`; never TTL schedules/nonterminal rows.

Prefer CAS `revision` and single-writer serialized transitions over Mongo transactions (transactions require replica-set deployment). If atomic multi-document correctness becomes necessary, use a durable outbox/group document and recovery invariants before requiring transactions. Create indexes at startup and fail readiness on incompatible duplicate data.

## Dora wiring

Add `recording-scheduler` node to `orchestra/orchestra-dataflow.yml` with bounded ports:

- web -> scheduler: `recording_schedule_command`, `recording_scheduler_recorder_feedback`, reconciliation snapshot.
- scheduler -> web: `recording_schedule_command_result`, `recording_schedule_status`, `scheduled_recording_intent`.
- web -> existing recorder and Zenoh paths remain unchanged.

Give scheduler `MONGODB_URI`, `MONGODB_DATABASE`, retry/reconcile/horizon settings, and `RUST_LOG`. Keep queue sizes explicit. Add scheduler to `Cargo.toml`; add Mongo/timezone dependencies at crate scope unless genuinely shared.

## Alternatives/tradeoffs

- Direct scheduler -> recorder/rover: superficially fewer bridge changes, but creates dual media authorities and unsafe OFF races. Reject.
- Timers inside web bridge: less Dora plumbing, but couples auth/socket lifecycle to durable scheduling and worsens testing/restart recovery. Reject.
- Persist only schedules, derive occurrences: simpler schema but loses acknowledgement, retry, suppression, clip ownership, and audit. Reject.
- Mongo transactions: strongest cross-document atomicity, but deployment burden. Defer; single writer + unique indexes + outbox is sufficient v1.

Historical commits `92c2fb9`, `0855a01`, `caa47c7`, `992afac` are design references for Mongo/timezone/revision/lease behavior. Reuse isolated validation/index/fake-clock ideas only. Their rover device-lease and recording contracts differ from HEAD; do not cherry-pick or restore direct lease authority.

## Test and phase gates

1. Contract: JSON round trips, unknown protocol, UUID/length/duration/directory/timezone validation, stale revision conflict.
2. Recurrence: one-time/daily/weekly; month/year rollover; DST gap/fold policies; duration across DST; property tests ensure ordered, unique occurrences.
3. Repository/state: unique indexes, CAS, duplicate materialization, durable intent replay, illegal/out-of-order feedback ignored.
4. Coordinator: demand `0->1/1->0`, duplicate intent, two overlapping owners, unrelated browser/manual demand survives, deterministic directory, manual suppression/resume.
5. Restart matrix: scheduler only, bridge only, recorder only, each crash before/after start/stop ack; adopt matching active session; no blind duplicate; expired window becomes missed/completed.
6. Recorder integration: one active entity, repeated stop, startup failure/retry, partial clip recovery, non-empty MP4 using existing workflow harness.
7. Dora/Mongo integration: real bounded ports, Mongo restart, unavailable rover then recovery, status reconnect snapshot, two rovers independently.

## Key risks

- Current recorder has no durable idempotency/status query; explicit reconciliation is a release gate.
- In-memory bridge demand/active maps vanish on restart; scheduler replay + recorder snapshot must restore them.
- Manual override wording is underspecified for overlapping windows and simultaneous manual start/stop.
- Mongo unavailable policy must avoid running untracked recording actions; default fail closed for new transitions, preserve/reconcile already-active sessions.
- Wall-clock jumps: always re-read clock and recompute due work; use monotonic timers only for sleeping/backoff.

## Unresolved questions

1. Which roles may create/edit/delete/disable schedules versus view status?
2. Confirm DST policies: gap shift-forward and fold-earlier, or skip/choose-later?
3. Manual stop during overlap: suppress all active occurrences until next boundary (recommended), or only one selected schedule?
4. Manual start during scheduled recording: adopt existing clip, replace it, or reject as already recording?
5. Retention period for terminal occurrences/audit and whether Mongo is guaranteed replica-set capable.
6. When recovered after recorder crash, may one logical occurrence reference multiple partial clips?
