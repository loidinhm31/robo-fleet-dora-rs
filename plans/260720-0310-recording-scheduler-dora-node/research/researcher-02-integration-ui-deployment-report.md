# Research Report: Scheduler integration, UI, deployment, verification

Research date: 2026-07-20 (Asia/Saigon)  
Scope: current HEAD plus commits `92c2fb9`, `0855a01`, `caa47c7`, `992afac`; repository evidence only. No product implementation.

## Executive summary

Recommended boundary: scheduler owns durable schedule/occurrence state; authenticated Socket.IO enters through `common/web_bridge`; web bridge alone mutates `MediaDemandRegistry` and forwards existing recorder commands. Do not restore historical rover device leases or let scheduler emit camera/audio OFF directly.

UI should add a third top-level Scheduler view beside CONTROL/RECORDINGS, with its own store/event hook and authoritative refresh-on-connect. Existing scheduler override drafts encode obsolete lease fields and are reference-only. Deployment must add scheduler to the Orchestra image/dataflow and hard-coded container health probe. Mongo is available in Compose, but scheduler needs explicit startup/index/migration/retention policy.

## Current HEAD evidence

### Recording and Socket.IO boundary

- `common/web_bridge/src/recording-socket.rs`: bounded in-memory command/query queues, 15s pending-request TTL, cached recorder statuses, active recording-to-entity mapping. It is transport state, not durable schedule state.
- `common/web_bridge/src/media-demand-registry.rs`: demand key is `(entity_id, consumer_id, resource)`; OFF emits only after final consumer release. Existing tests cover duplicate events, two entities, consumer isolation, selection migration, and shutdown.
- `common/web_bridge/src/main.rs`: authenticated connection installs handlers for `recording_session_command`; rejects invalid/rate-limited/over-capacity requests; publishes `recording_session_command` to Dora and routes status/results to sockets.
- `orchestra/orchestra-dataflow.yml`: web bridge already owns `targeted_media_control`; recorder command/status/result ports exist. Scheduler ports should join these two nodes, not the rover bridge directly.
- `robo_rover_lib/src/types/recording_types.rs` and related modules: current session/clip protocol is source of truth. Add a separate schedule family; avoid overloading manual session commands.

### UI boundary

- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx`: state supports only `control | recordings`; nav and content switch are centralized here.
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/media-recording-page.tsx`: existing recording page consumes selected entity, authentication, fleet state, and recording store. Mirror this injection style for scheduler page.
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts`: typed `ServerToClientEvents`/`ClientToServerEvents`; add schedule CRUD/query/status/conflict events here before UI work.
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording.ts`: current recorder plus draft scheduler-device types. Schedule definitions/occurrences should be new focused types or a separate `schedule.ts` export.
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/use-scheduler-device-overrides.ts` and `packages/ui/src/lib/scheduler-media-commands.ts`: draft expects authority epochs, revisions, leases, and `recording_command`; incompatible with the recommended schedule/session contract. Do not build CRUD on it.
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/src/recording-e2e-harness.tsx` and `apps/web/e2e/recording-control.spec.ts`: deterministic browser harness already exists; extend with scheduler fixtures/events rather than requiring live Dora for all UI tests.

### Mongo and deployment

- `common/web_bridge/src/main.rs` + `common/web_bridge/src/security.rs`: web bridge connects using `MONGODB_URI`/`MONGODB_DATABASE`; existing DB dependency is auth/session oriented. Scheduler node needs its own client, readiness/retry behavior, and indexes.
- `docker/docker-compose.yml`: `mongo:8.0`, persistent `mongodb-dev-data`, loopback-only port, ping healthcheck. Orchestra env already carries Mongo config. Base container health command hard-codes required binaries and must include `/app/bin/recording_scheduler`.
- `docker/docker-compose.workstation.yml`: Orchestra waits for healthy Mongo and overrides DB settings. Preserve this gate; test scheduler restart without deleting Mongo volume.
- `Makefile`: `up-mongodb`, `up-workstation`, compose validation, logs, and status paths already exist. Extend help/env docs and validation only where scheduler adds configuration.
- Orchestra image build definition and workspace `Cargo.toml`: add scheduler binary/member and ensure image copies it; dataflow start alone is insufficient if binary is absent.

## Historical evidence: reuse selectively

- `92c2fb9`: useful Mongo document/model and schedule contract naming; old device-demand/lease model conflicts with HEAD.
- `caa47c7`: useful fake clock, recurrence, occurrence, Mongo store/index, arbiter, and restart test ideas.
- `992afac`: useful Dora queue-policy and web-boundary test shapes; its recording boundary routed the retired lease architecture.
- `0855a01`: rover lease lifecycle is explicitly out of the recommended v1 boundary.
- Rule: port individual invariants/tests after comparing current types; never cherry-pick these commits wholesale.

## Recommended implementation phases

### Phase 1 — Contract, ACL, and UI state model

- Define Rust + TypeScript schedule commands/results/snapshots: request ID, schedule ID, occurrence ID, entity ID, revision, actor, server timestamp.
- Separate `recording_schedule_create/update/delete/list/status` from `recording_session_*`.
- Define role matrix in web bridge: suggested viewer=list/status; operator=create/update/enable/disable; admin=delete. Derive actor from JWT claims, never accept UI `requested_by`.
- Gate: serialization fixtures match Rust/TS; unauthorized and stale-revision responses are deterministic; no recorder behavior changes.

### Phase 2 — Scheduler persistence, indexes, retention

- Node owns `recording_schedules`, `recording_occurrences`, and optional `recording_scheduler_meta` migration marker.
- Idempotent startup indexes: unique schedule ID; unique `(schedule_id, occurrence_id)`; query indexes `(entity_id, enabled, next_run_ms)` and `(state, planned_end_ms)`; CAS filter includes revision.
- Store UTC Unix-ms instants plus IANA zone/local rule. Keep active/failed/audit rows durable; apply TTL only to terminal occurrences through explicit `expires_at` Date. Mongo TTL cleanup is approximate, never correctness logic.
- Define DB unavailable behavior: node not ready, bounded exponential reconnect, no in-memory authoritative writes, resume/reconcile after reconnect.
- Gate: temporary Mongo integration test proves indexes, CAS conflict, restart recovery, duplicate occurrence rejection, terminal-only retention.

### Phase 3 — Web-bridge coordinator and ownership

- Add scheduler Dora queues/status cache separate from browser pending queues. Socket connect/query returns authoritative list + occurrence snapshot; broadcasts are entity-scoped.
- Coordinator maps coalesced active windows to one stable consumer/session per entity. Reference-count occurrence owners; last release alone stops. Use deterministic recording ID/idempotency key.
- Manual command override suppresses current scheduled occurrence only; persist suppression/result in scheduler. Never use mutable selected rover for scheduled ownership.
- On web-bridge restart, request scheduler snapshot + recorder statuses before emitting start/stop. Do not trust prior in-memory demand state.
- Gate: unit/integration tests cover browser/manual demand survival, overlap, two rovers, duplicate events, queue saturation, timeout, reconnect, and out-of-order recorder result.

### Phase 4 — Dora and container deployment

- Wire scheduler inputs/outputs between `recording-scheduler`, `web-bridge`, and recorder in `orchestra/orchestra-dataflow.yml`; use bounded queues for commands/status and document overflow policy.
- Add crate to root `Cargo.toml`, binary build/copy to Orchestra Docker build, Mongo/timezone/retry env, and scheduler process to Compose healthcheck.
- Startup order: Mongo healthy -> Orchestra container/dataflow -> scheduler loads/indexes/reconciles -> schedule API ready. Surface degraded scheduler without hiding recorder/web health.
- Gate: `cargo build/test`, dataflow validation, `make validate-compose`, `make validate-workstation-compose`, cold start and scheduler-only kill/restart with Mongo volume retained.

### Phase 5 — Scheduler UI

- Add `scheduler` view in `RoboRoverControl.tsx`; focused page/components for list, editor, occurrence/status, conflict/error banner. Avoid enlarging the already-large page beyond routing/prop wiring.
- Editor: one-time/daily/weekly, local date/time, weekdays, duration, IANA timezone, safe relative output directory, enabled flag. Display resolved next-run instant and DST adjustment before save.
- Dedicated store/hooks: request correlation, revision CAS, reconnect resync, entity switch reset, optimistic pending indicator but authoritative commit. Disable writes when unauthenticated/insufficient role/offline.
- Gate: component/store tests cover validation, role visibility, conflict refresh, entity isolation, DST preview, reconnect and terminal/retry/missed states; type-check/lint/build linked app.

### Phase 6 — End-to-end and operations release gate

- Extend deterministic harness for CRUD/status/reconnect/mobile/accessibility. Playwright: create/edit/disable/delete, stale edit, role denial, entity switch, active/retry/missed transitions.
- Live Compose test: one-time start produces non-empty playable MP4; scheduled end releases only its demand; overlapping windows share session; manual/browser demand survives; two rovers isolated.
- Fault tests: rover offline then returns, recorder rejected/full disk, Mongo restart, scheduler crash after start-before-ack, web-bridge restart, duplicate/reordered messages.
- Observability: structured logs keyed by `request_id`, `schedule_id`, `occurrence_id`, `recording_id`, `entity_id`; counters for due/start/stop/retry/missed/conflict/failure; gauges for enabled schedules, active occurrences, queue depth; reconciliation duration/outcome. Never log JWTs or Mongo credentials.
- Release gate: no duplicate MP4/session after crash; no unrelated OFF; bounded retry/no busy loop; status/UI converges after every restart; health probe detects missing scheduler process.

## Primary risks

- Split authority: scheduler directly controls rover or recorder. Mitigation: mandatory coordinator/demand registry path.
- Restart race: in-memory web state contradicts durable occurrence. Mitigation: deterministic IDs + three-way scheduler/recorder/demand reconciliation.
- TTL deletes evidence/current work. Mitigation: terminal-only `expires_at`; retain schedules and active occurrences.
- Cross-rover/status leakage. Mitigation: entity in every key/filter/event plus interleaving tests.
- Role spoofing. Mitigation: actor/role only from verified JWT; per-operation ACL and audit.
- Historical contract contamination. Mitigation: fresh schedule protocol; borrow algorithms/tests only.

## Unresolved questions

1. Exact viewer/operator/admin permissions, especially delete and output-directory edit?
2. Occurrence retention duration; must audit failures be retained longer than successful runs?
3. Mongo production topology/credentials/backups beyond local Compose single instance?
4. Manual stop suppresses current occurrence only (recommended) or requires a timed resume option?
5. Retry limits/backoff and final missed-vs-failed classification?
6. Should scheduler degradation fail whole Orchestra container health or expose a separate degraded readiness signal?
