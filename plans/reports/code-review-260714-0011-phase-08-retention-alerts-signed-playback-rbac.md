# Code Review Summary

### Scope
- Files reviewed: `Cargo.lock`, `common/web_bridge/Cargo.toml`, `common/web_bridge/src/main.rs`, `common/web_bridge/src/security.rs`, `common/web_bridge/src/recording_access.rs`, `common/web_bridge/src/recording_playback.rs`, `orchestra/recording_scheduler/src/lib.rs`, `orchestra/recording_scheduler/src/main.rs`, `orchestra/recording_scheduler/src/mongo_store.rs`, `orchestra/recording_scheduler/src/spool.rs`, `orchestra/recording_scheduler/src/alerts.rs`, `orchestra/recording_scheduler/src/playback.rs`, `orchestra/recording_scheduler/src/retention.rs`, `orchestra/recording_scheduler/tests/retention_policy.rs`
- Lines analyzed: ~17.3k LOC in scoped files, concentrated on Phase 08 additions
- Review focus: security, retention correctness, playback path safety, YAGNI/KISS/DRY
- Updated plans: `plans/260713-0310-scheduler-recording-r2-retention/phase-08-retention-alerts-signed-playback-rbac.md`

### Overall Assessment
Phase 08 moved useful pieces into place: conservative role matrix, short-lived playback ticket signing, no-buffer range streaming, idempotent paired-delete helper, and a targeted retention sweep. But the current authorization boundary is wrong, and pressure-deletion / alert-delivery are not wired through the promised state machine. Current state is not ready to mark Phase 08 complete.

Score: `4.5/10`

### Critical Issues
1. Playback/list scope is derived from mutable global fleet state, not user authorization scope, and any authenticated client can mutate that state through `fleet_subscription`. `common/web_bridge/src/main.rs:580`, `common/web_bridge/src/main.rs:970`, `common/web_bridge/src/main.rs:1036`, `common/web_bridge/src/main.rs:1703`
Impact: a viewer can expand `active_rovers_status`, then list clips and mint playback tickets for rovers they should not see. This is a cross-rover privacy break, not just a modeling bug.

### Warnings
1. Storage-pressure eviction still happens directly inside spool admission, bypassing the retention workflow, Mongo transition, and outbox/audit path. `orchestra/recording_scheduler/src/spool.rs:143`, `orchestra/recording_scheduler/src/spool.rs:480`, `orchestra/recording_scheduler/src/main.rs:401`, `orchestra/recording_scheduler/src/retention.rs:102`
Impact: early eviction can remove ready files while metadata still says `finalized_local`; pressure deletions are not tagged with `storage_pressure` through the durable retention path.

2. Durable alerting is only persisted, not delivered/retried. Outbox rows are inserted as `Pending`, but no scoped code reads them, marks `Delivered`/`Failed`, or performs hourly retry/backoff. `orchestra/recording_scheduler/src/mongo_store.rs:211`, `orchestra/recording_scheduler/src/mongo_store.rs:263`, `orchestra/recording_scheduler/src/mongo_store.rs:447`, `orchestra/recording_scheduler/src/mongo_documents.rs:87`
Impact: Phase 08 requirement says durable notifications plus retry; current code gives storage, not delivery.

3. Playback ticket issuance is not audited. `common/web_bridge/src/recording_access.rs:108`
Impact: sensitive clip access lacks the issuance audit trail called for in the phase requirements.

4. Focused web-bridge tests do not exercise the actual HTTP playback handler or filesystem safety path. Current tests cover range parsing and token bind semantics only. `common/web_bridge/src/recording_playback.rs:240`, `common/web_bridge/src/security.rs:996`
Impact: no coverage for `GET`/`HEAD` status/header behavior, expired/forged tickets at handler level, symlink/traversal denial, or cross-rover ticket misuse.

### Suggestions
1. Replace `current_recording_scope()` with scope from immutable auth/session claims or an explicit per-user grant table. Do not couple authorization to `ACTIVE_ROVERS` or `fleet_subscription`. `common/web_bridge/src/main.rs:580`, `common/web_bridge/src/main.rs:1703`

2. Move storage-pressure deletion decisions out of `Spool::reserve_with_input_bytes()` and through the same `request_clip_delete -> local delete -> confirm deleted -> outbox` flow used by age-based retention. `orchestra/recording_scheduler/src/spool.rs:143`, `orchestra/recording_scheduler/src/main.rs:415`

3. Add a small outbox worker or delivery adapter boundary now; otherwise `alerts.rs` stays dead model code and Phase 08 keeps a write-only outbox. `orchestra/recording_scheduler/src/alerts.rs:1`, `orchestra/recording_scheduler/src/mongo_store.rs:447`

4. Audit ticket issuance with actor, clip ID, entity ID, expiry, and request ID. Keep the ticket secret/token out of logs. `common/web_bridge/src/recording_access.rs:108`

5. Add handler-level tests for:
- valid `GET` 200 and `206`
- valid `HEAD` metadata
- invalid and multiple ranges -> `416`
- expired/forged/cross-rover tickets
- missing/size-mismatched media
- symlink and traversal attempts
Refs: `common/web_bridge/src/recording_playback.rs:46`, `common/web_bridge/src/recording_playback.rs:145`

### Positive Observations
- `recording_playback.rs` keeps error responses path-silent and does not buffer whole MP4s.
- `Spool::delete_ready_pair()` is simple and idempotent.
- `retention_candidate_clips()` is the right shape for the hourly age sweep after the follow-up fix.
- RBAC defaults unknown roles to deny.

### Validation
- `cargo test -p recording_scheduler --test retention_policy` ✅
- `cargo test -p web_bridge recording_access` ✅
- `cargo test -p web_bridge recording_playback` ✅
- `cargo test -p web_bridge playback_ticket_binds_clip_entity_and_purpose` ✅
- `cargo check -p recording_scheduler` ✅
- `cargo check -p web_bridge` ✅

### Unresolved Questions
- Where is per-user rover scope supposed to come from long-term: JWT claims, Mongo grants, or fleet roster config?
- Is pressure deletion meant to happen inside admission fast-path, or should admission fail once only protected clips remain until retention clears space?
