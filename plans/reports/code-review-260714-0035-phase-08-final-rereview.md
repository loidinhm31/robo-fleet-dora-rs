## Code Review Summary

### Scope
- Files reviewed: `Cargo.lock`, `common/web_bridge/Cargo.toml`, `common/web_bridge/src/main.rs`, `common/web_bridge/src/security.rs`, `common/web_bridge/src/recording_access.rs`, `common/web_bridge/src/recording_playback.rs`, `orchestra/recording_scheduler/src/lib.rs`, `orchestra/recording_scheduler/src/main.rs`, `orchestra/recording_scheduler/src/mongo_documents.rs`, `orchestra/recording_scheduler/src/mongo_store.rs`, `orchestra/recording_scheduler/src/spool.rs`, `orchestra/recording_scheduler/src/alerts.rs`, `orchestra/recording_scheduler/src/playback.rs`, `orchestra/recording_scheduler/src/retention.rs`, `orchestra/recording_scheduler/tests/retention_policy.rs`, `orchestra/recording_scheduler/tests/phase6_media.rs`
- Lines of code analyzed: ~8.9k LOC
- Review focus: Phase 08 final scoped re-review after second-review fixes
- Updated plans: none

### Overall Assessment
Most prior blockers are fixed. Alert payloads now carry `entity_id`, web delivery is role+rover scoped, viewers are excluded from recording alerts, pressure admission declines instead of evicting, and playback validation/tests are materially better.

Still not Phase-08 complete. Retention retry semantics are still broken in two places, and scoped clip listing paginates incorrectly because filtering happens after the DB query.

Score: `8.1/10`

### Critical Issues
- None.

### High Priority Findings
1. `delete_requested` is persisted before the outbox write, but a later outbox failure leaves the clip stranded in `delete_requested` with no retry path before day 15. `orchestra/recording_scheduler/src/mongo_store.rs:211`, `orchestra/recording_scheduler/src/mongo_store.rs:231`, `orchestra/recording_scheduler/src/main.rs:467`
Impact: a transient Mongo failure after the state flip can suppress both deletion and the day-14 notification. Later sweeps do not re-enter `request_clip_delete` because the clip is no longer `finalized_local`.

2. Retention never retries local deletion for clips already in `delete_requested`; day-15 handling only raises the critical outbox event. `orchestra/recording_scheduler/src/main.rs:410`, `orchestra/recording_scheduler/src/main.rs:415`, `orchestra/recording_scheduler/src/main.rs:475`
Impact: this misses the phase requirement for hourly retry/backoff after failed deletion. A clip that survives the first unlink attempt can remain on disk indefinitely while the scheduler only logs/alerts.

### Warnings
1. Scoped clip listing is filtered after `limit`/cursor are applied in Mongo. `common/web_bridge/src/recording_access.rs:121`, `common/web_bridge/src/main.rs:1036`
Impact: viewers/operators can receive empty or truncated pages when out-of-scope clips occupy the newest rows. This is not a data leak, but it breaks the bounded private list contract and makes pagination unstable for multi-rover scopes.

### Low Priority Suggestions
1. `common/web_bridge/src/main.rs` is now 4.3k LOC. Recording access, playback route wiring, and alert delivery are distinct enough to split out. `common/web_bridge/src/main.rs`

### Positive Observations
- Alert authorization is now enforced by immutable role+scope at both live delivery and reconnect snapshot. `common/web_bridge/src/main.rs:915`, `common/web_bridge/src/main.rs:1976`, `common/web_bridge/src/main.rs:1996`
- Outbox documents now carry `entity_id`, which closes the earlier alert-scope leak. `orchestra/recording_scheduler/src/mongo_documents.rs:80`
- Pressure handling no longer deletes ready media directly at reservation time. `orchestra/recording_scheduler/src/spool.rs:106`, `orchestra/recording_scheduler/tests/phase6_media.rs:160`
- Playback ticket coverage now includes wrong-secret, expired, and cross-clip rejection. `common/web_bridge/src/recording_playback.rs:325`

### Recommended Actions
1. Make day-14 intent persistence atomic enough for retry: either write outbox + state in one transaction, or retry the outbox write while the clip is still eligible for deletion.
2. Rework retention sweeps so `delete_requested` clips continue through `delete_ready_pair` retries until the pair is confirmed absent, with the critical alert as an additional state, not the terminal action.
3. Push scope constraints into `RecordingAccessStore::list_clips` so Mongo applies rover filtering before sort/limit/cursor.

### Metrics
- Type Coverage: not measured
- Test Coverage: not measured
- Linting Issues: `git diff --check` clean for scoped files
- Fresh validation:
  - `cargo check -p web_bridge` ✅
  - `cargo check -p recording_scheduler` ✅
  - `cargo test -p web_bridge recording_alert_scope_tests` ✅
  - `cargo test -p web_bridge playback_ticket_binds_clip_entity_and_purpose` ✅
  - `cargo test -p web_bridge recording_playback -- --nocapture` ✅
  - `cargo test -p web_bridge recording_access` ✅
  - `cargo test -p web_bridge recording_rbac_defaults_unknown_roles_to_deny` ✅
  - `cargo test -p recording_scheduler --test retention_policy` ✅
  - `cargo test -p recording_scheduler --test phase6_media reservation_declines_when_ready_bytes_would_exceed_cap` ✅
  - `cargo test -p recording_scheduler --test local_storage_quota` ✅

### Unresolved Questions
- Is a Mongo transaction acceptable for `request_clip_delete`, or does this phase still need to avoid multi-document transactions?
- For multi-rover operators/admins, should the list API accept an explicit rover set, or should server-side scope expansion stay implicit?
