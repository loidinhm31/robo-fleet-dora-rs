# Code Review Summary

### Scope
- Files reviewed: `Cargo.lock`, `common/web_bridge/Cargo.toml`, `common/web_bridge/src/main.rs`, `common/web_bridge/src/security.rs`, `common/web_bridge/src/recording_access.rs`, `common/web_bridge/src/recording_playback.rs`, `orchestra/recording_scheduler/src/lib.rs`, `orchestra/recording_scheduler/src/main.rs`, `orchestra/recording_scheduler/src/mongo_store.rs`, `orchestra/recording_scheduler/src/spool.rs`, `orchestra/recording_scheduler/src/alerts.rs`, `orchestra/recording_scheduler/src/playback.rs`, `orchestra/recording_scheduler/src/retention.rs`, `orchestra/recording_scheduler/tests/retention_policy.rs`, `orchestra/recording_scheduler/tests/phase6_media.rs`
- Lines analyzed: ~8.5k LOC in scoped files
- Review focus: Phase 08 re-review after fixes, especially prior critical findings
- Updated plans: none

### Overall Assessment
Most of the first review’s blockers are actually fixed. Recording list/ticket scope no longer depends on mutable `active_rovers_status`, spool admission now declines instead of evicting, pressure deletion runs through `request_clip_delete -> delete_ready_pair -> confirm_clip_deleted`, playback ticket issuance is audited, and playback tests now cover the local file-safety/header cases that were missing before.

Still not ready to call Phase 08 complete. The new alert delivery path broadcasts recording alerts to every authenticated socket with no rover-scope filter, and the outbox schema omits `entity_id`, so the bridge cannot enforce per-rover alert isolation. Retry/reconnect semantics are also still thinner than the phase contract.

Score: `7.0/10`

### Critical Issues
- None.

### High Priority Findings
1. Recording alerts are broadcast to every authenticated socket, not to authorized rover-scoped recipients. `common/web_bridge/src/main.rs:1952`, `common/web_bridge/src/main.rs:1956`, `common/web_bridge/src/stt_socket_delivery.rs:7`, `common/web_bridge/src/main.rs:887`, `common/web_bridge/src/recording_access.rs:62`, `orchestra/recording_scheduler/src/mongo_documents.rs:88`
Impact: any authenticated user in the shared room can receive `recording_alert` events for clips outside their allowed rover scope. Because the outbox payload has `clip_id` but no `entity_id`, the bridge cannot filter these alerts by scope even if it wanted to. This violates the phase requirement for authorized alert/deletion state delivery.

### Warnings
1. Alert delivery is marked globally `delivered` after any one authenticated socket receives the event. Offline but authorized recipients will miss it on later reconnect because delivered rows are no longer replayable. `common/web_bridge/src/main.rs:1955`, `common/web_bridge/src/main.rs:1959`, `common/web_bridge/src/recording_access.rs:182`, `common/web_bridge/src/recording_access.rs:201`
Impact: durable outbox semantics are still incomplete; this is effectively first-consumer-wins, not durable authorized delivery.

2. There is still no failure/backoff state machine for the web-bridge outbox worker. Rows are queried as `pending`/`failed`, but this path never marks `failed`, records retry attempts, or applies backoff. `common/web_bridge/src/recording_access.rs:182`, `common/web_bridge/src/main.rs:1937`
Impact: delivery retries are opportunistic polling, not the independent retry/backoff flow promised in the phase plan.

3. Playback tests improved, but they still stop short of full handler-level auth coverage for forged/expired ticket rejection and cross-rover misuse. Current tests exercise file serving and range parsing via `serve_authorized_clip`, not the full ticket-validation path in `serve_recording_playback`. `common/web_bridge/src/recording_playback.rs:64`, `common/web_bridge/src/recording_playback.rs:311`
Impact: the sensitive auth boundary for `/recordings/:clip_id/playback` still lacks direct regression coverage.

### Low Priority Suggestions
1. The new recording access paths continue expanding `common/web_bridge/src/main.rs`, now ~4.1k LOC. Pulling alert delivery and recording route wiring behind smaller modules would reduce review and regression surface. `common/web_bridge/src/main.rs`

### Positive Observations
- The prior cross-rover access bug from mutable `active_rovers_status` is closed. `common/web_bridge/src/main.rs:620`, `common/web_bridge/src/main.rs:1012`, `common/web_bridge/src/main.rs:1079`
- Spool admission no longer performs direct pressure eviction; it now declines cleanly when capped. `orchestra/recording_scheduler/src/spool.rs:137`, `orchestra/recording_scheduler/tests/phase6_media.rs:161`
- Pressure deletion now uses the retention path instead of ad hoc file deletion. `orchestra/recording_scheduler/src/main.rs:453`, `orchestra/recording_scheduler/src/main.rs:467`
- Playback ticket issuance now writes scheduler audit rows. `common/web_bridge/src/recording_access.rs:156`, `common/web_bridge/src/recording_access.rs:252`
- Playback safety coverage is much better than the first pass: GET/HEAD/range headers, traversal rejection, size mismatch, symlink denial. `common/web_bridge/src/recording_playback.rs:311`

### Recommended Actions
1. Add `entity_id` to recording outbox documents and deliver alerts only to sockets whose role+scope authorize that rover. Do not reuse the global authenticated room for recording alerts.
2. Change outbox delivery tracking from single global `Delivered` to either per-recipient ack or replayable authorized snapshots on reconnect, depending on product intent.
3. Implement explicit `failed`/retry bookkeeping with backoff if Phase 08 is meant to satisfy durable delivery, not just best-effort polling.
4. Add handler-level playback tests that hit `serve_recording_playback` with valid, forged, expired, and cross-rover tickets.

### Metrics
- Type Coverage: not measured
- Test Coverage: not measured
- Linting Issues: `git diff --check` clean for scoped files
- Fresh validation:
  - `cargo test -p recording_scheduler --test retention_policy` ✅
  - `cargo test -p recording_scheduler --test phase6_media reservation_declines_when_ready_bytes_would_exceed_cap` ✅
  - `cargo test -p web_bridge recording_access` ✅
  - `cargo test -p web_bridge recording_playback` ✅
  - `cargo test -p web_bridge playback_ticket_binds_clip_entity_and_purpose` ✅
  - `cargo test -p web_bridge recording_rbac_defaults_unknown_roles_to_deny` ✅
  - `cargo check -p recording_scheduler` ✅
  - `cargo check -p web_bridge` ✅

### Unresolved Questions
- Should alert durability be per-user/per-role, or is a scoped replay-on-connect enough for this phase?
- Are recording alerts intended for viewers too, or only operator/admin roles?
