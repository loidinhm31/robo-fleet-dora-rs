# Phase 03: Backend Control, Catalog, and Playback

## Context links

- [Parent plan](./plan.md)
- [Phase 01 contracts](./phase-01-shared-contracts-and-media-demand.md)
- [Phase 02 recorder](./phase-02-media-recorder-ffmpeg-and-storage.md)
- [Architecture contract](../../ARCHITECTURE.md#manual-fleet-media-recording-and-playback)
- Depends on: Phases 01-02. Blocks Phases 05-06.

## Overview

- Date: 2026-07-17
- Description: Wire the recorder into Dora/Zenoh and expose signed-in-user session, catalog, and range-playback APIs.
- Priority: P1 integration/security
- Implementation status: Done (2026-07-17)
- Review status: Approved after critical findings fixed; medium/low follow-ups accepted (2026-07-17)
- Effort: 10h

## Key Insights

- `common/web_bridge` is the real web server; stale prose naming `orchestra/web_bridge` must not guide file placement.
- Socket.IO is appropriate for commands/status/catalog metadata, not MP4 bytes.
- HTML video cannot attach the current bearer token reliably; short-lived opaque playback tickets allow normal range requests.
- The recorder owns clip truth. `web-bridge` may cache correlated lookup results but must not invent its own catalog.

## Requirements

- Fan bridge `video_frame`/`audio_frame` to recorder with explicit bounded queues and no backpressure into web/STT.
- Signed-in-session, rate-limited Socket.IO handlers for session command, clip list, and playback-ticket request; no role checks.
- Broadcast authoritative per-rover session statuses to authenticated clients; replay current snapshots after auth/reconnect.
- Query/lookup requests flow through Dora to recorder and return by `request_id` with timeout/error handling.
- Playback tickets are random, single-clip, short-lived, revocable on expiry, and never logged.
- HTTP endpoint supports `GET`/`HEAD`, one byte range, `206`, `Content-Range`, `Accept-Ranges`, MIME/length, and streaming without whole-file buffering.

## Architecture

- UI `recording_session_command` -> active-session validation -> demand acquire/release + Dora recorder command.
- Recorder status -> demand lifecycle reconciliation -> authenticated Socket.IO `recording_session_status`.
- UI `recording_clip_list` -> Dora catalog query -> `recording_clip_list_result` with opaque ID/relative display path only.
- UI `recording_playback_ticket` -> recorder clip lookup -> in-memory ticket map -> `recording_playback_ticket_result` URL/expiry.
- HTTP ticket route revalidates expiry, root containment, finalized state, manifest/file identity, and requested range before streaming.

## Related code files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml` — recorder node, media/control/query/status edges and queue sizes.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs` — exact target media-control routing from Phase 01.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/recording-socket.rs` — handlers, admission, correlation.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/recording-access.rs` — signed-in session guard and ticket registry.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/recording-playback.rs` — safe HTTP range responses.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs` — focused module wiring, Dora ports, status delivery.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/Cargo.toml` — only required HTTP/range dependencies.
- Add focused tests under `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/` and `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/tests/`.

## Implementation Steps

1. Add recorder/dataflow ports and explicit queue budgets: video drops oldest, audio remains bounded, control/query/status remain small and correlated.
2. Add target-aware Orchestra bridge routing and active-rover validation; prove commands for rover A cannot reach rover B.
3. Implement modular Socket handlers. Validate the live signed-in session, rate limit, request IDs, entity activity, action/state, and relative path before enqueue; do not compare roles.
4. Coordinate demand registration with recorder status: keep recordings alive across initiating socket disconnect; release only on rejected/terminal lifecycle or explicit signed-in-user stop.
5. Implement query correlation tables with bounded size/deadlines and sanitized failures. Replay status snapshots after authenticated reconnect.
6. Implement short-lived ticket issuance only after a fresh finalized-clip lookup; store server-side relative path and file identity, not absolute browser-visible paths.
7. Add streaming range route with overflow-safe parsing and `416` behavior; reject multi-range and expired/unknown tickets.
8. Add audit logs for actor, action, entity, clip ID, request ID, outcome, and reason while excluding ticket/path secrets.
9. Test auth, routing, correlation, replay, path attacks, range semantics, ticket expiry, and large-file streaming behavior.

## Todo list

- [x] Wire recorder node and bounded Dora edges.
- [x] Add target-aware Zenoh bridge controls.
- [x] Add signed-in-session recording Socket.IO modules.
- [x] Add catalog/lookup correlation.
- [x] Add ticket registry and range endpoint.
- [x] Add backend integration/security tests.

## Success Criteria

- Recording works for two named active rovers regardless of current UI fleet selection.
- Unauthenticated, expired, malformed, and rate-limited requests mutate nothing and reveal no host paths.
- Reconnect receives authoritative active statuses and a fresh clip list on request.
- Finalized MP4 seeks through `206` byte ranges; partial/missing/corrupt clips and expired tickets are denied.
- Slow playback clients do not buffer whole clips or block Dora processing.

## Risk Assessment

- Risk: status races leak demand. Mitigation: server-assigned recording ID, idempotent transitions, bounded correlation expiry, reconciliation tests.
- Risk: web/recorder path disagreement. Mitigation: recorder lookup is authoritative; web revalidates shared root and file identity.
- Risk: main.rs grows further. Mitigation: keep handlers/access/playback in focused modules.

## Security Considerations

- Accept any current signed-in session. Reject missing/expired sessions; do not add role parsing, comparisons, or RBAC.
- Tickets are unguessable, least-lived, clip-scoped capabilities. Never expose the recording root or manifest path.
- Protect against traversal, symlink swap, range integer overflow, response splitting, token leakage, and unbounded request maps.

## Next steps

- Phase 04 packages runtime/storage dependencies.
- Phase 05 consumes the frozen Socket/HTTP contract.

## Verification notes

- Targeted workspace verification: 161 tests passed plus 1 shared-library doc test; YAML wiring and Phase 03 Rust formatting checks passed.
- Playback authorization uses component-wise Unix `openat` traversal with `O_NOFOLLOW`, short-lived in-memory tickets, and MP4/manifest identity checks.
- Pending command, catalog, and playback requests emit correlated typed/admitted failures on timeout or local queue rejection.
- The independent code-reviewer’s remaining medium/low findings are intentionally deferred: fair queue scheduling, terminal-status cache retention, shared playback error typing, richer audit fields, and browse activity refresh.

## Unresolved questions

- None.
