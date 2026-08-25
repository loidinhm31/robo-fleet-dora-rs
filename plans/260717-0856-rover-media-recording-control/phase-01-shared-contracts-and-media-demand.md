# Phase 01: Shared Contracts and Target-Aware Media Demand

## Context links

- [Parent plan](./plan.md)
- [Backend research](./research/researcher-01-media-backend-report.md)
- [UI research](./research/researcher-02-recording-ui-report.md)
- [Architecture contract](../../ARCHITECTURE.md#manual-fleet-media-recording-and-playback)
- Depends on: existing fleet, video, audio, auth, and Dora contracts. Blocks Phases 02, 03, and 05.

## Overview

- Date: 2026-07-17
- Description: Freeze cross-process recording contracts and add per-rover aggregation for camera, JPEG, and microphone demand.
- Priority: P1 correctness prerequisite
- Status: Done (2026-07-17 11:47 +07)
- Progress: 100%
- Implementation status: Complete
- Review status: Complete
- Effort: 7h

## Key Insights

- JPEG publication and microphone capture are demand-gated; recording cannot depend on an open viewer.
- Existing browser demand is globally counted and routed through mutable `selected_entity`; that cannot support concurrent rover recordings.
- UI drafts already claim `recording_command`/`recording_status` for scheduler resource state. File sessions need separate names.
- Current access boundary is a valid signed-in session. This feature must not add role checks or RBAC.

## Requirements

- Shared, versioned JSON types for session start/stop, status, clip query/result, playback lookup, and structured errors.
- Every command/result carries `request_id`; server assigns UUID `recording_id`; every status/clip carries `entity_id`.
- Record, stop, list, ticket, and playback require a current authenticated session; all signed-in users use the same permissions.
- Socket/Dora file-session names: `recording_session_command`, `recording_session_command_result`, `recording_session_status`.
- Retain non-conflicting `recording_clip_list(_result)` and `recording_playback_ticket(_result)` after fixture reconciliation.
- Aggregate independent consumers by `(entity_id, consumer_id, resource)` for camera capture, JPEG publication, and microphone.
- Emit targeted effective-state changes only on per-resource `0 -> 1` and `1 -> 0` transitions.

## Architecture

- `web-bridge` owns an in-memory `MediaDemandRegistry`; consumer IDs include `browser:<socket>` and `recording:<recording_id>`.
- A browser demand pins the selected entity at acquisition. Fleet selection migrates only that browser's demand.
- Recording admission acquires demand before forwarding start. Rejection, startup timeout, stop acknowledgement, failure, and shutdown release it idempotently.
- `TargetedMediaControl` carries authoritative `entity_id` plus desired camera/JPEG/mic state. `orchestra-bridge` decomposes it into existing rover commands/topics.
- Recorder status is a map keyed by rover/recording ID, not one global selected-rover state.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/recording_types.rs` — versioned commands, statuses, clips, queries, reason codes.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/mod.rs` — exports.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/recording_types_tests.rs` — validation and golden JSON.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/media-demand-registry.rs` — pure per-entity aggregation.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs` — replace global browser stream count at integration points.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs` — target-aware media-control routing.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml` — targeted control edge.
- Later mirror types in `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording.ts`.

## Implementation Steps

1. Inventory untracked UI recording tests; document ownership and never delete/overwrite them implicitly.
2. Define bounded enums/fields and validation: protocol version, UUIDs, entity IDs, relative directory, timestamps, byte counts, codecs, and reason codes.
3. Add Rust golden fixtures that TypeScript will consume or reproduce byte-for-byte.
4. Implement the pure demand registry with idempotent acquire/release and effective transition output.
5. Refactor browser camera/JPEG/mic handlers to register a pinned entity demand; preserve current reconnect/disconnect cleanup.
6. Define target-aware bridge envelope and route only to active, exact-match rover IDs. Keep legacy selected-rover controls during coordinated migration.
7. Add lifecycle tests for simultaneous entities, duplicate events, fleet selection changes, client disconnect, recorder rejection, and shutdown.

## Todo list

- [x] Freeze recording and targeted-media contracts.
- [x] Add cross-language fixtures.
- [x] Implement/test per-entity demand aggregation.
- [x] Route target-aware controls in Orchestra bridge.
- [x] Prove one consumer cannot stop another consumer's resources.

## Success Criteria

- Two rover recording demands produce independent start/stop transitions.
- Viewer plus recorder on one rover emits one start; either may stop without an upstream stop until both release.
- No recording route falls back to `selected_entity`.
- Malformed IDs, paths, versions, states, and unauthenticated requests fail before Dora/Zenoh publication.
- Existing live-view demand tests continue passing.

## Risk Assessment

- Risk: refactor regresses live streaming. Mitigation: preserve legacy behavior behind pure transition tests before integration.
- Risk: leaked demand after failed start. Mitigation: idempotent release on every terminal path plus shutdown sweep.
- Risk: event collision with scheduler drafts. Mitigation: reserve `recording_session_*` for files and test event maps.

## Security Considerations

- `entity_id` is server-validated against active fleet; browser selection is not routing authority.
- Relative directories are validated again by the recorder; shared type validation is not the filesystem boundary.
- New commands require a live signed-in session and existing command rate limits; no claim-role comparison is allowed.

## Next steps

- Phase 02 consumes frozen frame/control/status contracts.
- Phase 03 wires auth, query, and playback only after contract tests pass.

## Unresolved questions

- None.
