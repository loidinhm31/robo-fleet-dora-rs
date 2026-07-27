# Phase 05: Recording Control and Playback UI

## Context links

- [Parent plan](./plan.md)
- [Phase 01 contracts](./phase-01-shared-contracts-and-media-demand.md)
- [Phase 03 backend](./phase-03-backend-control-catalog-and-playback.md)
- [UI research](./research/researcher-02-recording-ui-report.md)
- Depends on: frozen Phase 01/03 wire contracts. Blocks Phase 06 UX acceptance.

## Overview

- Date: 2026-07-17
- Description: Add a shared web/Tauri page for path selection, concurrent rover sessions, clip browsing, and ticketed playback.
- Priority: P2 signed-in workflow
- Implementation status: Done (2026-07-18)
- Review status: Done (2026-07-18, 10/10, no findings)
- Effort: 7h

## Key Insights

- UI is a separate Git checkout; backend and frontend cannot be one atomic commit.
- Both app shells render the shared `RoboRoverControl`; no router exists. Use a small internal view switch and keep one socket/auth owner mounted.
- `RoboRoverControl.tsx` is already large. Compose a focused page/hook rather than adding feature logic inline.
- Untracked scheduler/recording tests are user work and reveal event-name collisions. Reconcile deliberately before editing.

## Requirements

- Ship in both web and Tauri through shared packages.
- Page shows active fleet rover selector, normalized relative output subfolder, start/stop per rover, state, elapsed time, current file, A/V/gap indicators, and errors.
- Support simultaneous status cards for different rovers; changing selected rover never retargets running sessions.
- Clip browser supports all-rover or per-rover filter, finalized metadata, refresh, selection, and play.
- Inline `<video controls preload="metadata">` uses a short-lived ticket URL and supports seeking; refresh expired tickets and clear them on auth loss/reconnect.
- Browser never uses `MediaRecorder`, accumulates JPEGs, creates files, or receives absolute server paths.

## Architecture

- Existing page remains socket/auth authority and passes typed `activeSocket`, auth state, fleet roster/selection into `MediaRecordingPage`.
- `use-recording-store` owns subscriptions, request correlation, status maps, list state, ticket expiry, reconnect clearing/refetch, and cleanup.
- Feature components remain presentational: output path, session controls/status, clip browser, playback panel.
- Visual direction follows the existing dark terminal/IDE surface, glass cards, monospace labels, syntax tokens, Lucide icons, responsive layout, and accessible status text.

## Related code files

- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording.ts` — mirrored wire/domain types.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording-fixtures.ts` — Rust-compatible fixtures needed by existing drafts.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts` — typed events.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/index.ts` and package exports.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-recording-store.ts` — lifecycle/catalog/ticket state.
- Preserve/reconcile `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-recording-store.test.tsx` — untracked user work.
- Create feature files in `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/`: `recording-output-path-control.tsx`, `recording-session-control.tsx`, `recording-clip-browser.tsx`, `recording-playback-panel.tsx`.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/media-recording-page.tsx`.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx` — view switch and prop wiring only.
- Modify feature/page index exports; add focused Vitest/Playwright files under existing test locations.

## Implementation Steps

1. Capture target repo status/diff. Review untracked `CameraViewer.test.tsx`, `use-scheduler-device-overrides.ts`, and `use-recording-store.test.tsx`; preserve their intent and isolate manual file events as `recording_session_*`. Done.
2. Mirror Rust contracts and golden fixtures. Remove quota/storage-control assumptions from manual store only with explicit user-work reconciliation, not blind deletion. Done.
3. Implement reducer/hook maps keyed by entity/recording ID, request correlation, signed-in-session gating, reconnect reset, list refresh, and ticket expiry cleanup. Done.
4. Build accessible path/session controls. Label path as relative to the deployment-configured root; reject obvious absolute/traversal input client-side while treating server validation as authoritative. Done.
5. Build multi-rover active-session layout and finalized clip browser with loading/empty/error states. Done.
6. Build playback panel using returned ticket URL. Handle `loadedmetadata`, range/seek errors, expiry refresh, selection change, logout, and unmount cleanup. Done.
7. Add the internal Control/Recordings view switch without reconnecting the socket or duplicating auth/session state. Done.
8. Add hook/component tests and deterministic fake Socket.IO Playwright coverage for desktop/mobile layouts and concurrent rover states. Done.

## Todo list

- [x] Reconcile/preserve untracked UI work.
- [x] Add shared recording types/fixtures/events.
- [x] Implement recording store hook.
- [x] Build path/session/list/player components.
- [x] Integrate shared page into web and Tauri.
- [x] Add unit/component/E2E tests.

## Success Criteria

- Admin can start rover A and rover B independently, stop either, and see correct terminal state.
- UI sends only relative paths and displays no host absolute path or ticket secret after expiry.
- Reconnect clears stale tickets/status, fetches authoritative state/list, and does not duplicate handlers.
- Only finalized clips expose Play; seeking issues range requests to the ticket endpoint.
- Existing live camera/audio controls and both application builds remain functional.

## Risk Assessment

- Risk: socket ownership refactor reconnects or duplicates listeners. Mitigation: keep one owner, pass typed socket, add mount/reconnect listener-count tests.
- Risk: old draft contracts conflict. Mitigation: event namespace table and deliberate test migration before component code.
- Risk: dense multi-rover UI. Mitigation: compact status cards, filterable table, responsive single-column fallback.

## Security Considerations

- Hide/disable controls unless signed in, but rely on server session enforcement; do not add role-based UI branches.
- Never place auth tokens, host paths, or long-lived capabilities in persisted frontend state/logs.
- Escape filenames/paths as text, validate ticket URL origin, and avoid HTML injection.

## Next steps

- Phase 06 validates both repositories together against live native/container backends.

## Final validation

- Outcome: 10/10
- Review: no findings
- Status: Phase 05 complete

## Unresolved questions

- None.
