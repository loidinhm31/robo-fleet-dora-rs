# Recording UI research

## Finding

- Target UI is a separate Git checkout: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`, remote `git@github.com:loidinhm31/robo-control-app.git`.
- `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo-control-app` does not exist, despite the backend repository guidance describing it there.
- Therefore backend/Socket.IO and UI changes cannot be one atomic single-repository commit. Plan explicit contract synchronization and validate both checkouts together.
- Target UI `main` is 7 commits ahead of `origin/main`. Preserve current user work: untracked `CameraViewer.test.tsx`, `use-scheduler-device-overrides.ts`, and `use-recording-store.test.tsx`.
- The untracked recording-store test references missing implementation/types. It is evidence of in-progress work, not a working feature.

## Existing UI architecture

- pnpm/Turborepo: web and Tauri native apps both import the same `RoboRoverControl` page from `packages/ui`.
- Entry points: `apps/web/src/App.tsx`, `apps/native/src/App.tsx`; both render only `RoboRoverControl` with URL/auth props.
- No router/navigation dependency. `RoboControlApp` documents `useRouter` as a no-op and supports parent-provided children only.
- The connected socket and authentication lifecycle live inside `packages/ui/src/components/pages/RoboRoverControl.tsx`; a sibling page cannot currently reuse that session without refactoring.
- `RoboRoverControl.tsx` is already 1,181 lines. Do not add the recording feature body there.
- Current camera panel is `packages/ui/src/components/features/CameraViewer.tsx`; it subscribes to binary JPEG `video_frame` and audio, emits `stream_control`, `camera_control`, and `audio_control`, and paints JPEG frames to canvas.
- Theme/conventions: terminal/IDE dark UI, `glass-card`, monospace labels, syntax color tokens, Lucide icons, responsive grids, `data-testid` on key controls, concise status badges.
- Atomic exports flow through feature/page `index.ts` files. Shared wire types flow through `packages/shared/src/types/index.ts` and package root `src/index.ts`.

## Current media contract

- Source-of-truth comment in `packages/shared/src/types/socket.ts` points to backend `web_bridge/src/main.rs`.
- Server: `video_frame(metadata, binary JPEG)`, `audio_frame(metadata, binary?)`, detections/tracking/telemetry, fleet/auth events.
- Client: `camera_control`, `stream_control`, `audio_control`, plus rover/tracking/fleet/auth commands.
- No committed recording types or events exist.
- `IMediaService` / `MediaService` cover live capture/playback only. They do not model server-side files, recording state, listing, or playback tickets.
- The untracked `use-recording-store.test.tsx` anticipates server events `recording_status`, `recording_clip_list_result`, `recording_playback_ticket_result`, `recording_quota`, `storage_control_status`; client request `recording_clip_list`; authenticated reconnect reset/refetch behavior.

## Recommended page design

- Add an internal two-view control/recordings switch, not React Router. Keep the existing socket/session owner mounted and pass its typed socket, auth state, and selected rover into a small recording page.
- New `MediaRecordingPage`: full-width terminal panel containing:
  - rover selector/context and `[IDLE|STARTING|RECORDING|FINALIZING|ERROR]` badge;
  - server output directory field, validation/error text, and saved effective path;
  - primary Start/Stop button, elapsed time, current file, received A/V indicators;
  - finalized recording table/cards: filename, rover, start/end, duration, size, container/codecs, state;
  - Refresh and Play actions; inline `<video controls preload="metadata">` player for selected finalized clip.
- Disable Start until authenticated, rover selected, path accepted, and no transition is pending. Disable Play until finalized. Confirm path changes while recording and surface finalization failures.
- Server owns recording and muxing. Browser must never accumulate live JPEGs, run `MediaRecorder`, or create raw JPEG downloads.
- Playback should be an HTTP range-capable URL obtained from an authenticated short-lived ticket. Avoid sending whole video files over Socket.IO or loading a complete file into a Blob.
- Treat the path as a server filesystem path. Backend must canonicalize and restrict it to deployment-configured recording roots; UI hints are not a security boundary.

## Proposed shared contract

- Add `packages/shared/src/types/recording.ts`; export it from `types/index.ts`.
- Core types: `RecordingState`, `RecordingStatus`, `RecordingConfigRequest/Result`, `RecordingControlRequest/Result`, `RecordingClipSummary`, `RecordingClipListRequest/Result`, `RecordingPlaybackTicketRequest/Result`, `RecordingError`, optional `RecordingQuota`.
- Every command/result should carry `request_id`; every status/clip should carry `entity_id`; timestamps use epoch milliseconds; sizes use integer bytes.
- Status includes effective output directory, active clip ID, state, started time, duration, video/audio presence, and structured error.
- Clip includes opaque `clip_id`, display filename, finalized relative path (not arbitrary host path), duration, byte length, MIME/container, video/audio codec, and finalized state.
- Extend typed socket maps with client `recording_configure`, `recording_control`, `recording_clip_list`, `recording_playback_ticket`; server matching results plus `recording_status` and optional `recording_quota`.
- Reconcile these names with backend source of truth before coding. Reuse the in-progress test's established list/ticket/status names unless backend research finds a committed contract.

## Exact UI implementation boundaries

- `packages/shared/src/types/recording.ts`: wire/domain types.
- `packages/shared/src/types/socket.ts`: typed Socket.IO events only.
- `packages/shared/src/types/index.ts`: exports.
- `packages/ui/src/hooks/use-recording-store.ts`: subscribe/unsubscribe, reconnect reset, request correlation, status/list/ticket state; complete or supersede existing untracked test deliberately.
- `packages/ui/src/components/features/recording-output-path-control.tsx`: path form and config result.
- `packages/ui/src/components/features/recording-session-control.tsx`: start/stop/status.
- `packages/ui/src/components/features/recording-clip-browser.tsx`: list, selection, refresh.
- `packages/ui/src/components/features/recording-playback-panel.tsx`: ticket lifecycle and `<video>`.
- `packages/ui/src/components/pages/media-recording-page.tsx`: page composition under 200 lines.
- `packages/ui/src/components/pages/RoboRoverControl.tsx`: only view switch/navigation and props into the page; extract socket/session context first if conditional rendering would duplicate connection logic.
- Update `features/index.ts` and `pages/index.ts`; web/native entry points need no changes if view switching stays within shared page.

## Validation

- Vitest (`pnpm --filter @robo-fleet/ui test`): reducer/hook transitions, auth gating, reconnect cleanup/refetch, stale result IDs, start/stop/finalize errors, ticket expiry, URL cleanup.
- Component tests: path validation feedback, disabled-state matrix, recording status, empty/list/error states, play only finalized clips.
- Playwright: responsive navigation and recording page; use fake typed Socket.IO backend for deterministic status/list/ticket; verify `<video>` receives ticket URL, not JPEG frames.
- Gates: root `pnpm check-types`, `pnpm lint`, `pnpm build`; live integration against backend confirms auth, allowed path, A/V MP4 finalization, HTTP Range seek, and no raw JPEG files in output.

## Unresolved questions

1. Is output path freely chosen per session, or selected beneath one deployment-configured recording root?
2. Should recordings be scoped to selected rover only or expose an all-rover fleet list?
3. Required container/codecs: MP4/H.264+AAC, WebM, or another browser-playable target?
4. Must the same recordings UI ship in both web and Tauri apps, or web only?
5. Should playback be direct authenticated HTTP, short-lived ticket URL, or existing storage/R2 signed URL contract?
6. Does the in-progress untracked recording-store test belong to this task and define intended event names?
