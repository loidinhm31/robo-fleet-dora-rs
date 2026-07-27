# Codebase Analysis

Date: 2026-06-28

## Documentation

- `docs/codebase-summary.md`: fresh (same day); no scout run required.
- `ARCHITECTURE.md`: canonical architecture equivalent to requested system architecture doc.
- Missing: `docs/code-standards.md`, `docs/system-architecture.md`, `docs/project-overview-pdr.md`.
- Follow repository `CLAUDE.md`, root development rules, existing Rust/TypeScript patterns.

## Backend Files

- `rover-kiwi/audio_capture/src/main.rs`: frame creation, ring overflow currently silent.
- `rover-kiwi/zenoh_bridge/src/main.rs`: raw F32 byte publication, ignored publish result.
- `orchestra/zenoh_bridge/src/main.rs`: unsafe raw-byte F32 decode, generated metadata.
- `orchestra/audio_converter/src/main.rs`: F32 -> S16LE, metadata clone already present.
- `common/web_bridge/src/main.rs`: JSON byte-array event, generated capture identity, ignored emit result.
- `robo_rover_lib/src/types/video_types.rs`: unused audio types mixed into 500+ line video module.
- `robo_rover_lib/src/utils/metric_window.rs`: reusable window and sequence tracking patterns.
- Split and direct dataflow YAMLs already route the required audio path; no new node/port needed.

## Frontend Files

- Repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.
- `packages/ui/src/components/features/CameraViewer.tsx`: 1,100+ lines; audio must be extracted.
- `packages/shared/src/types/socket.ts`: current audio event embeds `number[]`.
- No frontend unit-test runner currently configured.
- `CameraViewer.tsx` has unrelated uncommitted video payload validation; preserve it.

## Existing Patterns to Reuse

- Versioned `JpegFramePacket` encode/decode and validation tests.
- `MetricWindow` and `FrameSequenceTracker` in each video transport stage.
- `browser_video_frame_payload()` plus `Packet::bin_event` protocol test.
- Frontend video binary normalization for `ArrayBuffer`, views, Blob, and legacy arrays.

## Modularization Boundary

- Create Rust `audio_types.rs`; snake_case required by Rust module syntax.
- Extract TypeScript `audio-frame.ts`, `audio-stream-metrics.ts`, and `audio-timeline-scheduler.ts`.
- Wrap browser lifecycle in `use-audio-stream.ts`; keep CameraViewer presentation-focused.

## Working Tree Constraint

- Both repositories are dirty. Implementation must patch narrowly and never replace unrelated edits.
- `ARCHITECTURE.md` had CRLF-only changes before this plan; retain CRLF.

## Unresolved Questions

- None beyond the source report's deployment/SLA questions.
