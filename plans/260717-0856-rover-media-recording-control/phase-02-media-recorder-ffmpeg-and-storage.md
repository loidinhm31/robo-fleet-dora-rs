# Phase 02: Media Recorder, FFmpeg, and Safe Storage

## Context links

- [Parent plan](./plan.md)
- [Phase 01 contracts](./phase-01-shared-contracts-and-media-demand.md)
- [Backend research](./research/researcher-01-media-backend-report.md)
- [Architecture contract](../../ARCHITECTURE.md#manual-fleet-media-recording-and-playback)
- Depends on: Phase 01. Blocks Phases 03, 04, and 06.

## Overview

- Date: 2026-07-17
- Description: Build a multi-rover Dora recorder that streams JPEG and microphone PCM into finalized MP4 clips.
- Priority: P1 core functionality
- Implementation status: Done (2026-07-17 17:00 +07)
- Review status: Complete (2026-07-17 17:00 +07)
- Effort: 12h

## Key Insights

- Orchestra already receives validated JPEG and S16LE frames with entity/timestamp metadata; do not change rover encoding or Zenoh topics.
- Raw JPEGs must remain bounded in memory and pass directly to FFmpeg.
- Durable listing across restarts needs a small validated sidecar manifest; MongoDB and a retention ledger are unnecessary for manual MVP.
- Independent FFmpeg children isolate rover failures and permit concurrency, but fleet-wide limits must bound CPU/disk use.

## Requirements

- One active session per rover; configurable `RECORDING_MAX_CONCURRENT`, duration, output bytes, startup timeout, finalization timeout, and minimum free bytes.
- H.264 video plus AAC mono audio in MP4; input JPEG dimensions/fps and S16LE sample rate/channels come from validated metadata.
- Anonymous inherited pipes and argument arrays; never invoke a shell or interpolate user input into command text.
- Use capture timestamps as session time zero. Reject regressions/resets, count dropped video, and insert S16LE silence for audio gaps or absent mic.
- Write only encoded partial MP4 plus metadata under the root; never create `.jpg`/`.jpeg` files.
- Atomically publish MP4 and manifest only after FFmpeg exits successfully and media verification passes.

## Architecture

- Dora loop parses commands/frames and performs bounded dispatch only; session workers own FFmpeg I/O and never block live/control inputs.
- `SessionManager<HashMap<entity_id, Session>>` filters multiplexed bridge frames and enforces per-rover/global limits.
- Each worker owns video/audio queues, timestamp normalizer, anonymous pipe writers, FFmpeg child, counters, and cancellation/finalization state.
- Storage layout: `<root>/.partial/<recording_id>.mp4.partial`, `<root>/<relative-dir>/<recording_id>.mp4`, and adjacent `<recording_id>.manifest.json`.
- `ClipCatalog` validates manifests and file containment at startup/query time. Corrupt/partial entries are reported, never silently exposed.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/Cargo.toml` — node dependencies.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/main.rs` — Dora loop and bounded dispatch.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/config.rs` — strict env parsing/limits.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/path-resolver.rs` — root containment, symlink, relative path checks.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/session-manager.rs` — per-rover lifecycle/concurrency.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/ffmpeg-session.rs` — child, anonymous pipes, finalization.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/frame-timeline.rs` — PTS, silence, gap/reset counters.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/clip-catalog.rs` — manifests, listing, lookup.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/tests/recording-workflow.rs` — synthetic process tests.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml` — workspace member/dependencies only when used.

## Implementation Steps

1. Add the crate and strict startup config; require an existing/canonicalizable root, FFmpeg executable, writable partial/target directories, and sane limits.
2. Implement path resolution: reject absolute paths, `..`, empty/control components, symlink escape, excessive length, and non-directory parents; create allowed subfolders with restrictive permissions.
3. Implement session state machine `starting -> recording -> stopping -> completed|failed`; duplicate start/stop stays idempotent/correlated.
4. Implement per-entity frame filtering and bounded queues. Drop oldest video under pressure; synthesize timestamp-bounded silence for missing PCM.
5. Spawn FFmpeg with two anonymous input FDs, explicit demux/codec/container arguments, stderr cap, and owned-process supervision. Keep flags in one tested builder.
6. On stop/limit/shutdown, stop intake, close pipes, wait with timeout, terminate only the owned child if required, and collect exit diagnostics.
7. Run ffprobe validation for expected streams/container/duration; fsync as supported, write manifest via create-new temp, then atomically rename within one filesystem.
8. Scan startup partials/manifests safely; report stale partials and invalid clips without deleting unrelated files.
9. Add unit/process tests for paths, collisions, concurrency, gaps, process failure/hang, limits, restart catalog, and graceful shutdown.

## Todo list

- [x] Scaffold focused recorder modules under 200 LOC where practical.
- [x] Implement safe root/subfolder resolver.
- [x] Implement per-rover session manager and frame timelines.
- [x] Supervise dual-input FFmpeg without shell/temp JPEGs.
- [x] Finalize verified MP4 and manifest atomically.
- [x] Add synthetic and failure integration tests.

## Success Criteria

- Synthetic JPEG + PCM produces one seekable MP4 with one H.264 video and one AAC audio stream.
- Two rover sessions record/finalize independently; duplicate session for one rover is rejected.
- Output tree contains no `.jpg` or `.jpeg`; partial MP4 is never listed/playable.
- Audio gaps preserve duration within acceptance tolerance and appear in counters.
- FFmpeg crash/hang, disk failure, invalid path, or shutdown cannot block Dora or publish a false completed clip.

## Risk Assessment

- Risk: A/V drift from pipe arrival jitter. Mitigation: capture-time normalization, silence insertion, explicit FFmpeg sync policy, long-run skew tests.
- Risk: CPU overload with many rovers. Mitigation: configured concurrency cap, conservative preset, queue metrics, container benchmark gate.
- Risk: corrupt MP4 after interruption. Mitigation: partial state, owned child timeout, ffprobe verification, atomic publish.

## Security Considerations

- Open files with create-new semantics and restrictive modes; never overwrite caller-selected names.
- Resolve and re-check containment at creation, finalization, query, and playback handoff to reduce TOCTOU/symlink attacks.
- Bound FFmpeg stderr, manifest sizes, frame metadata/payloads, duration, bytes, and silence synthesis.

## Next steps

- Phase 03 exposes typed control/query/status and authenticated playback.
- Phase 04 installs/pins FFmpeg and mounts the same filesystem for recorder/web bridge.

## Unresolved questions

- None.
