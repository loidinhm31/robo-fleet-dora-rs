---
title: "Manual Fleet Media Recording and Playback"
description: "Add concurrent Orchestra-side rover A/V recording with safe server paths and shared web/Tauri controls."
status: pending
priority: P2
effort: 46h
branch: main
tags: [feature, backend, frontend, multimedia, infra]
created: 2026-07-17
---

# Manual Fleet Media Recording and Playback

## Overview

Add one Orchestra `media-recorder` Dora node. It consumes per-rover JPEG and microphone PCM from `orchestra-bridge`, writes browser-playable H.264/AAC MP4 files without raw JPEG persistence, and exposes signed-in-user start/stop/list/play controls in the separate shared web/Tauri UI.

## Frozen decisions

- Manual recording only. No schedules, R2, retention engine, quota UI, or MongoDB.
- Any user with a valid signed-in session may record, list, and play; this feature adds no role checks or RBAC.
- Concurrent sessions allowed across rovers; one active session per rover.
- Deployment owns an allowlisted recording root in a dedicated directory below `/home`; UI supplies only a relative subfolder.
- Rover microphone only. Missing/gapped audio becomes timestamp-correct silence.
- `web-bridge` aggregates per-rover browser and recorder media demand; recorder commands never use mutable fleet selection.
- File-session events use `recording_session_*` to avoid the existing scheduler `recording_command`/`recording_status` draft contract.
- Final output: H.264/AAC MP4. Only atomically finalized clips are listed/playable.

## Phases

| # | Phase | Status | Progress | Effort | Link |
|---|---|---|---:|---:|---|
| 1 | Shared contracts and media demand | Done (2026-07-17 11:47 +07) | 100% | 7h | [phase-01](./phase-01-shared-contracts-and-media-demand.md) |
| 2 | FFmpeg recorder and storage core | Pending | 0% | 12h | [phase-02](./phase-02-media-recorder-ffmpeg-and-storage.md) |
| 3 | Backend control, catalog, playback | Pending | 0% | 10h | [phase-03](./phase-03-backend-control-catalog-and-playback.md) |
| 4 | Orchestra container deployment | Pending | 0% | 5h | [phase-04](./phase-04-orchestra-container-deployment.md) |
| 5 | Recording control and playback UI | Pending | 0% | 7h | [phase-05](./phase-05-recording-control-and-playback-ui.md) |
| 6 | End-to-end verification and rollout | Pending | 0% | 5h | [phase-06](./phase-06-end-to-end-verification-and-rollout.md) |

## Dependencies

- Existing versioned JPEG and S16LE PCM bridge outputs with `entity_id` and capture timestamps.
- FFmpeg/ffprobe available in native and Orchestra container environments.
- Writable host directory below `/home`, mounted at `/recordings` in the container.
- Coordinated changes in `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.

## Research

- [Backend/media report](./research/researcher-01-media-backend-report.md)
- [External UI report](./research/researcher-02-recording-ui-report.md)
- [Architecture contract](../../ARCHITECTURE.md#manual-fleet-media-recording-and-playback)

## Validation Summary

**Validated:** 2026-07-17
**Questions asked:** 4

### Confirmed Decisions

- Recordings are server-owned and continue across initiating UI disconnect/reload; reconnect receives authoritative status.
- Server may create validated relative subfolders beneath the configured root with restrictive permissions.
- Default concurrency covers all active rovers, subject to hard CPU, disk, duration, byte, and free-space safety guards.
- Missing microphone input produces a silent AAC track and an observable missing-audio condition.

### Action Items

- [ ] Before implementation, revise Phase 02/04 concurrency-default wording and tests from a small fixed cap to the active fleet size while retaining an explicit deployment override and fail-closed resource guards.

## Unresolved questions

- None. Timed-out choices use the recommended defaults recorded above.
