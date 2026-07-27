---
title: "Manual recording deletion, camera controls, and timestamp burn-in"
description: "Add safe clip deletion, persistent camera recording controls, and UTC timestamps to saved MP4s."
status: complete
priority: P1
effort: 2d
branch: main
tags: [feature, backend, frontend, media, security]
created: 2026-07-19
---

# Manual Recording Controls and Timestamp Plan

## Overview

Extend the existing Orchestra recorder path and shared React UI. Keep one rover camera owner and one JPEG fan-out; do not introduce browser-side recording or a second `/dev/video*` reader.

## Phases

| # | Phase | Status | Link |
|---|---|---|---|
| 1 | Recorder delete contract and saved timestamp | Complete | [phase-01](./phase-01-backend-media.md) |
| 2 | Shared UI store, CameraViewer controls, delete UX | Complete | [phase-02](./phase-02-frontend-recording-ui.md) |
| 3 | Verification, browser gate, rollout | Complete | [phase-03](./phase-03-validation-rollout.md) |

## Preflight contract

- Output: authenticated permanent deletion of finalized MP4+manifest pairs; CameraViewer start/stop and elapsed REC indicator; Recordings-tab active badge; saved-MP4 UTC `YYYY-MM-DD HH:MM:SS UTC` burn-in.
- Acceptance: active sessions remain stoppable while switching views; viewer and recorder demands coexist; unsafe/active/partial clips cannot be deleted; successful delete clears playback and list state; extracted saved frames contain UTC text.
- Scope: existing Rust Dora/Socket.IO recorder path and `robo-control-app` shared UI. Out: trash/restore, live burn-in, pagination, browser `MediaRecorder`, second camera opener, unrelated voice-camera arbitration.
- Public/risk areas: versioned shared types, authenticated/rate-limited destructive action, path/symlink containment, ticket invalidation, FFmpeg/font capability, bounded queues and disk usage, existing dirty test edit preservation.
- Side effects: no DB/schema migration; no new rover topic; no new camera process; deploys need timestamp config/font validation; tests must cover auth, races, partial pairs, and UI navigation.
- Testing: Rust unit/integration + FFmpeg/ffprobe/frame extraction; Vitest/typecheck/build/lint; Playwright recording flow and focused accessibility/browser evidence.
- Open questions: none (UTC, permanent delete, CameraViewer + tab badge confirmed).

## Design choice

Recommended: recorder-owned delete and FFmpeg `drawtext` in the existing session, with one hoisted recording store. Rejected browser `MediaRecorder` (duplicates capture, weak audio/path lifecycle) and a direct second FFmpeg camera reader (device conflict and bypasses demand ownership).
