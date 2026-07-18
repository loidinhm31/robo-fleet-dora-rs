---
title: "Recorder queue and elapsed timer repair"
description: "Prevent audio backlog from truncating recorded video and show live recording elapsed time."
status: in-progress
priority: P1
effort: 6h
branch: main
tags: [bugfix, backend, frontend, media]
created: 2026-07-18
---

# Recorder queue and elapsed timer repair

## Overview

Repair two confirmed recording regressions without changing Zenoh, Dora, Socket.IO,
the shared wire contract, or recorder configuration. Keep the recorder's existing
total media-memory bound. Fix video admission inside it, then add a browser-local
elapsed display that stays live between deduplicated status events.

## Phases

| # | Phase | Status | Effort | Link |
|---|---|---|---|---|
| 1 | Preserve video admission in bounded recorder queue | Done | 2h | [phase 01](./phase-01-preserve-video-admission.md) |
| 2 | Render monotonic active elapsed time in shared UI | Pending | 2h | [phase 02](./phase-02-render-live-recording-elapsed-time.md) |
| 3 | Verify sustained A/V recording and UI lifecycle | Pending | 2h | [phase 03](./phase-03-verify-recording-regressions.md) |

## Fixed design decisions

- Retain one bounded FIFO recorder queue. When it is audio-only and full, a
  new video frame evicts the oldest audio frame; it is never rejected solely
  because audio filled the queue.
- Keep existing newest-video replacement when the full queue already contains
  video. No queue-size or wire-protocol change.
- Keep backend duration authoritative at terminal states. The UI derives a
  live active display from the last received duration plus `performance.now()`.
- Architecture Gate 1 is intentionally skipped: this is a pure bug fix that
  preserves the documented media paths and bounds.

## Dependencies

- [Queue research](./research/researcher-01-recorder-queue.md)
- [Elapsed-time research](./research/researcher-02-elapsed-time.md)
- `ARCHITECTURE.md` manual recording invariants

## Unresolved questions

- None blocking. A later observability enhancement may add explicit audio-drop
  counters if post-fix load testing needs finer diagnosis.
