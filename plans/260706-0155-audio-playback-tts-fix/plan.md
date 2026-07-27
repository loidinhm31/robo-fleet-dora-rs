---
title: "Audio Playback and TTS Reliability Fix"
description: "Pace rover TTS, preserve walkie PCM timing, and make microphone suppression reliable and observable."
status: in-progress
priority: P1
effort: 29h
branch: main
tags: [bugfix, audio, backend, frontend, critical]
created: 2026-07-06
---

# Audio Playback and TTS Reliability Fix

## Overview

Fix choppy walkie audio, speaker-to-microphone loopback, and truncated rover TTS across the Rust backend and current UI checkout. Use paced 20 ms media frames, bounded latency queues, all-or-none playback enqueue, sequenced suppression state, and explicit terminal accounting.

## Phases

| # | Phase | Status | Effort | Progress | Link |
|---|---|---|---:|---:|---|
| 1 | Contract and architecture | Complete | 4h | 100% | [phase-01](./phase-01-contract-and-architecture.md) |
| 2 | Walkie ingress and transport | Complete | 6h | 100% | [phase-02](./phase-02-walkie-ingress-and-transport.md) |
| 3 | TTS pacing and lifecycle | Complete | 6h | 100% | [phase-03](./phase-03-tts-pacing-and-lifecycle.md) |
| 4 | Playback, suppression, observability | Complete | 8h | 100% | [phase-04](./phase-04-playback-suppression-and-observability.md) |
| 5 | End-to-end verification | In progress | 5h | 85% | [phase-05](./phase-05-end-to-end-verification.md) |

## Dependencies

- Backend: `/mnt/data/ws/sharing/robo-fleet-dora-rs`
- UI: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Root-cause report: [brainstorm report](../reports/brainstorm-260706-0107-audio-playback-tts-fix.md)
- Coordinated UI/backend deployment required for the hard Socket.IO cutover.
- Workstation speaker and microphone access still required for final hardware acceptance and follow-up research.

## Fixed Decisions

- `audio_stream` becomes metadata plus exactly one binary F32LE attachment; it is not a binary-only message.
- Legacy `{ audio_data }` frames are rejected after coordinated deployment.
- Browser emits 20 ms mono frames at the actual `AudioContext.sampleRate`.
- Walkie one-way acceptance ceiling: 250 ms.
- TTS buffer: 1,000 ms; full-buffer retry deadline: 60 ms.
- Dora media queues: four 20 ms frames; lifecycle/control queues: eight.
- Microphone suppression tail remains 400 ms; walkie authority remains 250 ms.
- Credit/ACK flow control is deferred unless paced acceptance still overruns.

## Validation Summary

**Validated:** 2026-07-06  
**Questions asked:** 5

Confirmed: current UI sibling checkout is in scope, 250 ms walkie ceiling, hard wire cutover, 60 ms TTS stall failure, and mandatory acoustic hardware acceptance.

## Unresolved Questions

- Manual hardware acceptance on 2026-07-06 produced bad results. Code-level verification is done, but acoustic evidence and root-cause research remain before Phase 05 can close.
