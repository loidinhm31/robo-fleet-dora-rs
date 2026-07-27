---
title: "Approach A Audio/Video Stream Performance"
description: "Replace JSON audio and timer-driven playback with capture-correlated binary PCM and a bounded Web Audio timeline scheduler."
status: done
priority: P2
effort: ~28h
branch: main
tags: [performance, backend, frontend, refactor]
created: 2026-06-28
updated: 2026-07-01
---

# Approach A Audio/Video Stream Performance

## Overview

Implement revised Approach A across rover, orchestra, web bridge, and UI. Audio conversion (Float32→Int16LE) now happens on the rover (`rover-kiwi/audio_converter`), reducing Zenoh bandwidth by 50%. Preserve one capture identity end-to-end, measure current behavior before cutover, send S16LE as a Socket.IO binary attachment, and schedule playback without recursive timers.

## Scope

- Backend: `/mnt/data/ws/sharing/robo-fleet-dora-rs`
- Frontend: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Keep one Socket.IO connection and current PCM format.
- Defer separate socket, AudioWorklet, codecs, and WebRTC.

## Phases

| # | Phase | Status | Progress | Effort | Link |
|---|---|---|---:|---:|---|
| 1 | Capture identity and transport observability | Done | 100% | 7h | [phase 01](./phase-01-capture-identity-and-transport-observability.md) |
| 2 | Controlled baseline and evidence gate | Done | 100% | 4h | [phase 02](./phase-02-controlled-baseline-and-evidence-gate.md) |
| 3 | Binary browser audio cutover | Done | 100% | 4h | [phase 03](./phase-03-binary-browser-audio-cutover.md) |
| 4 | Bounded Web Audio timeline scheduler | Done | 100% | 7h | [phase 04](./phase-04-bounded-web-audio-timeline-scheduler.md) |
| 5 | End-to-end validation and rollout | Done | 100% | 4h | [phase 05](./phase-05-end-to-end-validation-and-rollout.md) |

**Overall progress:** 100% (5/5 phases complete). The plan is closed
from an automation/code perspective. The 10-minute live matrix, direct-mode
lifecycle, and capture-to-audible latency remain **operator-deferred** and
are documented in the [Phase 05 validation report](./reports/phase-05-validation.md#operator-runbook)
(Operator Runbook).

## Dependencies

- Frontend-first rollout for legacy JSON compatibility.
- Rover publishes Int16LE via Zenoh; orchestra receives Int16LE directly (no orchestra-side audio_converter). Orchestra must handle both Float32 (old rover) and Int16LE (new rover) during transition.
- Stable microphone and no operator audio toggles during benchmarks.
- Rover/workstation clock offset recorded; <=5 ms for latency acceptance.
- Phase 3 automated validation passed; bounded scheduler implementation completed in Phase 4 (new `useAudioStream` hook, `audio-timeline-scheduler.ts`, `audio-stream-metrics` updates, `CameraViewer` audio-section refactor); source-controlled Playwright `stream-live.spec.ts` is now the live e2e gate. Phase 5 covers the 10-minute matrix, `useAudioStream` lifecycle, and capture-to-audible latency.

## Key Gates

- Stop after Phase 2 if controlled reproduction is invalid or stop/start commands persist.
- Do not claim capture-to-audible latency without hardware loopback; browser metric is scheduled-start estimate.
- Advance to Approach B/C only from measured post-Approach-A evidence.

## Architecture

- Design recorded in [ARCHITECTURE.md](../../ARCHITECTURE.md#binary-browser-audio-and-bounded-playback).
- Research: [Approach A report](./research/researcher-01-report.md).
- Code map: [codebase analysis](./reports/codebase-analysis.md).
- Phase 01 evidence: [completion report](./reports/phase-01-completion.md).
- Phase 04 evidence: [implementation report](../../reports/implementation-260630-2103-audio-video-stream-performance-approach-a.md).
- Phase 05 evidence: [validation report](./reports/phase-05-validation.md) (rerun after the Phase 04 `robo-control-app` updates).

## Backlog

- ~~Web shutdown totals currently sum per-client counters only for clients still connected. Disconnect cleanup removes historical audio delivery/drop counts. Future fix: maintain process-level cumulative counters and test that totals retain counts after client disconnect.~~ **Resolved 2026-07-01** — see [phase 06 completion report](./reports/phase-06-cumulative-audio-counters.md). `web_bridge` now maintains `Arc<AudioDeliveryCounters>` in `SharedState` (lifetime totals via `AtomicU64` with relaxed ordering), increments it next to every per-client counter mutation, and reads it at shutdown. A unit test in `common/web_bridge/src/audio_counters.rs` (`cumulative_totals_retain_counts_after_caller_drops`) directly covers the bug.

## Validation Summary

**Validated:** 2026-06-28
**Questions asked:** 7

### Confirmed Decisions

- **audio_converter location**: Move from `orchestra/audio_converter/` to `rover-kiwi/audio_converter/`. Conversion happens on the rover to reduce Zenoh audio bandwidth from 512 Kbps (Float32) to 256 Kbps (Int16LE).
- **Zenoh audio format**: Int16LE only over Zenoh in split mode. The rover converts Float32→Int16LE before publishing to Zenoh.
- **speech_recognizer rename**: Rename `orchestra/speech_recognizer` to `orchestra/central_speech_recognizer` (package `central_speech_recognizer`). Upgrade from whisper tiny to **whisper base** model.
- **central_speech_recognizer scope**: Runs in split mode only (on workstation). Processes only web UI microphone audio (`web-bridge/voice_command_audio`, already Float32). Does NOT receive rover audio — that goes only to `web_bridge` for browser playback.
- **edge_speech_recognizer**: New node at `rover-kiwi/edge_speech_recognizer/`. Placeholder/TODO crate for now. When implemented, it will receive Float32 audio directly from `audio_capture` (bypassing audio_converter). Handles STT on the rover in direct/standalone mode.
- **edge_speech_recognizer dataflow wiring**: Add to both `rover-kiwi-dataflow.yml` and `rover-kiwi-direct-dataflow.yml`, commented out with TODO notes until implemented.
- **audio_converter in direct mode**: Continues to run on the rover in direct/standalone mode, feeding Int16LE audio to the local web_bridge.

### Architecture Change

```text
SPLIT MODE (before):
  audio_capture (rover, F32) → Zenoh F32 → orchestra audio_converter (F32→S16LE) → web_bridge → browser
                                          → speech_recognizer (F32, whisper tiny) ← also receives web UI audio

SPLIT MODE (after):
  audio_capture (rover, F32) → audio_converter (rover, F32→S16LE) → Zenoh S16LE → web_bridge → browser
  web-bridge/voice_command_audio (F32) → central_speech_recognizer (whisper base, web UI audio only)
  [future] audio_capture (rover, F32) → edge_speech_recognizer (rover, local STT)

DIRECT MODE (after):
  audio_capture (rover, F32) → audio_converter (rover, F32→S16LE) → web_bridge → browser
  [future] audio_capture (rover, F32) → edge_speech_recognizer (rover, local STT)
```

### Action Items

- [x] Update phase-01 file references: `orchestra/audio_converter` → `rover-kiwi/audio_converter`
- [x] Update phase-01 to reflect `central_speech_recognizer` rename (web UI audio only)
- [x] Add `edge_speech_recognizer` TODO crate creation to phase-01 or a new prerequisite phase
- [x] Update `orchestra-dataflow.yml`: rename `speech-recognizer` to `central-speech-recognizer`, remove rover audio input (`audio_rover`), keep only `audio_web`, update model path to whisper base
- [x] Update `rover-kiwi-dataflow.yml`: add `audio-converter` node (fed by `audio-capture/audio`), add commented-out `edge-speech-recognizer` node
- [x] Update `rover-kiwi-direct-dataflow.yml`: update `audio-converter` input (already correct), add commented-out `edge-speech-recognizer` node
- [x] Update `orchestra-dataflow.yml`: remove `audio-converter` node, wire `orchestra-bridge/audio_frame` (now Int16LE) directly to `web-bridge`
- [x] Update rover Zenoh bridge to publish Int16LE (from `audio_converter` output) instead of raw Float32
- [x] Update orchestra Zenoh bridge to expect Int16LE instead of Float32
- [x] Update `central_speech_recognizer`: remove rover audio input, keep only web UI audio input (Float32)
- [x] Update workspace `Cargo.toml`, `docker/Cargo.rover.toml`, `docker/Cargo.orchestra.toml` member paths
- [x] Update `Dockerfile.orchestra` and `Dockerfile.rover-kiwi` COPY/build paths
- [x] Update `ARCHITECTURE.md` directory tree and audio pipeline documentation

## Unresolved Questions

- Confirm required production network path.
- Confirm prior stop/start commands were intentional.
