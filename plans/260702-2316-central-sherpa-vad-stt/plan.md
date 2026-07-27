---
title: "Central Sherpa VAD STT and Dual-Source Voice"
description: "Replace Whisper with startup-selected Sherpa VAD/offline Zipformer for isolated browser and fleet rover speech flows."
status: deprecated
priority: P2
effort: 60h
branch: main
tags: [feature, backend, frontend, api, infra]
created: 2026-07-02
updated: 2026-07-03
superseded_by: ../260703-1124-central-sherpa-stt/plan.md
---

# Central Sherpa VAD STT and Dual-Source Voice

> **Deprecated 2026-07-03.** Superseded by [Central Sherpa STT Remaining Delivery](../260703-1124-central-sherpa-stt/plan.md). Phases 1-3 remain completed history; all remaining delivery is tracked in the new plan.

## Overview

Implement final-only English/Vietnamese STT using official `sherpa-onnx`, Silero VAD, and offline Zipformer. Browser and rover microphones remain separate from capture through UI display and command targeting.

## Decisions

- Profiles: `en-vad-offline` default, `vi-vad-offline`; startup-only.
- Results: final utterances only; confidence optional and never fabricated.
- Browser transcript: origin socket only; command targets rover selected at capture start.
- Rover transcript: broadcast with `entity_id`; command targets source rover.
- Parser: deterministic only; automatic TTS feedback removed.
- Runtime: CPU, one bounded offline decode worker, independent per-source VAD sessions.

## Phases

| # | Phase | Status | Progress | Effort | Detail |
|---|---|---|---:|---:|---|
| 1 | Architecture, contracts, baseline | Complete | 100% | 5h | [Phase 01](./phase-01-architecture-contracts-baseline.md) |
| 2 | Sherpa runtime and models | Complete | 100% | 6h | [Phase 02](./phase-02-sherpa-runtime-models.md) |
| 3 | Central VAD recognizer | Complete | 100% | 12h | [Phase 03](./phase-03-central-vad-recognizer.md) |
| 4 | Web bridge dual-source transport | Superseded | — | 10h | Implemented as [new Phase 01](../260703-1124-central-sherpa-stt/phase-01-web-bridge-dual-source-transport.md) |
| 5 | Source-aware command routing | Superseded | — | 6h | Tracked as [new Phase 02](../260703-1124-central-sherpa-stt/phase-02-source-aware-command-routing.md) |
| 6 | Dual transcription UI | Superseded | — | 8h | Tracked as [new Phase 03](../260703-1124-central-sherpa-stt/phase-03-dual-source-voice-ui.md) |
| 7 | System validation gate | Superseded | — | 7h | Tracked as [new Phase 04](../260703-1124-central-sherpa-stt/phase-04-system-validation-gate.md) |
| 8 | Whisper/edge retirement and deployment | Superseded | — | 6h | Tracked as [new Phase 05](../260703-1124-central-sherpa-stt/phase-05-legacy-retirement-and-deployment.md) |

## Dependencies

- Reference API: `/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/`
- UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Research: [research synthesis](./research/research-synthesis.md)
- Source brainstorm: [central Sherpa STT brainstorm](../reports/brainstorm-260702-1750-central-sherpa-stt.md)

## Delivery Gates

- Historical phases 1-3 are complete.
- Remaining delivery gates moved to the [superseding plan](../260703-1124-central-sherpa-stt/plan.md).

## Unresolved Questions

None.
