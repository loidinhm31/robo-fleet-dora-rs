---
title: "Edge Voice Supertonic x86 Implementation"
description: "Replace Kokoro and Piper with rover-role Supertonic, then validate native Dora, live web E2E, and amd64 Docker on the current workstation."
status: completed
priority: P1
effort: 66h
branch: main
tags: [feature, backend, frontend, infra, audio, tts, tech-debt]
created: 2026-07-04
---

# Edge Voice Supertonic x86 Implementation

## Overview

Implement the agreed edge-voice architecture from the brainstorm report. Keep Orchestra and Rover as separate Dora dataflows on the current x86_64 workstation. Use local Docker MongoDB for native testing, then verify the complete amd64 container stack. Raspberry Pi and ARM acceptance are excluded.

## Phases

| # | Phase | Model | Status | Effort | Progress | Link |
|---|---|---|---|---:|---:|---|
| 1 | Architecture and contracts | GPT-5.5 | Completed | 4h | 100% | [phase-01](./phase-01-architecture-and-contract-gate.md) |
| 2 | Model reset and bootstrap | GPT-5.4 | Completed | 6h | 100% | [phase-02](./phase-02-model-cache-reset-and-bootstrap.md) |
| 3 | Edge voice engine | GPT-5.5 | Completed | 10h | 100% | [phase-03](./phase-03-edge-voice-engine.md) |
| 4 | Playback and capture suppression | GPT-5.5 | Completed | 10h | 100% | [phase-04](./phase-04-source-aware-playback-and-capture-suppression.md) |
| 5 | Fleet transport and authority | GPT-5.4 | Completed | 8h | 100% | [phase-05](./phase-05-fleet-transport-and-runtime-authority.md) |
| 6 | Web controls and alerts | GPT-5.4 | Completed | 6h | 100% | [phase-06](./phase-06-web-ui-controls-and-alerts.md) |
| 7 | Docker cleanup and MongoDB | GPT-5.4 | Completed | 6h | 100% | [phase-07](./phase-07-docker-cleanup-and-local-mongodb.md) |
| 8 | Native x86 integration | GPT-5.4 + mini | Completed | 6h | 100% | [phase-08](./phase-08-native-x86-integration-and-benchmark.md) |
| 9 | Live web E2E | GPT-5.4 + mini | Completed | 5h | 100% | [phase-09](./phase-09-live-web-e2e.md) |
| 10 | Docker verification and docs | GPT-5.4 + mini | Completed | 5h | 100% | [phase-10](./phase-10-amd64-docker-verification-and-documentation.md) |

## Execution Model Routing

- GPT-5.5: architecture, FFI/concurrency, real-time audio, cross-layer root cause work. Never assign raw long-log reading.
- GPT-5.4: normal implementation, tests, bridge/UI/Docker changes, reviews.
- GPT-5.4-mini subagent: bounded Docker/Dora/Playwright logs, repetitive inventory, test-output classification. Return command, exit code, first causal error, evidence lines, and suspected subsystem.
- Escalate mini findings to GPT-5.4 when ambiguous. Escalate to GPT-5.5 only for contract, unsafe FFI, concurrency, or real-time audio decisions.
- Run at most one log-reading subagent alongside the main implementer. The main session owns pass/fail decisions.

## Dependencies

- Parent report: [Edge Voice Supertonic brainstorm](../reports/brainstorm-260704-1042-edge-voice-supertonic.md)
- Research: [Rust production comparison](./research/01-supertonic-rust-production-comparison.md)
- Decisions: [Locked decisions and model routing](./reports/01-locked-decisions-and-model-routing.md)
- Scout: [Repository integration surface](./scout/01-repository-integration-surface.md)
- External UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`

## Global Completion Gate

- Native Orchestra and Rover Dora dataflows pass live TTS E2E with local container MongoDB.
- amd64 Orchestra/Rover Docker stack passes the same E2E.
- No active Kokoro, Piper, GGML Whisper, or `sherpa-rs` production path remains.
- Existing dirty changes in both repositories remain intact.

## Progress Update

- 2026-07-04 17:59 ICT: Phase 01 completed and approved. Plan remains in progress; Phases 02-10 pending.
- 2026-07-04 21:25 ICT: Phase 02 completed and approved. Plan remains in progress; Phases 03-10 pending.
- 2026-07-04 23:19 ICT: Phase 03 completed and approved. Plan remains in progress; Phases 04-10 pending.
- 2026-07-05 05:18 ICT: Phase 04 completed and approved. Plan remains in progress; Phases 05-10 pending.
- 2026-07-05 06:46 ICT: Phase 05 completed and approved. Plan remains in progress; Phases 06-10 pending.
- 2026-07-05 09:40 +07: Phase 06 completed and approved. Plan remains in progress; Phases 07-10 pending.
- 2026-07-05 11:02 ICT: Phase 07 completed and approved. Plan remains in progress; Phases 08-10 pending.
- 2026-07-05 15:08 +07: Phase 08 completed and approved. Native x86 benchmark gates passed; Phases 09-10 pending.
- 2026-07-05 17:24 +07: Phase 09 completed and approved. Live web E2E passed root scripts, stream regression, and three consecutive focused edge-voice runs; Phase 10 pending.
- 2026-07-05 21:46 +07: Phase 10 completed and approved. amd64 Docker stack stayed healthy, both Docker live Playwright suites passed, docs were aligned to the verified workstation flow, and the overall plan is complete.

## Unresolved Questions

- Legal approval for redistributing the OpenRAIL-M Supertonic model remains external to engineering.
