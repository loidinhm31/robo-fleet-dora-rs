---
title: "Central Sherpa STT Remaining Delivery"
description: "Complete secure dual-source transport, source-aware command routing, UI integration, validation, and conditional legacy retirement for the committed Sherpa VAD/offline runtime."
status: completed
priority: P2
effort: 37h
branch: main
tags: [feature, backend, frontend, api, infra, critical]
created: 2026-07-03
updated: 2026-07-04
---

# Central Sherpa STT Remaining Delivery

## Overview

Finish the already-started central Sherpa migration. Reuse committed contracts, model provisioning, and dual-source VAD/offline decode. Deliver browser ownership/privacy, rover-safe command targeting, UI wiring, system validation, then remove obsolete Whisper and edge STT paths.

## Locked Scope

- Profiles: global startup-only `en-vad-offline` or `vi-vad-offline`.
- Results: final-only; confidence optional and never synthesized.
- Browser: private transcript; target rover captured server-side at stream start.
- Rover: fleet-visible transcript; command target equals source rover.
- Parser: deterministic only; no automatic parser TTS.
- Deferred: online/partial STT, runtime profile switching, AI interpreter, TTS suppression/AEC.

## Baseline

- Complete: contracts/architecture (`3d8c57c`).
- Complete: Sherpa runtime/models (`bbbe251`).
- Complete: central VAD/offline runtime (`96b54e6`).
- Complete: web bridge dual-source transport (implemented and reviewed 2026-07-03).
- Preserve current uncommitted dataflow and prior-plan files during implementation.

## Phases

| # | Phase | Status | Progress | Effort | Detail |
|---|---|---|---:|---:|---|
| 1 | Web bridge dual-source transport | Complete | 100% | 10h | [Phase 01](./phase-01-web-bridge-dual-source-transport.md) |
| 2 | Source-aware command routing | Complete | 100% | 6h | [Phase 02](./phase-02-source-aware-command-routing.md) |
| 3 | Dual-source voice UI | Complete | 100% | 8h | [Phase 03](./phase-03-dual-source-voice-ui.md) |
| 4 | System validation gate | Complete | 100% | 7h | [Phase 04](./phase-04-system-validation-gate.md) |
| 5 | Legacy retirement and deployment | Complete | 100% | 6h | [Phase 05](./phase-05-legacy-retirement-and-deployment.md) |

## Dependencies

- [Research](./research/current-state-and-sherpa-report.md)
- [Reconciliation report](./reports/brainstorm-reconciliation-report.md)
- [Source brainstorm](../reports/brainstorm-260702-1750-central-sherpa-stt.md)
- [Superseding prior plan](../260702-2316-central-sherpa-vad-stt/plan.md)
- UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Official Sherpa models and pinned native build must remain available on Orchestra.

## Delivery Gates

- Phases 1 and 2 must pass before commands can be enabled from central STT.
- Phase 3 must deploy with matching Rust/TypeScript contracts.
- Phase 4 must record an explicit approval decision and any accepted follow-up backlog.
- Phase 5 must not remove rollback assets before Phase 4 passes.

## Architecture Gate

`ARCHITECTURE.md` already contains the intended final-only dual-source flow and target/privacy invariants from the completed contract phase. No new structural design introduced here.

## Current Milestone

Phases 01-05 are complete and approved. Phase 04 validation completed 2026-07-03 and was
approved on 2026-07-04 with accepted follow-up backlog: browser and rover STT quality are
not yet aligned, the rover soak lost 13 finalizations, representative bilingual accuracy
corpora were unavailable, and latency/resource measurements are still incomplete. Phase 05
completed and was approved on 2026-07-04 after legacy retirement and deployment cleanup.

## Unresolved Questions

- Root cause and remediation for the manual browser-versus-rover STT quality gap.
- Root cause and remediation for the Phase 04 rover finalization loss.
- Availability of labeled bilingual acoustic corpora and target-hardware benchmark capture.
