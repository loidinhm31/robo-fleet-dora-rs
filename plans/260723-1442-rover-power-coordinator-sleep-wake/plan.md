---
title: "Rover Power Coordinator Sleep/Wake"
description: "Add workload-level Awake/Auto/Sleep orchestration, scheduled prewarm, local voice wake, and durable power history."
status: in-progress
priority: P1
effort: 272h
branch: main
tags: [feature, backend, frontend, database, infra, critical]
created: 2026-07-24
---

# Rover Power Coordinator Sleep/Wake

## Overview

Add one policy layer above lifecycle managers. Scheduler, UI, media, and KWS produce bounded demands; coordinators choose static profiles and lifecycle managers execute exact-node transitions. V1 never suspends hosts, stops containers, or interprets voice as an actuator command.

## Locked decisions

- Policy: `Awake | Auto | Sleep`; policy and effective state stay separate.
- Fresh restart: discard transient demands/policy, increment durable authority epoch, boot `Awake`.
- `Auto`: demand-first, fresh per-domain low CPU only confirms five demand-free minutes.
- `Sleep`: scheduled reservation may prewarm; UI Wake changes policy to `Auto`.
- Voice: continuous Rover KWS v1; prerecorded `WakeAck`; no local general STT/NLU.
- Live coordinator status is current-state authority; Mongo supplies 90-day history/cold projection.
- Scheduler owns occurrence truth; coordinator owns power; lifecycle manager owns exact-target execution.

## Phases

| # | Phase | Status | Progress | Effort | Dependency |
|---|---|---|---:|---:|---|
| 1 | [Contracts and lifecycle hardening](./phase-01-contracts-and-lifecycle-hardening.md) | Complete | 100% | 32h | — |
| 2 | [Coordinator core, profiles, and Auto](./phase-02-coordinator-core-profiles-and-auto.md) | Complete | 100% | 48h | 1 |
| 3 | [Local journal and Mongo projection](./phase-03-local-journal-and-mongo-projection.md) | Pending | 0% | 40h | 1–2 |
| 4 | [Authority, Zenoh, and direct routing](./phase-04-authority-zenoh-and-direct-routing.md) | Pending | 0% | 36h | 1–3 |
| 5 | [Scheduler reservations and measured prewarm](./phase-05-scheduler-reservations-and-measured-prewarm.md) | Pending | 0% | 32h | 1–4 |
| 6 | [Rover continuous KWS and WakeAck](./phase-06-rover-continuous-kws-and-wake-ack.md) | Pending | 0% | 36h | 1–4 |
| 7 | [Authenticated API and external power UI](./phase-07-authenticated-api-and-external-power-ui.md) | Pending | 0% | 28h | 1–5 |
| 8 | [Fault validation, target benchmarks, and rollout](./phase-08-fault-validation-target-benchmarks-and-rollout.md) | Pending | 0% | 20h | 1–7 |

## Release gates

- `WakeAck <1.5 s p95`; `NormalRover Ready <5 s p95`; scheduler lead = measured profile p95 + margin.
- Duplicate, stale, reordered, expired, or cross-entity inputs never regress policy/effective state.
- Every applied transition follows a synced local journal intent; Mongo outage never blocks local safety/wake.
- Docker workstation, direct mode, split mode, restart, partition, and physical Rover acceptance pass.

## Validation Summary

**Validated:** 2026-07-26
**Questions asked:** 8

### Confirmed Decisions
- Auto rests in `IdleListening`; explicit Sleep reaches `Dormant`.
- UI activity demand expires two minutes after the last interaction.
- Every authenticated user may set persistent Awake/Sleep, with exact-entity validation, rate limits, and audit identity.
- Orchestra remains Reconciling and waits for a fresh Rover snapshot; it never forces timed authority takeover.
- V1 detects one input phrase, `Hey Kiwi`; prerecorded `I am on` remains a separate output-only WakeAck.
- Schedule edit/delete/supersession releases immediately; still-valid occurrences retry transient recorder/storage failures only inside a bounded start window.

### Action Items
- [ ] Reflect these decisions in each remaining Phase 4, 5, 6, and 7 before implementation.
- [ ] Derive KWS error limits, Auto thresholds, and timeout/capacity values from physical-target evidence in Phase 8.

## Out of scope

OS suspend/Wake-on-LAN, GPIO wake, process kill, runtime profile learning, general offline STT/NLU, Grafana integration, raw-history retention beyond 90 days.

## Unresolved questions

- Exact checksum-pinned KWS model and acceptable noisy-hour false-accept/false-reject limits for `Hey Kiwi`.
- Target-derived domain CPU thresholds, consecutive sample count, minimum awake hold, and bootstrap prewarm values.
- Snapshot retry/staleness thresholds and local journal byte budget/reserved wake capacity.
