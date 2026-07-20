---
title: "Recording Scheduler Dora Node"
description: "Add durable one-time, daily, and weekly Orchestra recording schedules without creating a second media-control authority."
status: in_progress
priority: P2
effort: 58h
branch: main
tags: [feature, backend, frontend, database, infra]
created: 2026-07-20
---

# Recording Scheduler Dora Node

## Overview

Add `orchestra/recording_scheduler` as durable schedule/occurrence authority. Keep `web_bridge` as sole media-demand and recorder-session coordinator. Reuse `media_recorder`; no rover encoder or direct scheduler-to-rover controls.

## Phases

| # | Phase | Status | Progress | Effort | Link |
|---|---|---|---:|---:|---|
| 1 | Contracts and decision freeze | Done | 100% | 7h | [phase-01](./phase-01-contracts-and-decision-freeze.md) |
| 2 | Scheduler core, recurrence, persistence | Done | 100% | 14h | [phase-02](./phase-02-scheduler-core-recurrence-and-persistence.md) |
| 3 | Web coordinator and recorder reconciliation | Done | 100% | 14h | [phase-03](./phase-03-web-bridge-coordinator-and-recorder-reconciliation.md) |
| 4 | Dora, container, operations integration | Pending | 0% | 7h | [phase-04](./phase-04-dora-container-and-operational-integration.md) |
| 5 | Scheduler UI and client state | Pending | 0% | 9h | [phase-05](./phase-05-scheduler-ui-and-client-state.md) |
| 6 | End-to-end, fault, rollout verification | Pending | 0% | 7h | [phase-06](./phase-06-end-to-end-fault-and-rollout-verification.md) |

## Dependencies

- Phase 1 blocks all implementation.
- Phase 2 depends on frozen contracts; Phase 3 depends on Phases 1-2.
- Phase 4 depends on a buildable scheduler/coordinator path.
- Phase 5 may start after Phase 1 fixtures, but release waits for Phase 3 snapshots.
- Phase 6 requires Phases 1-5 complete.
- Phase 2/CI follow-up: validate canonical Rust/TypeScript fixture discovery paths and JSON fixture type assumptions in an automated cross-package check.

## Completion Log

- **2026-07-20 13:40 +07 (UTC+0700):** Phase 1 approved and complete. Version-1 Rust and TypeScript contracts, validation, deterministic IDs, and canonical fixtures are in place; the review's fixture path/type validation follow-up is assigned to Phase 2/CI.
- **2026-07-20 17:00 +07 (UTC+0700):** Phase 2 hardening approved and complete. Crash-safe occurrence/group/outbox transitions, deterministic bridging-overlap directory selection, durable outbox replay/acknowledgement, and superseding schedule mutation handling are verified by targeted fault, overlap, replay, and update/delete-race tests; package tests, live standalone Mongo verification, Clippy, and code review passed. Follow up in later phases on reconciliation/order integration and production Mongo persistence operations.

## Frozen Architecture

- Scheduler: durable single writer for schedules, occurrences, retries, suppression, reconciliation.
- Web bridge: only media-demand and recorder-session authority.
- Overlap: one per-rover group/session; earliest occurrence then ID selects directory.
- Time: IANA local intent; Unix-ms occurrences; gap shifts forward; fold chooses earlier.
- Identity: deterministic occurrence/start request IDs; recorder-generated `recording_id` stays random.
- Restart: reconcile durable intent with explicit recorder snapshot before replay.
- Manual stop suppresses to next boundary; manual start finalizes/replaces the current scheduled group.

## Evidence

- [Brainstorm](../reports/brainstorm-260720-0233-recording-scheduler-dora-node.md)
- [Backend research](./research/researcher-01-backend-scheduler-report.md)
- [Integration/UI/deployment research](./research/researcher-02-integration-ui-deployment-report.md)
- [Architecture](../../ARCHITECTURE.md)

## Validation Summary

**Validated:** 2026-07-20  
**Questions asked:** 6

### Confirmed Decisions

- Manual start replaces the scheduled group: suppress current owners, finalize, then start manual recording.
- Any authenticated user may view and mutate schedules, including delete; no scheduler RBAC in v1.
- Keep every terminal occurrence/audit record for 90 days using terminal-only TTL.
- Retry transient failures after 1, 2, 4, 8, 16, then every 30 seconds until window end.
- Retry alerting is future scope and not part of this plan.
- One logical occurrence may reference multiple failed/partial/recovered clip attempts.
- Missing scheduler process fails health; Mongo/reconciliation outage degrades scheduling only, preserving manual control.

### Action Items

- [x] Propagate decisions through all six phase specifications.
- [x] Update architecture, collision, persistence, retry, and health acceptance rules.

## Unresolved Questions

1. Production Mongo topology, credential owner, backup, and restore policy?
2. Is Raspberry Pi/ARM physical acceptance part of this release or a follow-up gate?
