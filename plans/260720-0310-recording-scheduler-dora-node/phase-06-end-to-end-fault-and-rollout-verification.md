# Phase 06 — End-to-End, Fault, and Rollout Verification

## Context links

- [Parent plan](./plan.md)
- [Phase 02](./phase-02-scheduler-core-recurrence-and-persistence.md)
- [Phase 03](./phase-03-web-bridge-coordinator-and-recorder-reconciliation.md)
- [Phase 04](./phase-04-dora-container-and-operational-integration.md)
- [Phase 05](./phase-05-scheduler-ui-and-client-state.md)
- Existing workflow test: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/tests/recording-workflow.rs`

## Overview

- Date: 2026-07-20
- Description: Prove time, persistence, ownership, crash recovery, MP4 output, UI convergence, security, and rollback before release.
- Priority: P1 release gate
- Implementation status: Pending
- Review status: Pending final code/operations review
- Effort: 7h

## Key Insights

- Happy-path CRUD does not prove no duplicate clips or unrelated OFF transitions.
- Crash points around persist/emit/ack need a matrix, not one restart test.
- Workstation Phase 10 is amd64 Docker/Podman verification, not ARM acceptance.
- Physical rover results must be reported separately from workstation evidence.

## Requirements

- Unit, property, Mongo, coordinator, recorder, Socket.IO, UI, Compose, and live MP4 gates.
- Verify one-time/daily/weekly and DST gap/fold with fake clock.
- Verify overlap union, manual/browser survival, two-rover isolation, limits.
- Verify offline rover/recorder/Mongo, disk/encoder failure, and all three process restarts.
- Verify no blind start during reconciliation or duplicate logical replay.
- Verify UI convergence after reconnect, conflict, suppression, retry, missed, failed.
- Verify manual start suppresses/finalizes the scheduled group before starting the requested manual session.
- Verify all logged-in users can query/mutate/delete, logged-out users cannot, and no scheduler RBAC branch exists.
- Verify terminal-only 90-day TTL, exact 1/2/4/8/16/30-second retry cadence until window end, and no alert delivery in v1.
- Verify recorder crash keeps failed/partial/recovered clip attempts on one logical occurrence.
- Verify missing scheduler fails health while Mongo/reconciliation degradation preserves the manual path.
- Record evidence/residual risk; do not claim Raspberry Pi acceptance from amd64.

## Architecture

- Layers: pure contracts/time; real Mongo/CAS; coordinator fake recorder; FFmpeg workflow; Dora/Socket.IO; UI harness; full workstation Compose.
- Crash matrix: before persist, after persist-before emit, after emit-before ack, after ack-before save, during stop/finalize.
- Assert invariants from status/logs/MP4/catalog, not command admission alone.

## Related code files

- Extend `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/tests/recurrence.rs` — timezone/property suite.
- Extend `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/recording_scheduler/tests/mongo-recovery.rs` — CAS/restart/outbox.
- Extend `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/tests/recording-scheduler-coordinator.rs` — ownership/fault matrix.
- Extend `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/tests/recording-workflow.rs` — snapshot/adoption/partial recovery.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/tests/recording-scheduler-dataflow.rs` — Dora end-to-end harness or existing equivalent.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/plans/260720-0310-recording-scheduler-dora-node/reports/phase-06-verification-report.md` — evidence.
- Extend `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/e2e/recording-scheduler.spec.ts` — browser release gate.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docs/codebase-summary.md` — only after implementation evidence.

## Implementation Steps

1. Run contract/fake-clock suites with DST/month/year/leap cases.
2. Run Mongo indexes, CAS, duplicate materialization, exact 90-day terminal TTL, restart.
3. Run ownership tests: zero/one transitions, overlap, browser/manual holds, two entities, reordered feedback.
4. Execute crash matrix and assert adoption/no blind duplicate.
5. Run FFmpeg workflow; assert non-empty MP4, H.264/AAC, manifest, and multiple attempt references on one recovered occurrence.
6. Run Socket.IO login/rate/queue/snapshot tests; confirm uniform CRUD/delete for authenticated users and denial when logged out.
7. Run linked UI type/lint/build/Vitest/Playwright/a11y.
8. Export `XDG_RUNTIME_DIR=/run/user/$(id -u)`; run `docker info` and a real smoke container.
9. Bring up workstation amd64 Mongo/Orchestra/rover; inspect health/process/logs.
10. Run short one-time/overlap windows; verify output and last-owner stop.
11. Inject rover/recorder/Mongo/process failures; verify exact retry cadence, window-end cutoff, degraded scheduler status, and process-liveness health split.
12. Disable feature; verify manual recording/list/playback/delete; document rollback.
13. Write report with commands, versions, durations, evidence, and hardware gaps.

## Todo list

- [ ] Rust/core/coordinator/recorder tests green.
- [ ] UI type/lint/build/unit/E2E/a11y green.
- [ ] Compose and smoke green.
- [ ] Time/DST deterministic.
- [ ] Overlap one session for union.
- [ ] Scheduled stop preserves unrelated demand.
- [ ] Crash matrix no blind duplicate.
- [ ] Retry bounded; missed/failed visible.
- [ ] Manual replacement finalizes scheduled clip before manual start.
- [ ] Authenticated CRUD/delete has no role split.
- [ ] One occurrence retains all crash/recovery clip attempts.
- [ ] Missing-process vs dependency-degraded health behavior proven.
- [ ] Two rovers isolated.
- [ ] Feature-disable rollback preserves manual path.
- [ ] Workstation vs ARM labeled.

## Success Criteria

- Every brainstorm acceptance criterion maps to report evidence.
- Scheduled window yields playable MP4 and completed occurrence.
- Overlap stops only at union end.
- Browser/manual demand remains after scheduled release.
- Restart after start-before-ack adopts/waits safely; never blindly duplicates.
- Transient retry follows 1/2/4/8/16/30 seconds without attempt cap and stops at window end; no alert service is introduced.
- Terminal history expires at 90 days, while nonterminal state never receives TTL.
- No busy loop, unbounded queue/map, cross-rover event, secret/path leak, stale UI.
- Rollback preserves schedules/history/manual recording.

## Risk Assessment

- Live timing: fake clock for correctness; live tests use tolerances.
- Fault injection: dedicated test DB/recording root and explicit containers.
- Podman confusion: runtime dir plus real smoke before conclusion.
- Clip-attempt list growth: prove it remains bounded by the finite window/retry policy while preserving every actual attempt.

## Security Considerations

- Test stale JWT, logged-out denial, authenticated CRUD/delete, actor spoof, cross-entity query, unsafe path, oversized payload, and rate limits.
- Use non-secret test credentials; redact tokens/URIs.
- Verify private/non-root Mongo/recording mounts and ticketed playback.
- Never target broad directories or production volumes.

## Next steps

- Obtain code review and operations sign-off.
- Roll out disabled, enable one workstation/rover, observe one cycle, broaden.
- Schedule separate ARM/physical acceptance if required.

## Unresolved questions

1. Canary observation period/count before general enablement?
2. Is ARM/physical acceptance this release or follow-up?
