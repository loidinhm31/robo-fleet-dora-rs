# Phase 02 — Controlled Baseline and Evidence Gate

## Context links

- [Parent plan](./plan.md)
- [Phase 01](./phase-01-capture-identity-and-transport-observability.md)
- [Research measurement contract](./research/researcher-01-report.md#measurement-contract)

## Overview

- Date: 2026-06-28
- Description: capture the report's missing runtime evidence before changing transport/playback behavior.
- Priority: P1
- Implementation status: Done
- Review status: Approved
- Effort: 4h

## Key Insights

- Existing logs are confounded by explicit microphone stop/start commands.
- Browser receive timing, queue horizon, long tasks, and selected Engine.IO transport are absent.
- Per-frame logging itself can alter the result; diagnostics need bounded aggregation.

## Requirements

- New frontend understands origin metadata while retaining current JSON byte-array playback.
- Collect receive inter-arrival, sequence, age estimate, queue depth, scheduled horizon, underruns, and long tasks.
- React stats update at most once/second. Detailed logs emit at most once/five seconds and only with `?audioDebug=1`.
- Run audio-only and audio+video for 10 minutes each with no control interaction.
- Record browser/Orchestra/rover hosts, path, Engine.IO transport, DevTools state, and clock offset.
- Gate later phases on valid source cadence and absence of unintended control events.

## Architecture

- Add pure frame normalization and metrics collectors now; reuse them during binary/scheduler phases.
- Keep existing recursive scheduler untouched so baseline represents current behavior.
- Store bounded samples/percentiles, not an unbounded event history.
- **Note:** The `audio_converter` node now runs on the rover (F32→S16LE before Zenoh transport). Orchestra no longer has an `audio-converter` node. Baseline metrics must be captured from the rover-side `audio_converter`, not orchestra.

## Related code files

- Create — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-frame.ts`: validate metadata and normalize legacy bytes.
- Create — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-stream-metrics.ts`: bounded metrics accumulator/snapshot.
- Create — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-frame.test.ts`: validation/compatibility tests.
- Create — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-stream-metrics.test.ts`: percentile/reset tests.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts`: origin fields and transitional event union.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/CameraViewer.tsx`: attach non-behavioral metrics; preserve dirty video changes.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/package.json`: focused Vitest script/dependency.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/pnpm-lock.yaml`: test dependency lock.
- Create — `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/benchmark-audio-video-stream.sh`: run metadata/log checks and collate results. Must also collect rover-side `audio_converter` conversion duration and error metrics.
- Create — `/mnt/data/ws/sharing/robo-fleet-dora-rs/plans/260628-1619-audio-video-stream-performance-approach-a/reports/phase-02-baseline.md`: environment, results, gate decision.

## Implementation Steps

1. Define transitional `AudioFrameMetadata`/legacy JSON types with `stream_id`, origin timestamp, sequence, format, rate, channels, and sample count.
2. Implement pure metadata/payload validation. Reject unsupported format/channels/rate before sample conversion.
3. Build fixed-capacity metrics windows for inter-arrival and age. Track gaps, duplicates/regressions, invalid frames, underruns, and long tasks.
4. Add development-only `PerformanceObserver` for `longtask` when supported; record unsupported state explicitly.
5. Integrate observation around the current handler without changing queue threshold, timer, or playback policy.
6. Remove per-frame/random console logging; publish structured five-second summary only in debug mode.
7. Add Vitest locally to `@robo-fleet/ui`; test pure collectors without DOM/audio hardware.
8. Add benchmark helper checks for 20 Hz cadence, origin continuity, transport, stop/start commands, and backend errors.
9. Synchronize clocks, close DevTools, hold capture active, run audio-only 10 minutes, then audio+video 10 minutes.
10. Run one debug/profiling reproduction separately. Save backend summaries and browser metrics in the phase report.
11. Mark gate pass/fail with concrete evidence. Stop if control-plane interruptions or source gaps invalidate the run.

## Todo list

- [x] Add transitional frame types and validator.
- [x] Add bounded browser metrics.
- [x] Throttle UI/log updates.
- [x] Configure focused frontend unit tests.
- [x] Add benchmark helper.
- [x] Run controlled matrix.
- [x] Write baseline report and gate decision.

## Success Criteria

- Two valid 10-minute runs with capture continuously active.
- Source cadence averages 20 frames/s within documented tolerance.
- Origin sequence identifies exact loss boundary or confirms no upstream loss.
- Existing ~1 s depletion cadence is confirmed or disproved with queue/horizon data.
- Audio+video inter-arrival and long-task results are directly comparable to audio-only.
- `pnpm --filter @robo-fleet/ui test`, `pnpm check-types`, and `pnpm lint` pass.

## Risk Assessment

- Instrumentation can perturb timing. Keep aggregate-only default and separate debug run.
- Long Task API support varies. Record capability; do not fail validation solely when unavailable.
- Dirty CameraViewer changes can be overwritten. Patch only audio sections and inspect diff before/after.

## Security Considerations

- Debug output contains timestamps/host topology; do not expose it to unauthenticated UI or production logs by default.
- Bound diagnostic history to avoid memory growth.

## Next steps

- Gate pass: proceed to binary cutover.
- Gate fail from control events/source drops: diagnose those first and revise plan.
- Evidence of last-hop HoL remains informational; do not add Approach B yet.

## Approval note

- Phase approved on 2026-06-30 using a reduced runtime gate accepted by the user.
- Runtime evidence used 2-minute audio-only and 2-minute audio+video live runs on the temp-DB-auth rover flow instead of the original 10-minute matrix.
- Follow-up work also fixed the stale camera video-fps/bitrate UI caveat and moved the live browser verification into source-controlled `robo-control-app` Playwright coverage.

## Unresolved Questions

- Required network path and SLA interpretation must be recorded in the phase report.
