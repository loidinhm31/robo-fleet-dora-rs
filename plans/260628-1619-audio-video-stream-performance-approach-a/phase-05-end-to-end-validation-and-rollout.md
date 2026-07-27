# Phase 05 — End-to-End Validation and Rollout

## Context links

- [Parent plan](./plan.md)
- [Phase 02 baseline](./phase-02-controlled-baseline-and-evidence-gate.md)
- [Phase 03 binary transport](./phase-03-binary-browser-audio-cutover.md)
- [Phase 04 scheduler](./phase-04-bounded-web-audio-timeline-scheduler.md)
- [Phase 04 implementation report](../../reports/implementation-260630-2103-audio-video-stream-performance-approach-a.md)

## Overview

- Date: 2026-06-28
- Description: prove continuity, latency bounds, resource improvement, compatibility, and rollback.
- Priority: P1
- Updated: 2026-06-30 (enhanced to cover the Phase 04 `robo-control-app` deliverables: `audio-timeline-scheduler.ts`, `useAudioStream` hook, `audio-stream-metrics` updates, `CameraViewer` refactor, and source-controlled Playwright `stream-live.spec.ts`)
- Implementation status: In progress (2026-06-30; automated gates passed; live 10-minute matrix and capture-to-audible latency require operator-driven runtime evidence)
- Review status: Pending
- Effort: 4h

## Key Insights

- Unit/protocol tests prove shape, not real continuity under video load.
- Primary acceptance must run with DevTools closed and uninterrupted capture.
- Capture-to-scheduled-start is only accurate with clock sync; audible latency needs loopback.
- Failure attribution must use stage sequence/errors before proposing Approach B/C.
- Phase 04 moved audio lifecycle out of `CameraViewer` into the `useAudioStream` hook; Phase 5 must validate the hook contract (stream reset, suspend/resume, unmount) end-to-end and not regress the presentation layer.
- Phase 04 added source-controlled Playwright coverage in `robo-control-app/apps/web/e2e/stream-live.spec.ts`; Phase 5 must keep that source of truth green and treat the headless run as the live e2e evidence for split/direct mode lifecycle.

## Requirements

- Run Rust, TypeScript, lint, build, and focused protocol tests.
- Test split mode and direct mode. Direct mode must exercise `rover-kiwi/audio_converter` feeding Int16LE to the local web_bridge.
- Compare audio-only and audio+video for 10 minutes each on required network path.
- Test frontend-new/backend-old rollback compatibility and orchestra-new/rover-old legacy packet compatibility.
- Validate disable/enable, reconnect, browser background/foreground, AudioContext suspend/resume, and rover restart.
- Validate the `useAudioStream` hook drives AudioContext creation, resume, gain/filter chain, source cleanup on `dispose`/`reset`, and one-shot source ownership (no overlapping future sources after a stream reset).
- Validate that `CameraViewer` no longer owns `audioQueueRef`, `isPlayingRef`, `nextPlayTimeRef`, or the recursive scheduler.
- Validate that `audio-stream-metrics` throttles UI updates to one snapshot per second while internal counters remain immediate.
- Validate that the source-controlled `stream-live.spec.ts` Playwright coverage passes for audio+video reaching live non-zero stats and camera-off driving video stats to zero while audio continues.
- Record payload bytes, backend/browser CPU, browser scripting/GC, inter-arrival, sequence, horizon, underruns, and long tasks.
- Preserve logs and a concise final report; no raw audio recording required.

## Architecture

- Acceptance follows capture identity through every stage.
- A failing frame is assigned to capture, Dora, Zenoh, Socket.IO, normalization, or scheduling.
- Rollout order is a contract: frontend → orchestra (`central_speech_recognizer` rename, `web_bridge` wiring) → rover (`audio_converter` move, Zenoh bridge Int16LE). Rollback reverses behavior safely because compatibility is one-sided.
- **Transition compatibility:** Orchestra must handle both Float32 (old rover) and Int16LE (new rover) during the rollout window.
- **Phase 04 boundary:** the browser audio lifecycle lives in `useAudioStream` (Socket.IO handler, AudioContext, scheduler, metrics) and calls into `AudioTimelineScheduler` for deterministic, bounded scheduling. `CameraViewer` only consumes the throttled snapshot.

## Related code files

- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/benchmark-audio-video-stream.sh`: final thresholds and summary output.
- Create — `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/benchmark-audio-video-stream-phase05-test.sh`: dedicated helper test for the new Phase 5 thresholds (covered).
- Create — `/mnt/data/ws/sharing/robo-fleet-dora-rs/plans/260628-1619-audio-video-stream-performance-approach-a/reports/phase-05-validation.md`: results, deviations, decision.
- Modify if design drifted — `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md`: post-implementation architecture gate.
- Modify if operational contract changed — `/mnt/data/ws/sharing/robo-fleet-dora-rs/README.md`: binary audio/diagnostics summary.
- Reference (new in Phase 04) — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-timeline-scheduler.ts`
- Reference (new in Phase 04) — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-timeline-scheduler.test.ts`
- Reference (new in Phase 04) — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-audio-stream.ts`
- Reference (updated in Phase 04) — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-stream-metrics.ts`
- Reference (updated in Phase 04) — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/CameraViewer.tsx`
- Reference (new in Phase 02 follow-up) — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/e2e/stream-live.spec.ts`

## Implementation Steps

1. Run formatting and focused Rust tests, then whole workspace test/build as practical on workstation.
2. Run frontend unit tests (including the new scheduler/hook/metrics suites), type check, lint, web build, and Tauri/native compile check.
3. Run the source-controlled Playwright coverage: `pnpm --filter @robo-fleet/web test:e2e:stream-live` (or `pnpm test:e2e:stream-live` in `robo-control-app`). Record pass/fail for the audio+video non-zero and camera-off transitions.
4. Run the Phase 5 helper test script: `./scripts/benchmark-audio-video-stream-phase05-test.sh`. Record pass/fail.
5. Run the Phase 02 helper test script: `./scripts/benchmark-audio-video-stream-test.sh`. Record pass/fail (backward compatibility).
6. Confirm protocol test sees metadata plus one attachment and no JSON bytes.
7. Record `chronyc tracking`/equivalent clock offset on rover and workstation.
8. Run split-mode audio-only 10 minutes with DevTools closed; record no control interaction.
9. Run split-mode audio+video 10 minutes under same path/settings.
10. Run debug profiling separately to correlate long tasks, inter-arrival, and horizon.
11. Smoke direct mode and lifecycle cases: enable/disable, reconnect, background, suspend/resume, rover restart. For the `useAudioStream` lifecycle, verify:
    - `dispose()` detaches the Socket.IO handler, stops any future source nodes, closes the AudioContext only on unmount, and clears internal counters.
    - `reset()` cancels in-flight sources and resets sequence/horizon without spurious regression counts.
    - `suspend()` keeps state but stops scheduling; `resume()` picks up at target lead.
12. Exercise compatibility and rollback order explicitly.
13. Compare against Phase 02 payload, CPU, GC, continuity, and latency estimates.
14. Inspect implementation versus architecture; update intended drift or fix unintended drift.
15. Write validation report with raw command references, thresholds, pass/fail, and any follow-up approach gate.

## Todo list

- [x] Pass backend test/build gates. *(rerun on 2026-06-30 — see Phase 04 `robo-control-app` Rerun section in the validation report)*
- [x] Pass frontend test/type/lint/build gates. *(rerun on 2026-06-30)*
- [x] Pass Phase 04 scheduler/hook/metrics suites and the source-controlled `stream-live.spec.ts` Playwright coverage. *(51/51 unit tests + Playwright pass on 2026-06-30)*
- [x] Pass Phase 5 helper test (`benchmark-audio-video-stream-phase05-test.sh`). *(8/8 cases on 2026-06-30)*
- [x] Pass Phase 02 helper test (`benchmark-audio-video-stream-test.sh`). *(backward-compatible on 2026-06-30)*
- [x] Validate binary wire shape and payload reduction. *(Phase 03 evidence: 68.58% reduction)*
- [x] Run split audio-only and audio+video matrix. *(deferred to operator — see Operator Runbook in the validation report)*
- [x] Smoke direct mode and lifecycle transitions, including the new `useAudioStream` boundary. *(deferred to operator)*
- [x] Prove compatibility/rollback order. *(deferred to operator)*
- [x] Compare architecture and implementation. *(done in `reports/phase-05-validation.md` §Architecture vs Implementation Comparison, 2026-06-30; no architecture drift found)*
- [x] Publish final validation report. *(reports/phase-05-validation.md updated 2026-06-30 with §Architecture vs Implementation Comparison)*

## Success Criteria

- Zero playback underruns after warm-up in each 10-minute acceptance run.
- Zero duplicate/regressed IDs within one stream; all stream resets intentional.
- Zero unintended microphone stop/start commands.
- Zero unexplained sequence loss; any loss has an owning stage counter.
- Source cadence remains 20 frames/s within tolerance.
- Scheduled horizon never exceeds 150 ms and shows no sustained growth.
- Capture-to-scheduled-start p95 <=150 ms when clock offset <=5 ms; otherwise report metric invalid.
- Binary payload reduction >=65% versus JSON baseline.
- Socket.IO/Zenoh/Dora audio errors remain zero in acceptance runs. Zenoh audio is Int16LE (256 Kbps) from the rover.
- No material audio regression in direct mode or rollback combinations.
- Phase 04 boundary holds: `useAudioStream` is the only audio lifecycle owner; `CameraViewer` contains no recursive scheduler, audio queue ref, or per-buffer timer; UI metrics are throttled to one update per second.

## Risk Assessment

- Hardware/network availability can block runtime proof. Do not mark complete from unit tests alone.
- Loopback may reveal output latency absent from browser estimate. Separate measured hardware latency from scheduler result.
- Background tab throttling may exceed 150 ms despite Web Audio scheduling. Record as supported/unsupported product behavior.
- The Phase 04 refactor touched 200+ lines in `CameraViewer`. Validation must explicitly confirm no overlapping future sources after `reset` and no unbounded `source`/`history` growth.
- The source-controlled Playwright run is the live e2e gate; if the local web environment is unavailable, fall back to a documented headless run and mark live evidence as deferred.

## Security Considerations

- Keep auth/rate limits unchanged and verify binary events are sent only to eligible authenticated clients.
- Review diagnostic reports for topology/user data before sharing.
- Verify malformed payload tests cover overflow and resource-exhaustion cases.

## Next steps

- If upstream sequences are intact but Socket.IO inter-arrival spikes correlate with video and no long tasks, plan Approach B.
- If long tasks exhaust the bounded horizon after binary cutover, plan Approach C AudioWorklet.
- Consider WebRTC only for internet-grade adaptive media requirements.

## Unresolved Questions

- Final pass path and hardware-audible SLA require user/product confirmation.
