# Phase 05 — End-to-End Verification

## Context Links

- [Parent plan](./plan.md)
- [Walkie transport](./phase-02-walkie-ingress-and-transport.md)
- [TTS pacing](./phase-03-tts-pacing-and-lifecycle.md)
- [Playback and suppression](./phase-04-playback-suppression-and-observability.md)
- Docker stack: `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/docker-compose.yml`

## Overview

- Date: 2026-07-06
- Priority: P1
- Description: Prove deterministic media behavior; code verification is complete, but manual hardware acceptance produced bad results and needs follow-up research.
- Implementation status: Code-level verification complete
- Review status: Approved with known hardware failure pending research
- Completion: partial

## Key Insights

- Existing package tests and local verification cover contracts, pacing, queue behavior, and suppression, but not audibility or loopback.
- Synthetic faster-than-real-time input already demonstrated the old failure and paced path recovery in code-level tests.
- Workstation `linux/amd64` Docker verification is the current target; it is not ARM acceptance.
- Manual speaker/microphone acceptance was attempted on 2026-07-06 and the result was bad enough to defer closure pending more research.

## Requirements

- Cover contract, pacing, capacity, resampling, sequence, suppression, preemption, and terminal ordering.
- Exercise direct and orchestra/Zenoh dataflows.
- Measure one-way walkie latency and acoustic loopback with real speaker/microphone hardware.
- Preserve evidence: commands, structured counter snapshots, timings, and recordings.

## Architecture

Verification layers:

```text
Type/golden tests
  -> deterministic fake-clock and buffer tests
  -> Dora synthetic dataflow test
  -> UI/backend Socket.IO integration
  -> amd64 Docker full stack
  -> physical speaker/microphone acceptance pending
```

Automated layers passed. Manual hardware acceptance was attempted but failed qualitatively, so emitted/consumed/suppressed counters plus recordings still need follow-up research before closure.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/tests/dataflow-queue-policy.rs` — parse/assert YAML policies.
- Add integration fixtures/tests within `edge_voice`, `audio_playback`, `audio_capture`, and both Zenoh bridge crates.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-walkie-capture.test.ts` — actual-rate/binary/session tests.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/e2e/edge-voice-live.spec.ts` — complete long TTS and preemption evidence.
- Extend an existing audio verification script under `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/` rather than creating a duplicate harness.

## Implementation Steps

1. Assert explicit queue sizes for relevant inputs in remote rover, direct rover, and orchestra YAML.
2. Add fake-clock pacer cases: exact ordering, no early/burst sends, scheduler delay, final partial chunk, cancellation, and completion ordering.
3. Simulate 10-, 30-, and 60-second TTS streams through resampling and buffer consumption; compare generated, resampled, enqueued, retired, and consumed samples with documented resampler tolerance.
4. Test declared 16, 44.1, and 48 kHz walkie sine/voice fixtures for duration, RMS continuity, output count, and sequence propagation.
5. Force repeated callback underruns and prove interval coalescing yields `Active`, later `Idle`, and 400 ms capture suppression.
6. Test TTS preemption before synthesis, while pacing, while buffered, and while awaiting final consumption; each produces exactly one terminal result.
7. Run a Dora synthetic test first with an unpaced numbered burst to reproduce gaps, then with pacing to prove zero gaps.
8. Run backend package tests, workspace tests/build, and UI tests/type-check/lint/build.
9. Start the amd64 Docker stack with the documented `XDG_RUNTIME_DIR`; verify container health, processes, and periodic audio counters.
10. Record live walkie at representative speech level and TTS utterances of at least 10, 30, and 60 seconds. Measure first-capture-to-first-speaker latency and verify complete playback.
11. Keep rover microphone publication enabled during speaker tests; require suppression counters to show zero published frames during playback plus 400 ms.
12. Save acceptance evidence in the plan reports directory and obtain code review before marking the plan complete.

## Todo list

- [x] Dataflow policy test passing
- [x] Pacer tests passing
- [x] Long TTS simulations passing
- [x] Multi-rate walkie tests passing
- [x] Suppression/preemption tests passing
- [x] Direct and Zenoh paths passing
- [x] UI validation passing
- [x] Docker stack healthy
- [ ] Hardware recordings and counters accepted
- [x] Final code review completed

## Success Criteria

- Nominal walkie and TTS gaps: zero.
- Ordinary TTS capacity drops: zero.
- Generated/resampled/consumed TTS totals agree within resampler tolerance.
- Exactly one terminal result follows final consumption or explicit interruption/failure.
- Walkie one-way playback latency: at most 250 ms.
- Echo leakage: zero rover microphone frames during playback and 400 ms tail.
- Walkie speech is intelligible; 10/30/60-second TTS plays once, completely, and in order.

## Risk Assessment

- Acoustic results depend on device routing and room coupling. Record device names, levels, environment, and raw evidence for reproducibility.
- Container health does not prove Dora media flow. Use structured application counters and recordings as the source of truth.

## Security Considerations

- Do not store credentials, JWTs, environment secrets, or private speech recordings in Git.
- Sanitize logs before attaching acceptance evidence.
- Keep test Socket.IO clients authenticated and rate-limited like production clients.

## Next steps

After follow-up research fixes the bad manual result and the hardware gate passes, update plan statuses, reconcile architecture with implementation, and prepare one focused bug-fix commit per repository.
