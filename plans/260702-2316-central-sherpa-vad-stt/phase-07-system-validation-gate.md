# Phase 07 — System Validation Gate

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phases 01-06](./plan.md#phases)
- Orchestra dataflow: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`
- UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-02 |
| Description | Prove language accuracy, isolation, routing, privacy, latency, and stability before retirement. |
| Priority | P1 |
| Implementation status | Pending |
| Review status | Pending |
| Effort | 7h |

## Key Insights

- Model-backed tests are too large for default CI and need an explicit local/live suite.
- Unit success is insufficient: routing bugs appear only with concurrent browser/rover sources.
- Whisper must remain recoverable until both startup profiles pass.
- VAD accuracy must be tested with silence/noise, not only clean complete WAVs.

## Requirements

- Run deterministic unit/contract tests without model downloads.
- Run explicit model-backed English and Vietnamese suites on Orchestra.
- Exercise two rover streams and one browser stream concurrently.
- Measure RTF, endpoint-to-final latency, CPU, RSS, queue drops, and validation/reset counters.
- Verify command target and browser transcript privacy end to end.

## Architecture

Validation layers:

1. Pure units: conversion, metadata, sequence, contracts, UI state.
2. Model integration: VAD segmentation and Zipformer output.
3. Dora replay: dual-source input/output and status.
4. Live split: rover, Orchestra, browser, Zenoh, command routing.
5. Performance soak: bounded queues and resource stability.

## Related Code Files

- Add tests within each modified Rust crate using existing test conventions.
- Add UI tests under `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/`.
- Add optional model/live test scripts under `/mnt/data/ws/sharing/robo-fleet-dora-rs/plans/260702-2316-central-sherpa-vad-stt/reports/` only as completion evidence, not production tooling.
- Do not remove `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/.cache/ggml/ggml-base.bin` yet.

## Implementation Steps

1. Run Rust contract/unit tests for `robo_rover_lib`, central recognizer, web bridge, command parser, and Orchestra bridge.
2. Cover S16LE extrema, odd bytes, payload mismatch, wrong format/rate/channels, non-finite browser values, duplicate/regressed/gapped frames, stream replacement, VAD remainder, stop flush, queue full, and status error.
3. Run UI Vitest suite, type checks, lint, and production build.
4. Build central release binary with static Sherpa linkage and verify no unexpected shared Sherpa dependency with `ldd`.
5. Start Orchestra with `en-vad-offline`; replay labeled English commands plus silence, background noise, clipped audio, and short speech. Record normalized expected text/intent, RTF, and latency.
6. Restart with `vi-vad-offline`; run equivalent Vietnamese corpus and record results.
7. Replay original F32 corpus through S16LE conversion/transport and verify equivalent normalized parser outcomes where both are intelligible.
8. Inject two distinct rover streams concurrently with one browser stream. Assert transcript text, stream IDs, entity IDs, histories, and VAD resets never cross.
9. Change fleet selection mid-browser utterance. Assert command stays on captured target.
10. Speak on rover A while rover B is selected. Assert all parser command topics target A.
11. Connect two authenticated browsers. Assert browser A transcription never reaches B; rover transcriptions reach both.
12. Stop browser capture during speech. Assert central flushes and returns at most one final utterance.
13. Force missing model/profile error and reconnect UI. Assert stable `error` status and no node crash loop.
14. Run 10-minute one-rover-plus-browser soak. Require zero decode queue drops, bounded RSS, no growing stream maps, and no leaked browser audio resources.
15. Measure each profile: RTF below 1.0 and P95 endpoint-to-final latency at or below 2 seconds on target Orchestra hardware.
16. Produce a completion report with commands, hardware, model names, measurements, failures, and pass/fail decision.
17. Do not authorize Phase 08 retirement if either profile, routing, privacy, or performance gate fails.

## Todo List

- [ ] Run Rust units/contracts.
- [ ] Run UI tests/type/lint/build.
- [ ] Verify static linkage.
- [ ] Validate English profile.
- [ ] Validate Vietnamese profile.
- [ ] Compare F32/S16LE outcomes.
- [ ] Run concurrent isolation test.
- [ ] Verify target routing.
- [ ] Verify browser privacy.
- [ ] Run error/reconnect test.
- [ ] Run 10-minute soak and benchmark.
- [ ] Write completion report and gate decision.

## Success Criteria

- All automated checks pass without ignored failures.
- Both profiles produce usable labeled-corpus results and RTF below 1.0.
- P95 endpoint-to-final latency is at most 2 seconds.
- No source, sample, transcript, history, or command target mixes.
- Browser transcript privacy is enforced server-side.
- Ten-minute soak has zero decode drops and no unbounded memory/state growth.
- Completion report explicitly approves or blocks retirement.

## Risk Assessment

- Risk: Corpus is too clean and hides VAD false positives. Mitigation: include noise, silence, clipping, and short/non-command speech.
- Risk: Hardware-dependent timing creates flaky thresholds. Mitigation: record exact target hardware and use P95 across repeated samples.
- Risk: Vietnamese parser vocabulary is English-centric. Mitigation: separate STT accuracy from parser intent coverage and report parser gaps without blocking STT model validation unless required commands fail.

## Security Considerations

- Use non-sensitive recorded speech and do not commit user recordings.
- Validate privacy with two real authenticated sessions.
- Ensure logs avoid raw credentials, JWTs, and audio contents.

## Next Steps

Only a passing report permits Phase 08 Whisper/edge retirement. Failed gates return to the owning phase with recorded evidence.
