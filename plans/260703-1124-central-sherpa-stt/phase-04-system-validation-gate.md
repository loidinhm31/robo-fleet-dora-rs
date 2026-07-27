# Phase 04 — System Validation Gate

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phases 01–03](./plan.md#phases)
- Orchestra dataflow: `orchestra/orchestra-dataflow.yml`
- Prior baseline: `plans/260702-2316-central-sherpa-vad-stt/reports/phase-01-baseline.md`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-03 |
| Description | Prove accuracy, isolation, privacy, routing, latency, and bounded operation before legacy removal. |
| Priority | P1 |
| Implementation status | Complete (approved with follow-up backlog) |
| Review status | Approved 2026-07-04 |
| Effort | 7h |

## Key Insights

- Unit tests cannot prove Zenoh/Dora routing, browser privacy, or concurrent source isolation.
- Both language profiles require real model and corpus validation on target Orchestra hardware.
- Phase approval can accept known operational gaps if they are recorded as explicit follow-up backlog.

## Requirements

- Run deterministic tests without models and explicit model/live suites with models.
- Exercise two rover streams plus one browser stream concurrently.
- Measure RTF, endpoint-to-final latency, CPU, RSS, queue drops, resets, and state cleanup.
- Prove browser privacy and command target safety end to end.
- Produce a reproducible completion report with pass/fail per gate.

## Architecture

```text
units/contracts -> model integration -> Dora replay -> split live system -> soak/benchmark -> retirement decision
```

## Related Code Files

- Modify tests beside affected Rust modules under `/mnt/data/ws/sharing/robo-fleet-dora-rs/{robo_rover_lib,common,orchestra}/`.
- Modify UI tests under `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/plans/260703-1124-central-sherpa-stt/reports/system-validation-report.md`: commands, hardware, metrics, failures, decision.
- Keep large/private audio fixtures outside Git.

## Implementation Steps

1. Run Rust unit/contract tests for shared types, central recognizer, web bridge, parser, and Orchestra bridge.
2. Cover S16LE boundaries, malformed metadata, non-finite browser samples, sequence faults, VAD flush, queue full, ownership, privacy, and target rejection.
3. Run UI tests, type-check, lint, and production build.
4. Build release central binary; inspect static Sherpa linkage and runtime startup.
5. Replay labeled English commands with silence, noise, clipping, and short speech under `en-vad-offline`.
6. Repeat equivalent Vietnamese corpus under `vi-vad-offline`; separate ASR accuracy from parser language coverage.
7. Compare equivalent F32 and transported S16LE inputs by normalized transcription/intent outcomes.
8. Replay two distinct rover streams and one browser stream concurrently; assert samples, identities, histories, and utterances never cross.
9. Change selected rover during browser speech; verify captured target remains fixed.
10. Speak on rover A while rover B is selected; verify command publishes only to A.
11. Connect two authenticated browsers; verify browser transcript privacy and rover transcript broadcast.
12. Force stop during speech, model missing, malformed frame, decode saturation, and reconnect paths.
13. Run a 10-minute rover-plus-browser soak. Confirm bounded RSS/registries, zero unexpected drops, and no browser resource leaks.
14. Require RTF below 1.0 and P95 endpoint-to-final latency at or below 2 seconds per profile on target hardware.
15. Record exact hardware, model names, commands, metrics, failures, and pass/fail in the validation report.
16. Record an explicit phase decision and convert accepted gaps into backlog items before Phase 05.

## Todo List

- [x] Run Rust units/contracts.
- [x] Run UI checks/build.
- [x] Verify release linkage/startup.
- [x] Validate available English fixture; record missing representative corpus.
- [x] Validate available Vietnamese fixture; record missing labels and representative corpus.
- [x] Compare F32/S16LE outcomes.
- [x] Run concurrent source isolation.
- [x] Prove target routing and browser privacy.
- [x] Run failure/reconnect cases.
- [x] Run soak and benchmark; record failed finalization and incomplete metric gates.
- [x] Write explicit phase decision and backlog follow-up.

## Success Criteria

- All automated checks pass without ignored failures.
- Both profiles meet recorded accuracy, RTF, and latency gates.
- No source, transcript, history, or command target mixing occurs.
- Browser transcript privacy is enforced server-side.
- Soak shows bounded state/memory and no unexplained queue drops.
- Validation report explicitly records the phase decision and any accepted follow-up backlog.

## Risk Assessment

- Clean corpus can hide VAD errors. Include representative noise and silence.
- Timing varies by host. Record hardware and use repeated P95 measurements.
- Vietnamese parser coverage may lag ASR. Report separately; do not misdiagnose parser gaps as recognizer failure.

## Security Considerations

- Use non-sensitive speech fixtures and do not commit user recordings.
- Test privacy with two authenticated sessions.
- Keep credentials/JWTs and raw audio out of logs/reports.

## Next Steps

Validation completed 2026-07-03 and was approved on 2026-07-04 with an accepted follow-up
backlog. Phase 05 may proceed later, but the STT quality and measurement gaps captured in
[system-validation-report.md](./reports/system-validation-report.md) remain open engineering debt.

## Unresolved Questions

- Root cause of the observed STT quality gap between browser voice commands and rover fleet transcription.
- Root cause of 13 missing rover finals and 115 sequence resets during the concurrent soak.
- Labeled Vietnamese and representative acoustic command corpora.
- Profile-specific endpoint-to-final latency and clean CPU/RSS time series.
