# Phase 08 — Native x86 Integration and Benchmark

## Context Links

- [Parent plan](./plan.md)
- [Phase 07 local MongoDB](./phase-07-docker-cleanup-and-local-mongodb.md)
- [Current dataflow commands](../../README.md)
- Depends on: Phases 01–07

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Run both Dora dataflows directly on the target workstation and enforce functional/performance gates. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Approved |
| Approved at | 2026-07-05 15:08 +07 |
| Recommended model | GPT-5.4 orchestrator; GPT-5.4-mini for bounded Dora/benchmark logs |
| Estimated effort | 6h |

## Key Insights

- Target is this Ryzen 7 8840U host, not Raspberry Pi.
- Native stack still preserves Orchestra/Rover process and Zenoh boundaries.
- MongoDB is the only container in this phase.
- Attached Dora commands require separate terminals or a controlled harness.

## Requirements

### Functional

- Build all changed Rust packages and start MongoDB.
- Start Orchestra first, wait for readiness, then Rover.
- Verify config convergence, English/Vietnamese synthesis, all voices, playback, preemption, and suppression.
- Produce machine-readable benchmark summary and retained concise evidence.

### Non-functional

- Never use existing external Mongo URI from ignored `.env`.
- Use dedicated database `robo_fleet_edge_voice_e2e` with default credentials allowed only for local E2E.
- Bound logs by time/size and preserve raw logs outside tracked files.
- Stop dataflows/processes cleanly after validation.

## Architecture

Required manual-equivalent start order:

```bash
make up-mongodb
dora up

# Terminal 1
MONGODB_URI=mongodb://127.0.0.1:27017 \
MONGODB_DATABASE=robo_fleet_edge_voice_e2e \
ALLOW_DEFAULT_CREDENTIALS=true \
dora start orchestra/orchestra-dataflow.yml --name orchestra --attach

# Terminal 2, after Orchestra ready
ROVER_ORT_DYLIB_PATH="$PWD/models/.runtime/onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so" \
dora start rover-kiwi/rover-kiwi-dataflow.yml --name rover-kiwi --attach
```

Add a non-destructive benchmark script that assumes services are already running; do not hide startup failures behind mocks.

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Create | `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/benchmark-edge-voice-x86.sh` | Repeatable corpus/metrics gate |
| Create | `/mnt/data/ws/sharing/robo-fleet-dora-rs/scripts/fixtures/edge-voice-corpus.json` | English/Vietnamese test corpus |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/performance_monitor/src/main.rs` | Edge voice CPU/RSS visibility |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/Makefile` | Native validation helper target |

## Implementation Steps

1. Run focused unit/integration suites before starting services.
2. Start local MongoDB and verify health at loopback.
3. Run `dora up`; start Orchestra attached; wait for web bridge, STT, and bridge readiness.
4. Start Rover attached; wait for edge voice `ready`, audio playback, bridge, and camera nodes.
5. Verify UI/backend defaults and applied revision 0.
6. Synthesize short/medium English and Vietnamese corpus at balanced steps.
7. Exercise all ten SIDs; confirm non-empty output and command completion.
8. Enable vision workload and repeat benchmark; record baseline and concurrent metrics.
9. Start walkie during long TTS; assert interruption within one frame and visible command result.
10. Assert microphone suppression counter and absence of forwarded rover frames during playback/tail.
11. Run 100 sequential short utterances; record underruns, failures, CPU, RSS, TTFA, and RTF.
12. Give bounded logs/JSON summary to GPT-5.4-mini; main GPT-5.4 verifies every failed/threshold result.

## Todo List

- [x] Changed packages build/test
- [x] MongoDB healthy
- [x] Orchestra starts attached
- [x] Rover starts attached
- [x] Defaults converge
- [x] English/Vietnamese pass
- [x] Ten voices pass
- [x] Walkie preemption passes
- [x] Capture suppression passes
- [x] 100-utterance soak passes
- [x] Metrics summary generated/reviewed

## Completion Notes

- Final benchmark artifacts:
  - `plans/260704-1318-edge-voice-supertonic-x86/reports/phase-08-native-x86-benchmark.json`
  - `plans/260704-1318-edge-voice-supertonic-x86/reports/phase-08-native-x86-evidence.log`
- Final clean-stack benchmark result:
  - `default_revision=0`
  - `p95_ttfa_ms=60.0`
  - `p95_estimated_rtf=0.828`
  - `peak_edge_voice_rss_mb=340.5`
  - `vision_fps_regression_percent=0.00`
  - `capture_samples_rejected=60017`
  - `capture_drops=60017`
- Supporting runtime changes completed in this phase:
  - web bridge TTS bootstrap and ack delivery repaired
  - edge voice runtime asset path resolution added
  - edge voice lifecycle logging added for live diagnosis
  - audio capture suppression accounting fixed so dropped mic samples are observable in benchmark results
  - `performance_monitor` now logs `edge_voice` CPU/RSS at info level for the benchmark harness
  - `Makefile` now exposes `make validate-edge-voice-x86`
  - `scripts/` now contains a dedicated benchmark workspace with README and local `.gitignore`

## Operational Notes

- For local workstation validation, Orchestra now defaults `ALLOW_DEFAULT_CREDENTIALS` to `true` in `orchestra/orchestra-dataflow.yml`. Override it back to `false` in stricter environments.
- The `scripts/` workspace is repo-level runtime tooling, not `robo-control-app` application code.

## Success Criteria

- Cold model load `<10 s`.
- p95 TTFA `<1.0 s` for short balanced utterances.
- p95 RTF `<1.0` for English and Vietnamese with vision active.
- Peak `edge_voice` RSS `<2 GiB`.
- Vision FPS regression `<=10%` versus same stack without synthesis.
- Zero synthesis crashes/OOM, playback underruns, and self-triggered actuator commands.
- Config applies to healthy rover within 2 seconds.

## Risk Assessment

- Risk: physical audio device unavailable. Mitigation: preflight device inventory; distinguish device failure from synthesis failure.
- Risk: attached command orchestration leaves stale dataflows. Mitigation: named flows, trap cleanup in helper, verify `dora list` before/after.
- Risk: metrics affected by unrelated host load. Mitigation: record load, repeat failed threshold three times.
- Risk: camera unavailable. Mitigation: report blocked vision comparison separately; do not claim full gate.

## Security Considerations

- Local default web credentials are allowed only against dedicated loopback test DB.
- Generate ephemeral JWT secret in validation shell.
- Logs must redact credentials, JWTs, and Mongo URI user info.

## Next Steps

- Keep native services running and proceed to [Phase 09](./phase-09-live-web-e2e.md).
