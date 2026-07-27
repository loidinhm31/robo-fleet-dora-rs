# Phase 01: Measurement Contract and Baseline

## Context Links

- [Parent plan](./plan.md)
- [Bottleneck analysis](../reports/bottleneck-analysis-260618-2231-current-rover-video-pipeline.md)
- [Architecture](../../ARCHITECTURE.md)
- Depends on: none

## Overview

- Date: 2026-06-19
- Priority: P1
- Implementation status: Complete
- Review status: Approved
- Completed: 2026-06-21
- Purpose: establish stable frame identity, timestamps, and reproducible evidence before transport changes.

## Key Insights

- Raw bandwidth is mathematically proven, but ML, encoder, and browser shares are not measured.
- Existing performance telemetry infers FPS from CPU and cannot validate this plan.
- Both dataflows run on one host during current validation, so clock skew is not a blocker; capture identity must still survive every process boundary.

## Requirements

- Generate one monotonically increasing `frame_id` and `capture_timestamp_ms` at capture.
- Preserve both through every current boundary without regenerating them.
- Record p50/p95/p99, counts, bytes, drops, and oldest-frame age in five-second windows.
- Benchmark camera-only, detection-only, and typical tracking for 10 minutes after warmup. For crowded scaling, use a 10-minute mixed-scene run containing at least 30 five-second windows with three or more detections plus a direct payload sample confirming three concurrent objects; a hand-held static display is not required to remain fixed for 10 minutes.
- Run a native direct smoke profile and a split-container acceptance profile with the rover limited to 3 CPU and 4 GiB RAM.
- Use the live UVC camera for integration and a fixed corpus of real captured frames for repeatable ML comparisons.

## Architecture

```text
capture identity
  -> raw Dora metadata
  -> rover bridge metadata/envelope
  -> orchestra metadata
  -> encoder metadata
  -> web event
  -> browser render metric
```

Structured metrics are acceptance evidence. The existing CPU-derived dashboard values are not.

## Related Code Files

- Modify `rover-kiwi/kornia_capture/src/main.rs`: create identity and capture timing.
- Modify `orchestra/video_encoder/src/main.rs`: preserve identity; report encode distribution.
- Modify both Zenoh bridges and `common/web_bridge/src/main.rs`: preserve identity; report bytes, age, errors, and emit time.
- Modify UI `packages/ui/src/components/features/CameraViewer.tsx`: measure receive-to-render and capture-to-render age.
- Create `scripts/benchmark-rover-video-pipeline.sh`: reproducible collection commands only.
- Modify dataflow environment values and Docker configuration so camera, model, and ONNX Runtime paths are host-configurable.

## Implementation Steps

1. Add capture counter and Unix-millisecond timestamp after successful `grab_rgb8`; attach `frame_id`, `capture_timestamp_ms`, width, height, and encoding metadata.
2. Propagate metadata unchanged through the current raw path. Do not yet alter topic, codec, rate, or routing.
3. Add bounded metric windows for capture interval, vision total, YOLO, ReID total/count, CMC/tracker, encode, bridge bytes/age, web emit, browser decode/render, and drop/error counts.
4. Emit structured logs with stable field names every five seconds. Avoid per-frame info logs.
5. In the browser, compute render completion age after image draw; retain frame ID in metric records.
6. Add preflight checks for the stable V4L2 path, supported 640x480@30 format, model files, ONNX Runtime, Dora, and Docker/Podman (`XDG_RUNTIME_DIR`, `docker info`, and a real container smoke run).
7. Capture a short representative webcam corpus outside Git-tracked source, record its SHA-256 manifest under this plan, and extract fixed RGB frames for ML stage benchmarks. Accept a user video as an optional replacement, not a prerequisite.
8. Add benchmark collection for CPU equivalents, RSS, cgroup throttling/OOM state, bridge bytes, Dora logs, and live/corpus scenario identifiers.
9. Capture baseline results under this plan's `reports/` directory during implementation.
10. Review metrics for contradictions: input/output count mismatches, negative ages, regenerated IDs, missing stages, or mismatched fixture hashes.

## Todo List

- [x] Capture identity created once.
- [x] Metadata preserved end-to-end.
- [x] All stage windows report p50/p95/p99.
- [x] Benchmark script documented and executable.
- [x] Native and constrained-container profiles pass preflight.
- [x] Live camera and fixed corpus identities recorded.
- [x] Four baseline cases recorded under the revised mixed-scene crowded contract.
- [x] Evidence reviewed before Phase 2.

## Success Criteria

- A sampled frame can be correlated from rover capture to browser draw using one ID.
- Metrics distinguish processing time from queue/frame age.
- Counts show where every drop occurs.
- Baseline rerun produces comparable results and includes CPU, memory, cgroup, and transport evidence.
- Constrained rover execution cannot exceed 3 CPU or 4 GiB; any OOM or missing cgroup evidence fails the phase.
- Any missing or unreliable measurement fails the phase.

## Risk Assessment

- Wall-clock changes corrupt browser age: record monotonic stage durations and validate capture timestamps before each run.
- Instrumentation affects timing: aggregate in memory and log only per window.
- Excess logs fill storage: structured summaries only; rotate/cap benchmark capture.
- Sandbox may hide `/dev/video*` despite UVC binding: require explicit host/container device access before benchmarking.

## Security Considerations

- Do not log frame payloads, credentials, tokens, or Zenoh configuration secrets.
- Validate timestamps and counters before arithmetic to prevent overflow/negative-age artifacts.

## Next Steps

- Complete the isolated Phase 01.1 repository formatting baseline, then proceed to Phase 2.
- If evidence is inconsistent, stop and fix measurement at its source.

## Unresolved Questions

- None.
