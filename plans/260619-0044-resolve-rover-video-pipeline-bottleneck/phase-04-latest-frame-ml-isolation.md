# Phase 04: Latest-Frame ML Isolation

## Context Links

- [Parent plan](./plan.md)
- [Phase 03](./phase-03-binary-browser-delivery-and-demand-control.md)
- [Bottleneck analysis](../reports/bottleneck-analysis-260618-2231-current-rover-video-pipeline.md)
- Depends on: Phase 03 milestone passed

## Overview

- Date: 2026-06-19
- Priority: P1
- Implementation status: Complete 2026-06-24
- Review status: Complete 2026-06-24
- Purpose: prevent slow inference from blocking camera events or accumulating stale frames.

## Key Insights

- Capture, complete ML, commands, and telemetry currently share one synchronous Dora loop.
- Fresh lower-rate ML results are safer than higher-rate stale results for visual servo.
- An unbounded channel would move the bottleneck and increase latency rather than solve it.

## Requirements

- Dora event handling, camera control, capture, and view publication remain responsive on the main thread.
- `VisionPipeline` runs in one dedicated worker to avoid concurrent model access.
- At most one pending raw frame. New input replaces an unprocessed older frame.
- Tracking controls are applied between inference jobs; mode changes do not queue behind frames.
- Results older than 150 ms are discarded before servo output.
- All replacement, stale-result, worker, and command delays are measured.
- Detector and ReID intra-op thread counts are environment-configurable; the constrained host profile uses 2 detector threads and 1 ReID thread.

## Architecture

```text
main Dora/capture loop
  -> replaceable pending-frame slot (capacity 1) -> vision worker
  <- bounded result channel ----------------------|
  -> Dora detection/tracking/servo outputs

control channel -> worker, checked before next frame
```

The worker owns `VisionPipeline`. Frame and result queues are bounded. Shutdown closes channels and joins worker cleanly.

## Related Code Files

- Modify `rover-kiwi/kornia_capture/src/main.rs`: orchestration, result drain, shutdown.
- Modify `rover-kiwi/kornia_capture/src/vision_pipeline.rs`: worker-safe API and stage timing.
- Create focused modules under `rover-kiwi/kornia_capture/src/` for latest-frame slot and vision worker; keep files near 200 lines.
- Modify `rover-kiwi/visual_servo_controller/src/main.rs` only if explicit frame-age rejection is also required at the consumer boundary.

## Implementation Steps

1. Define owned `CapturedFrame` with frame ID, capture timestamp, dimensions, and RGB bytes. Copy once when handing a frame to the worker.
2. Implement a synchronized single-slot latest-frame buffer with atomic replacement and drop count. Do not use an unbounded FIFO.
3. Move `VisionPipeline` construction and processing into one worker thread.
4. Add a bounded command channel. Drain/coalesce tracking mode commands before taking the next frame; selection commands retain order where semantically required.
5. Return typed pipeline results through a bounded channel with original identity and processing timestamps.
6. On each main-loop event, drain available results before/after capture. Drop results whose capture age exceeds 150 ms.
7. Preserve progressive tracking state transitions and lazy model loading behavior.
8. Make worker failure observable and safe: stop autonomous servo output, report error, keep camera/control loop alive when possible.
9. On stop, signal worker, close channels, join with bounded shutdown behavior, and log final counters.
10. Add deterministic unit tests for replacement order, capacity, command precedence, stale rejection, shutdown, and worker error.
11. Expose detector/ReID thread settings without changing model semantics; set 2/1 only in the constrained workstation profile.
12. Benchmark typical and crowded full tracking using both the live camera and the checksum-identified frame corpus while toggling modes and camera commands.

## Todo List

- [x] Worker owns all ML state.
- [x] Pending frame depth cannot exceed one.
- [x] Replacement/stale drops counted.
- [x] Controls prioritized between jobs.
- [x] Old results never reach servo.
- [x] Shutdown and failure paths tested.
- [ ] Full tracking milestone passed in live/constrained benchmark.

## Success Criteria

- Pending ML depth never exceeds one.
- Capture averages >=28 FPS regardless of ML duration.
- Typical full tracking produces servo input >=10 Hz with p95 frame age <=150 ms.
- Camera commands apply within 100 ms.
- Tracking-mode change applies within 150 ms or one current inference duration, whichever is greater.
- Operator video still meets Phase 02 latency/FPS/CPU contracts.
- All contracts hold inside the 3 CPU/4 GiB rover container, with average CPU <=2.7 equivalents, peak RSS <=3.5 GiB, and no OOM.
- No deadlock, unbounded allocation, or stale control output during 10-minute crowded tracking.

## Risk Assessment

- Model/session types may not be `Send`: construct and own them entirely inside worker.
- RGB ownership adds a copy: bound to one pending plus one processing frame and measure cost.
- x86_64 results do not predict ARM64 throughput or thermals: treat them only as workstation regression evidence.
- Result drain tied to ticks adds <=one tick latency: include it in age gate.
- Worker panic could leave servo active: detect channel closure and force safe no-target/stop state.

## Security Considerations

- Validate frame dimensions/length before worker allocation and processing.
- Bound all channels and reject unreasonable control payloads upstream.
- Avoid logging ReID features or image data.

## Next Steps

- Proceed to Phase 5 only after freshness milestone passes.
- If it fails, stop and create a measured ML optimization plan. Do not silently reduce ReID/CMC cadence or tracking quality.

## Unresolved Questions

- None before implementation; a measured Phase 4 failure may expose a new ML-specific decision.
