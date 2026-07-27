# Phase 02: Rover JPEG and Zenoh Cutover

## Context Links

- [Parent plan](./plan.md)
- [Phase 01](./phase-01-measurement-contract-and-baseline.md)
- [Prior JPEG/HLS research](../reports/research-260524-1611-rover-encode-hls-streaming.md)
- Depends on: Phase 01 milestone passed

## Overview

- Date: 2026-06-19
- Completed: 2026-06-24
- Priority: P1
- Implementation status: Approved
- Review status: Approved
- Purpose: replace raw RGB Zenoh traffic with rate-limited rover-side JPEG.

## Key Insights

- Direct mode already runs the existing JPEG encoder on the rover.
- Compression must occur before Zenoh; moving downstream throttles cannot reduce rover bandwidth.
- Local ML must continue receiving full-resolution raw frames independently of viewing cadence.

## Requirements

- View branch: 640x480, JPEG quality 80, target 15 FPS.
- ML branch remains raw and local at capture cadence.
- JPEG is the fixed codec for this plan. Do not add an H.264 abstraction, feasibility spike, or browser decoder.
- Topic becomes `rover/{entity_id}/video/jpeg/v1`; raw topic removed from active code/config.
- Wire packet validates version, dimensions, size, and JPEG payload.
- Coordinated cutover only; rollback uses the previous known-good revision.

## Architecture

```text
camera RGB8 -> local ML/servo
           \-> 15 FPS view output -> JPEG encoder -> rover bridge
               -> Zenoh /video/jpeg/v1 -> orchestra bridge -> web bridge
```

Use `JpegFramePacket` in `robo_rover_lib`: magic/version, frame ID, capture timestamp, width, height, and payload. Fixed endian; bounded decode.

## Related Code Files

- Modify `robo_rover_lib/src/types/video_types.rs`: packet contract and codec tests.
- Modify `rover-kiwi/kornia_capture`, `orchestra/video_encoder`, and both Zenoh bridges: bounded view cadence and packet transport.
- Modify `rover-kiwi/rover-kiwi-dataflow.yml`, direct dataflow, and `orchestra/orchestra-dataflow.yml`: encoder relocation/routing.
- Modify `ARCHITECTURE.md`: replace raw transport decision and bandwidth claims.
- Modify Docker configuration for an x86_64 workstation profile with UVC device mapping and a 3 CPU/4 GiB rover limit; preserve the ARM64 production configuration.

## Implementation Steps

1. Define `JpegFramePacket` encode/decode helpers with a stable magic/version and explicit maximum payload/dimensions.
2. Add tests for round-trip, truncation, bad magic/version, invalid dimensions, oversized payload, and empty/non-JPEG payload.
3. Add `VIEW_STREAM_FPS=15` to capture configuration. Emit view frames by capture timestamp cadence; never reduce local ML input cadence.
4. Wire the existing `video_encoder` after capture in normal rover mode. Ensure it copies capture metadata and records encode duration/size.
5. Change rover bridge video input to encoded `BinaryArray`; validate metadata, build packet, publish only `/video/jpeg/v1`.
6. Change orchestra subscriptions to `/video/jpeg/v1`; decode/validate packet and output JPEG `BinaryArray` with original metadata.
7. Route orchestra bridge JPEG output directly to web bridge. Remove orchestra encoder node and raw bridge output from active dataflow.
8. Align direct mode with the same 15 FPS view output and metadata behavior.
9. Update architecture and deployment comments. Do not rename/move the encoder package during this performance fix.
10. Add host-configurable camera, model, and ONNX Runtime paths. Do not rely on `/home/raspb4`, `/usr/local/lib/libonnxruntime.so`, or a fixed `/dev/video0` when a stable V4L2 path is available.
11. Run the direct dataflow as the functional smoke path, then run separate local rover and orchestra dataflows for the Zenoh cutover gate.
12. Build affected packages; validate both Dora graphs; deploy rover, orchestra, then UI as one coordinated revision.

## Todo List

- [x] Packet contract and negative tests complete.
- [x] Rover view branch capped before encode.
- [x] JPEG published on versioned topic.
- [x] Orchestra encoder removed from active path.
- [x] Raw topic absent from active path.
- [x] Direct and orchestra modes aligned.
- [x] Workstation container profile enforces 3 CPU and 4 GiB without changing ARM64 production defaults.
- [x] Phase benchmark passed.

## Benchmark Evidence

- Native split benchmark: 600s.
- Encoded frames: 8986.
- Throughput: 14.98 FPS.
- Encode errors: 0.
- Rover bridge published: 8986.
- Orchestra received: 8986.

## Success Criteria

- No raw RGB crosses Zenoh.
- Video traffic average <=15 Mbps and at least 90% below baseline.
- Delivered video average >=14.5 FPS; p95 capture-to-display <=500 ms; p99 <=750 ms.
- Under the constrained rover container, camera-only and detection-only transport
  cases average <=2.7 CPU equivalents, peak RSS <=3.5 GiB, no OOM, and no
  unbounded memory growth.
- In full tracking, servo rate and frame age regress no more than 10% from Phase 01.
- In full tracking, transport changes do not increase average CPU by more than 10%
  from Phase 01. The absolute <=2.7 CPU full-tracking gate is evaluated in Phase
  04 after ML isolation and detector/ReID thread controls are implemented.
- All criteria hold for 10 minutes in each vision mode.

## Risk Assessment

- Software JPEG competes with inference: cap input before encode and enforce resource/servo gates.
- Mixed versions lose video: coordinated deploy and immediate rollback revision.
- Corrupt network payload terminates bridge: reject and count packet; continue event loop.
- JPEG size spikes: enforce maximum and report drops.

## Security Considerations

- Treat Zenoh video payload as untrusted at orchestra decode.
- Use checked lengths/arithmetic before slicing or allocation.
- Never deserialize unbounded payloads into large vectors.

## Next Steps

- Proceed to Phase 3 only when every transport milestone passes.
- On failure, stop, isolate the failed stage, and revise the plan. Do not lower FPS/quality or switch codecs automatically.

## Residual Caveat

- Final hybrid cadence not rerun under constrained 3CPU/4GiB container profile. Native split passed and user approved.

## Unresolved Questions

- None.
