# Phase 01 Preflight Evidence

Date: 2026-06-20 Asia/Saigon

## Implemented

- Capture creates one monotonic frame ID and Unix-millisecond timestamp.
- Raw Zenoh envelope validates version, dimensions, payload length, and RGB8 size.
- Capture identity survives rover bridge, orchestra bridge, JPEG encoder, web bridge, and browser.
- Five-second windows report count, bytes, drops/errors, p50/p95/p99/max processing and frame age.
- YOLO, ReID total/per-detection, CMC, tracker, serialization, encode, bridge, emit, and browser render timings added.
- Frame sequence gaps, duplicate IDs, and regressions are observable.
- Preflight/corpus/collection script and constrained workstation Compose override added.

## Automated Evidence

- Affected Rust package check: passed.
- Affected Rust tests: 24 unit tests plus 1 doctest passed.
- UI web and native type checks: passed.
- Rover, direct, and orchestra Dora graph parsing: passed.
- Benchmark script syntax: passed.
- YOLO model, ReID model, ONNX Runtime, and Dora CLI preflight: passed.
- Real-host `docker info` passed using Fedora Podman compatibility.
- Real-host `docker run --rm hello-world` passed.

## Blocking Evidence

- `/dev/video0` exists on the real host but is `root:video` mode `0660`; user `loidinh` is not in group `video`.
- Passwordless sudo is unavailable, so Codex cannot apply the temporary device ACL without user action.
- No live camera/corpus baseline, constrained-container run, 10-minute scenarios, browser correlation sample, or consistency audit can be accepted.
- Phase 01 remains pending; Phase 02 must not start.

## Unresolved Questions

- After camera permission is granted, does the device advertise 640x480 RGB-compatible capture at 30 FPS?
