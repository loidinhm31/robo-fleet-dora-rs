# Brainstorm Summary: Video Encoder Optimization

## Problem Statement and Requirements
Currently, the rover video pipeline uses `kornia_capture` to decode the camera stream into raw `RGB8` frames. These raw frames are duplicated:
1. Sent to the `vision_worker` for ML processing (YOLO/ReID).
2. Sent to the `video_encoder` node, which re-encodes the RGB frame back into JPEG for the Web UI.

The `video_encoder` node currently relies on the pure Rust `image` crate for JPEG encoding. This has proven to be incredibly slow and highly CPU-intensive, creating a significant bottleneck on the rover. We need a way to efficiently deliver compressed frames to the UI without crippling the system.

## Evaluated Approaches

### Approach A: GStreamer Native Tee Pipeline (Bypass `video_encoder`)
Modify the GStreamer pipeline inside `kornia_capture` to use a `tee` element immediately after the camera source to extract the native compressed MJPEG stream alongside the decoded RGB stream.
**Pros:** 
- Eliminates software encoding cost entirely (huge CPU savings).
**Cons:**
- We discovered that `kornia-io` (up to v0.1.14) firmly hardcodes its camera wrappers (`V4L2CameraConfig`, etc.) to a single `appsink name=sink` returning RGB. It does not support dual-stream tees. Implementing this requires abandoning `kornia-io` and rewriting the GStreamer pipeline manually in `gstreamer-rs`, which increases maintenance burden and violates the **KISS** principle.

### Approach B: Hardware-Accelerated Encoding
Keep the architecture where `kornia_capture` outputs RGB8, but modify `video_encoder` to use hardware acceleration (e.g., VAAPI, NVENC).
**Pros:**
- Agnostic to the camera's source format. Maintains architectural separation of concerns.
**Cons:**
- Hardware encoders are platform-dependent (works on Jetson but fails on Raspberry Pi).

### Approach C: SIMD-Accelerated Encoding via TurboJPEG
Retain the current architecture but replace the slow pure Rust `image` crate in the `video_encoder` node with the `turbojpeg` crate (a wrapper around `libjpeg-turbo`).
**Pros:**
- Extremely fast: Uses SIMD instructions (AVX2/NEON) to encode JPEGs in a fraction of a millisecond.
- Adheres to **KISS**: We don't have to touch GStreamer, write custom pipelines, or manage dual appsinks.
- `kornia-io` already utilizes this under the hood for its own high-performance paths, validating its maturity in our tech stack.
**Cons:**
- Still technically performs a decode/re-encode cycle compared to native passthrough, though the overhead becomes negligible.

## Final Recommended Solution with Rationale
**Approach C: SIMD-Accelerated Encoding via TurboJPEG.**
By replacing the `image` crate with `turbojpeg` in the `video_encoder` node, we can drastically reduce CPU overhead while maintaining our clean, robust node architecture. This avoids the massive engineering effort and fragility of custom GStreamer tees while completely solving the performance bottleneck.

## Implementation Considerations and Risks
- **Dependencies:** Requires the `libjpeg-turbo` library to be available on the target system (e.g., `libturbojpeg0-dev` on Ubuntu/Debian). We must ensure the `Dockerfile.rover-kiwi` is updated to install this package.
- **Data mapping:** `turbojpeg::Image` wraps raw byte slices directly, so the conversion from Dora's Zenoh byte payload to `turbojpeg` input should be zero-copy.

## Success Metrics and Validation Criteria
- `video_encoder` CPU usage drops significantly.
- E2E video latency from camera to Web UI is visibly reduced.
- The pipeline maintains target frame rates (e.g., 30fps) without dropping frames due to encoding backpressure.

## Next Steps and Dependencies
- Update `Dockerfile.rover-kiwi` to install `libturbojpeg0-dev`.
- Update `orchestra/video_encoder/Cargo.toml` to depend on the `turbojpeg` crate.
- Refactor the `video_encoder` node to use `turbojpeg::Compressor::new().compress_to_vec()`.
