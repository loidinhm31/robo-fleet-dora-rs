# Brainstorm Summary: Video Encoder Redundancy Analysis

## Problem Statement and Requirements
Currently, the rover video pipeline uses `kornia_capture` (which wraps GStreamer) to capture frames from the camera. The camera stream is decoded into raw `RGB8` frames. These raw frames are then duplicated:
1. Sent to the `vision_worker` for ML processing (YOLO/ReID) which requires RGB tensors.
2. Sent out via `gst-camera/frame` to the `video_encoder` node, which re-encodes the RGB frame back into JPEG for the Web UI.

The user insight is brilliant: **If the source camera natively provides compressed video (like MJPEG or H264), decoding it to RGB8 only to re-encode it back to JPEG is fundamentally redundant and wastes significant CPU on the rover.** Because we draw bounding boxes client-side in the Web UI (using metadata) rather than burning them into the video frames, the rover does not actually need to modify the view stream pixels.

## Evaluated Approaches

### Approach A: GStreamer Native Tee Pipeline (Bypass `video_encoder`)
Modify the GStreamer pipeline inside `kornia_capture` to use a `tee` element immediately after the camera source:
- **Branch 1 (View Stream):** Extract the native compressed stream (e.g., MJPEG) via an `appsink` directly. Send these bytes straight to Zenoh as the `video_frame`.
- **Branch 2 (ML Stream):** Pass through `jpegdec -> videoconvert -> appsink` to get the RGB8 frames needed for the ML worker.
**Pros:** 
- Adheres strictly to **DRY** (Don't Repeat Yourself - computationally).
- Eliminates the `video_encoder` software encoding cost entirely (huge CPU savings).
**Cons:**
- Tightly couples the pipeline to the camera's native format. If a camera outputs YUYV (raw), we still need an encoder. If it outputs H.264, the frontend must support H.264 decoding instead of simple JPEG `<img>` tags.
- Requires synchronizing `frame_id` across two different GStreamer appsinks to ensure Web UI bounding boxes match the frames.

### Approach B: Hardware-Accelerated Encoding in `video_encoder`
Keep the current architecture where `kornia_capture` outputs RGB8, but modify `video_encoder` to use hardware acceleration (e.g., VAAPI, NVENC, or specialized edge TPU encoders).
**Pros:**
- Agnostic to the camera's source format.
- Maintains architectural separation of concerns.
**Cons:**
- Hardware encoders are heavily platform-dependent (e.g., works on Jetson but fails on Raspberry Pi).
- Still violates **KISS** by doing a fundamentally unnecessary decode/re-encode cycle.

## Final Recommended Solution with Rationale
**Approach A (GStreamer Native Tee) with a Fallback.**
We should modify `kornia_capture` to conditionally extract the native MJPEG stream if the camera supports it. 
- If `SOURCE_FORMAT=mjpeg`, GStreamer tees the native bytes to a new `encoded_frame` output and bypasses `video_encoder`.
- If `SOURCE_FORMAT=raw`, it outputs RGB8 and relies on `video_encoder` as a fallback.
This honors **YAGNI** by not building complex hardware encoders, and **KISS** by routing the already-encoded data straight to the destination.

## Implementation Considerations and Risks
1. **GStreamer Complexity:** Managing multiple appsinks in Rust GStreamer bindings requires careful synchronization. Both sinks must yield the exact same frame timestamp so the ML bounding boxes perfectly align with the visual frame in the UI.
2. **Web UI Compatibility:** This assumes the native compression format is MJPEG (which the current UI expects). If the camera is RTSP H.264, the Web Bridge/UI will need a massive overhaul to handle fragmented MP4/WebRTC.

## Next Steps and Dependencies
- Audit the current camera hardware (e.g., does `/dev/video0` output MJPEG natively via V4L2?).
- Experiment with a dual-appsink GStreamer pipeline in Rust.
- Evaluate if `video_encoder` can be conditionally removed from `rover-kiwi-dataflow.yml`.
