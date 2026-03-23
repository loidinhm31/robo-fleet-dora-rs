# Phase 3: Update Dataflow & Bridges

## 3.1 — rover-kiwi-dataflow.yml (REWRITE)

### Remove nodes

Delete these node definitions entirely:
- `object-detector`
- `reid-extractor`
- `object-tracker`

### Update gst-camera node

```yaml
- id: gst-camera
  build: cargo build -p kornia_capture --release
  path: ../target/release/kornia_capture
  inputs:
    tick: dora/timer/millis/33
    camera_control: zenoh-bridge/camera_command
    tracking_command: zenoh-bridge/tracking_command   # MOVED from object-tracker
  outputs:
    - frame                  # Raw RGB8 (always, for zenoh video stream)
    - tracked_detections     # NEW — when pipeline enabled
    - tracking_telemetry     # NEW — when pipeline enabled
  env:
    # Camera config (existing)
    SOURCE_TYPE: "webcam"
    SOURCE_URI: "/dev/video0"
    IMAGE_COLS: "640"
    IMAGE_ROWS: "480"
    SOURCE_FPS: "30"
    # Detector config (moved from object-detector)
    MODEL_PATH: "${HOME}/.cache/yolo/yolo12n.onnx"
    CONFIDENCE_THRESHOLD: "0.5"
    NMS_THRESHOLD: "0.4"
    TARGET_CLASSES: ""
    ORT_DYLIB_PATH: "/usr/local/lib/libonnxruntime.so"
    # ReID config (moved from reid-extractor)
    REID_MODEL_PATH: "${HOME}/.cache/reid/osnet_x0_25.onnx"
    MIN_BBOX_SIZE: "32"
    # Tracker config (moved from object-tracker)
    MAX_TRACKING_AGE: "50"
    MIN_HITS: "3"
    IOU_THRESHOLD: "0.3"
    REID_WEIGHT: "0.8"
    REID_THRESHOLD: "0.5"
    ENABLE_CMC: "true"
```

### Update visual-servo-controller

```yaml
- id: visual-servo-controller
  inputs:
    tracking_telemetry: gst-camera/tracking_telemetry   # CHANGED from object-tracker/tracking_telemetry
  # ... rest unchanged
```

### Update zenoh-bridge inputs

```yaml
- id: zenoh-bridge
  inputs:
    video_frame: gst-camera/frame                        # unchanged
    audio_frame: audio-capture/audio                     # unchanged
    servo_telemetry: visual-servo-controller/servo_telemetry  # unchanged
    performance_metrics: performance-monitor/metrics      # unchanged
    tracked_detections: gst-camera/tracked_detections     # CHANGED from object-tracker
    tracking_telemetry: gst-camera/tracking_telemetry     # CHANGED from object-tracker
  outputs:
    - camera_command
    - audio_command
    - rover_command
    - arm_command
    - tracking_command    # unchanged — now routed to gst-camera via dataflow
    - tts_command
    - audio_stream
  # ... rest unchanged
```

---

## 3.2 — No bridge Rust code changes

Existing `tracking_command` flow is fully wired end-to-end:

```
Web UI → tracking_command socket.io → web_bridge
  → tracking_command Dora output → orchestra zenoh_bridge
  → Zenoh rover/{id}/cmd/tracking
  → rover zenoh_bridge → tracking_command Dora output
  → gst-camera (via dataflow YAML routing — the only change)
```

**Zero Rust code changes** in:
- `common/web_bridge/src/main.rs`
- `orchestra/zenoh_bridge/src/main.rs`
- `rover-kiwi/zenoh_bridge/src/main.rs`

---

## 3.3 — Orchestra dataflow YAML

Verify no references to removed rover-side nodes. Orchestra dataflow doesn't directly reference rover nodes (separate machines, connected via Zenoh). **No changes expected.**

Confirm by checking: `orchestra/orchestra-dataflow.yml` should have no mention of `object-detector`, `reid-extractor`, or `object-tracker`.

---

## Verification

After Phase 3:

```bash
# Build the consolidated binary
cargo build --release -p kornia_capture

# Start rover dataflow
dora up && dora start rover-kiwi/rover-kiwi-dataflow.yml --name rover-kiwi --attach
```

Expected: Only these processes spawn:
- `kornia_capture` (camera + vision pipeline)
- `rover_zenoh_bridge`
- `audio_capture`
- `audio_playback`
- `performance_monitor`
- `visual_servo_controller`
- `arm_controller`
- `rover_controller`

**NOT** spawned: `object_detector`, `reid_extractor`, `object_tracker` (no longer nodes)
