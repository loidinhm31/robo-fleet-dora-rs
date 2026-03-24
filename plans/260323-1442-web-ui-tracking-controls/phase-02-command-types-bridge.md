# Phase 2: Command Types, Bridge Conversion + Dataflow Output

> Parent: [plan.md](plan.md) | Depends on: [Phase 1](phase-01-vision-pipeline-split.md)

## Overview

- **Priority:** P1
- **Status:** done
- **Effort:** 45m
- **Completed:** 2026-03-24

Wire `DetectionOnly` output through Dora → zenoh → web_bridge. Extend web_bridge's command converter for new variants. Add `detections` output to dataflow YAMLs.

## Changes

### 1. kornia_capture main.rs — handle DetectionOnly output

```rust
// In send_pipeline_output():
Ok(PipelineOutput::DetectionOnly { detections }) => {
    let det_json = serde_json::to_vec(&detections)?;
    node.send_output(
        DataId::from("detections".to_owned()),
        Default::default(),
        BinaryArray::from_vec(vec![det_json.as_slice()]),
    )?;
}
```

### 2. Dataflow YAMLs — add `detections` output

**rover-kiwi-dataflow.yml** (`gst-camera` node):
```yaml
outputs:
  - frame
  - detections            # NEW: DetectionFrame (detection-only mode)
  - tracked_detections
  - tracking_telemetry
```

Route `detections` through zenoh-bridge to orchestra.

**orchestra-dataflow.yml** (web-bridge inputs):
```yaml
inputs:
  detections: orchestra-bridge/detections    # NEW
  tracked_detections: orchestra-bridge/tracked_detections
```

**rover-kiwi-direct-dataflow.yml** (if applicable):
```yaml
# Route detections from gst-camera to web-bridge
```

### 3. Zenoh bridge — audit + wire `detections` topic (VALIDATED: needs investigation)

Both zenoh bridges need to forward the `detections` topic. **Must read source before implementing:**
- `rover-kiwi/zenoh_bridge/src/main.rs` — how topics are published, whether explicit list or pattern
- `orchestra/zenoh_bridge/src/main.rs` — how topics are subscribed, whether explicit list or pattern
- Document: does adding a new topic require code changes or just dataflow YAML?
- If explicit topic lists: add `detections` to both bridges
- If wildcard: verify `detections` matches the pattern

### 4. Web bridge — extend command converter

```rust
// In convert_web_command_to_tracking_command():
"enable_detection" => Some(TrackingCommand::EnableDetection { timestamp }),
"disable_detection" => Some(TrackingCommand::DisableDetection { timestamp }),
```

Bridge already has `detections` Dora input handler (line 1175) → emits Socket.IO `detections`. No new handler needed.

## Related Files

| File | Action |
|------|--------|
| `rover-kiwi/kornia_capture/src/main.rs` | Handle `PipelineOutput::DetectionOnly` |
| `rover-kiwi/rover-kiwi-dataflow.yml` | Add `detections` output + zenoh routing |
| `orchestra/orchestra-dataflow.yml` | Add `detections` input to web-bridge |
| `rover-kiwi/rover-kiwi-direct-dataflow.yml` | Add `detections` routing |
| `common/web_bridge/src/main.rs` | +2 cases in convert function (~6 lines) |
| Zenoh bridges (both) | Verify `detections` topic forwarded |

## Implementation Steps

- [x] Add `detections` output handling in kornia_capture main.rs
- [x] Add `detections` to gst-camera outputs in rover-kiwi-dataflow.yml
- [x] Route `detections` through zenoh-bridge in rover dataflow
- [x] Add `detections` to orchestra-bridge outputs in orchestra-dataflow.yml
- [x] Add `detections` to web-bridge inputs in orchestra-dataflow.yml
- [x] Update rover-kiwi-direct-dataflow.yml
- [x] **Audit zenoh bridges**: read both bridge source files, document topic wiring mechanism
- [x] Wire `detections` topic in zenoh bridges (method TBD from audit)
- [x] Add `enable_detection`/`disable_detection` cases to web_bridge convert function
- [x] `cargo build --release -p web_bridge` passes

## Success Criteria

- `cargo build --release` for web_bridge + kornia_capture
- Detection-only mode sends `detections` → arrives at web_bridge → emits Socket.IO `detections`
- Existing `tracking_command` enable/disable flow unchanged
- Dataflow graph validates: `dora check` on all 3 dataflow YAMLs
