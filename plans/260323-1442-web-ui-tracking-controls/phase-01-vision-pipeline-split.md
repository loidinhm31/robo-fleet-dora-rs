# Phase 1: Split VisionPipeline State

> Parent: [plan.md](plan.md) | Dependencies: none

## Overview

- **Priority:** P1
- **Status:** done
- **Effort:** 45m
- **Completed:** 2026-03-24

Split `pipeline_enabled: bool` into `detection_enabled + tracking_enabled`. Add `PipelineOutput::DetectionOnly`. Implement graceful degradation (ReID error → detection-only fallback, not full disable).

## Architecture

```rust
// New state machine in VisionPipeline
detection_enabled: bool,  // Gates YOLO
tracking_enabled: bool,   // Gates ReID + BoTSORT (implies detection)

// Process logic:
// !detection && !tracking → CameraOnly
// detection && !tracking  → YOLO only → DetectionOnly
// tracking (any)          → YOLO + ReID + BoTSORT → FullTracking
```

### PipelineOutput

```rust
pub enum PipelineOutput {
    CameraOnly,
    DetectionOnly {
        detections: DetectionFrame,  // Raw YOLO, no tracking IDs
    },
    FullTracking {
        tracked_detections: DetectionFrame,
        tracking_telemetry: TrackingTelemetry,
    },
}
```

### Error Recovery (graceful degradation)

| Error | Current behavior | New behavior |
|-------|-----------------|-------------|
| YOLO failure | Disable entire pipeline | Disable detection + tracking → CameraOnly |
| ReID failure | Disable entire pipeline | **Disable tracking only → fallback to DetectionOnly** |
| Model load failure | Disable entire pipeline | Same, but log which model failed |

### Command Handling

```rust
pub fn handle_tracking_command(&mut self, cmd: TrackingCommand) {
    match &cmd {
        TrackingCommand::EnableDetection { .. } => {
            self.detection_enabled = true;
            // tracking stays as-is
        }
        TrackingCommand::DisableDetection { .. } => {
            self.detection_enabled = false;
            self.tracking_enabled = false;  // auto-disable tracking
        }
        TrackingCommand::Enable { .. } => {
            self.detection_enabled = true;   // auto-enable detection
            self.tracking_enabled = true;
        }
        TrackingCommand::Disable { .. } => {
            self.detection_enabled = false;
            self.tracking_enabled = false;
        }
        _ => {} // SelectTarget/ClearTarget → delegated to tracker
    }
    self.tracker.handle_tracking_command(cmd);
}
```

## Related Files

| File | Action |
|------|--------|
| `rover-kiwi/kornia_capture/src/vision_pipeline.rs` | Rewrite: split state, new output, graceful degradation |
| `robo_rover_lib/src/types/detection_types.rs` | Add TrackingCommand variants, TrackingState::DetectionOnly, remove DetectionControl |

## Implementation Steps

- [x] Add `EnableDetection`/`DisableDetection` to `TrackingCommand` enum + constructors
- [x] Add `DetectionOnly` to `TrackingState` enum
- [x] Remove dead `DetectionControl` + `DetectionAction` types
- [x] Replace `pipeline_enabled` with `detection_enabled` + `tracking_enabled` in VisionPipeline
- [x] Add `PipelineOutput::DetectionOnly` variant
- [x] Update `process_frame()`: 3-way branch (camera / detect / full)
- [x] Update `handle_tracking_command()`: handle new variants + auto-enable/disable logic
- [x] Implement graceful degradation: ReID error → fallback to DetectionOnly
- [x] `cargo build -p kornia_capture` passes

## Success Criteria

- `cargo build --release -p kornia_capture` compiles
- `TrackingCommand::Enable` behavior unchanged (backward compatible)
- ReID failure degrades to detection-only, not camera-only
- Detection-only mode skips ReID entirely (saves ~50ms/frame)

## Edge Cases

| Scenario | Expected |
|----------|----------|
| Enable tracking directly (skip detection) | Auto-enables detection, loads all models |
| Disable detection while tracking active | Auto-disables tracking |
| Re-enable after auto-disable from error | Models stay loaded, instant re-enable |
| YOLO error in detection-only | → CameraOnly |
| ReID error in full pipeline | → DetectionOnly (NOT CameraOnly) |
