# Phase 2: Build VisionPipeline in kornia_capture

## 2.1 — Add DetectionControl type (future-proofing)

**File:** `robo_rover_lib/src/types/detection_types.rs` (+15 lines)

Add after existing `TrackingCommand` enum:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionControl {
    pub command: DetectionAction,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionAction {
    Enable,
    Disable,
}
```

Already re-exported via `pub use detection_types::*;` in mod.rs — no mod.rs change needed.

---

## 2.2 — Create vision_pipeline.rs

**File:** `rover-kiwi/kornia_capture/src/vision_pipeline.rs` (NEW, ~180 lines)

### Structs

```rust
use std::sync::Arc;
use eyre::Result;
use object_detector::{YoloDetector, DetectorConfig};
use reid_extractor::{ReIdExtractor, ReIdConfig};
use object_tracker::{ObjectTracker, TrackerConfig};
use robo_rover_lib::types::{DetectionFrame, TrackingCommand, TrackingTelemetry};

pub struct VisionPipeline {
    ort_env: Option<Arc<ort::Environment>>,
    detector: Option<YoloDetector>,
    reid: Option<ReIdExtractor>,
    tracker: ObjectTracker,

    pipeline_enabled: bool,

    detector_config: DetectorConfig,
    reid_config: ReIdConfig,
}

pub enum PipelineOutput {
    CameraOnly,
    FullTracking {
        tracked_detections: DetectionFrame,
        tracking_telemetry: TrackingTelemetry,
    },
}
```

### Constructor

```rust
impl VisionPipeline {
    pub fn new(
        detector_config: DetectorConfig,
        reid_config: ReIdConfig,
        tracker_config: TrackerConfig,
    ) -> Self {
        Self {
            ort_env: None,
            detector: None,
            reid: None,
            tracker: ObjectTracker::new(tracker_config),
            pipeline_enabled: false,
            detector_config,
            reid_config,
        }
    }
}
```

### Core: process_frame

```rust
pub fn process_frame(&mut self, frame: &[u8], w: u32, h: u32) -> Result<PipelineOutput> {
    if !self.pipeline_enabled {
        return Ok(PipelineOutput::CameraOnly);
    }

    // Lazy-init shared ONNX environment + models
    self.ensure_models_loaded()?;

    let detector = self.detector.as_mut().unwrap();
    let reid = self.reid.as_mut().unwrap();

    // YOLO → ReID → CMC → Tracker
    let detection_frame = detector.detect(frame, w, h)?;
    let enriched = reid.process_detections(frame, w, h, detection_frame)?;

    self.tracker.process_frame(frame, w, h);  // CMC
    self.tracker.update(enriched.detections.clone());

    let telemetry = self.tracker.get_tracking_telemetry();
    let mut tracked_frame = enriched;
    tracked_frame.detections = self.tracker.get_all_tracks();

    Ok(PipelineOutput::FullTracking {
        tracked_detections: tracked_frame,
        tracking_telemetry: telemetry,
    })
}
```

### Lazy model init

```rust
fn ensure_models_loaded(&mut self) -> Result<()> {
    if self.detector.is_some() && self.reid.is_some() {
        return Ok(());
    }

    tracing::info!("Loading ML models (first enable)...");

    let env = match &self.ort_env {
        Some(env) => env.clone(),
        None => {
            let env = ort::Environment::builder()
                .with_name("vision_pipeline")
                .with_execution_providers([
                    ort::ExecutionProvider::CPU(Default::default()),
                ])
                .build()?
                .into_arc();
            self.ort_env = Some(env.clone());
            env
        }
    };

    if self.detector.is_none() {
        let detector = YoloDetector::new(env.clone(), self.detector_config.clone())?;
        tracing::info!("YOLO detector loaded");
        self.detector = Some(detector);
    }

    if self.reid.is_none() {
        let reid = ReIdExtractor::new(env, self.reid_config.clone())?;
        tracing::info!("ReID extractor loaded");
        self.reid = Some(reid);
    }

    Ok(())
}
```

### Tracking command handler

```rust
pub fn handle_tracking_command(&mut self, cmd: TrackingCommand) {
    match &cmd {
        TrackingCommand::Enable { .. } => {
            tracing::info!("Pipeline enabled (tracking on)");
            self.pipeline_enabled = true;
        }
        TrackingCommand::Disable { .. } => {
            tracing::info!("Pipeline disabled (tracking off)");
            self.pipeline_enabled = false;
            // Don't drop models — avoids cold start on re-enable
        }
        _ => {}  // SelectTarget, ClearTarget — delegated to tracker below
    }
    self.tracker.handle_tracking_command(cmd);
}
```

### Error recovery

If YOLO or ReID fails during `process_frame`, catch at VisionPipeline level:
- Log error
- Auto-disable pipeline (`self.pipeline_enabled = false`)
- Return `CameraOnly` — camera keeps streaming
- User must re-enable tracking to retry

---

## 2.3 — Update kornia_capture/Cargo.toml

```toml
[package]
name = "kornia_capture"
version = "0.1.0"
edition = "2021"

[dependencies]
dora-node-api = { workspace = true }
robo_rover_lib = { path = "../../robo_rover_lib" }
kornia-io = { version = "0.1.10-rc.3", features = ["gstreamer"] }
tracing = { workspace = true }
serde_json = "1.0"
# Vision pipeline deps
object_detector = { path = "../object_detector" }
reid_extractor = { path = "../reid_extractor" }
object_tracker = { path = "../object_tracker" }
eyre = "0.6.12"
ort = { version = "1.16.3", default-features = false, features = ["load-dynamic", "download-binaries"] }
```

---

## 2.4 — Rewrite kornia_capture/src/main.rs

Current: 217 lines with duplicated webcam/rtsp branches.

### Refactored structure (~200 lines)

1. **DRY the camera branches**: Extract frame grabbing + pipeline processing into shared function
2. **Add `mod vision_pipeline;`**
3. **Parse all env vars** (camera + detector + reid + tracker)
4. **Handle new inputs**: `tracking_command` (deserialize BinaryArray → TrackingCommand → delegate)
5. **Tick handler** (shared between webcam/rtsp):
   - Grab frame from camera
   - Send raw `frame` output (always — for video streaming)
   - Call `pipeline.process_frame(frame_data, w, h)`
   - Match on `PipelineOutput`:
     - `CameraOnly` → no additional outputs
     - `FullTracking` → send `tracked_detections` + `tracking_telemetry` as BinaryArray JSON

### Input handling additions

```rust
"tracking_command" => {
    if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
        if binary_array.len() > 0 {
            let cmd_bytes = binary_array.value(0);
            if let Ok(cmd) = serde_json::from_slice::<TrackingCommand>(cmd_bytes) {
                tracing::info!("Tracking command: {:?}", cmd);
                pipeline.handle_tracking_command(cmd);
            }
        }
    }
}
```

### Output sending (after frame grab)

```rust
// Always send raw frame for video streaming
node.send_output_bytes(frame_output.clone(), params.clone(), frame.numel(), frame.as_slice())?;

// Run vision pipeline
match pipeline.process_frame(frame.as_slice(), width as u32, height as u32) {
    Ok(PipelineOutput::FullTracking { tracked_detections, tracking_telemetry }) => {
        let det_json = serde_json::to_vec(&tracked_detections)?;
        let det_data = BinaryArray::from_vec(vec![det_json.as_slice()]);
        node.send_output(DataId::from("tracked_detections".to_owned()), Default::default(), det_data)?;

        let tel_json = serde_json::to_vec(&tracking_telemetry)?;
        let tel_data = BinaryArray::from_vec(vec![tel_json.as_slice()]);
        node.send_output(DataId::from("tracking_telemetry".to_owned()), Default::default(), tel_data)?;
    }
    Ok(PipelineOutput::CameraOnly) => {} // No ML outputs
    Err(e) => {
        tracing::error!("Vision pipeline error: {:?}", e);
        // Pipeline auto-disables on error (handled inside VisionPipeline)
    }
}
```

---

## Verification

After Phase 2:

```bash
cargo build -p kornia_capture  # Should compile, pulling in all 3 lib crates
```

Binary exists but dataflow not yet wired — Phase 3 updates the YAML.
