# Phase 1: Convert to Library Crates

Extract core structs from `main.rs` → `lib.rs`. Remove Dora-specific code (event loop, node init). Keep pure domain logic.

## 1.1 — object_detector as lib

**File:** `rover-kiwi/object_detector/src/lib.rs` (NEW, ~200 lines)
- Move `YoloDetector` struct + all methods from main.rs
- Public API: `pub struct YoloDetector`, `pub fn new(env: Arc<Environment>, ...)`, `pub fn detect()`
- Constructor takes `Arc<Environment>` — no internal Environment creation
- Remove: Dora imports, event loop, node init
- Keep: `YOLO_CLASSES` const, ort/image/ndarray logic
- Add `pub struct DetectorConfig` for env var passthrough

**File:** `rover-kiwi/object_detector/src/main.rs` (DELETE)

**File:** `rover-kiwi/object_detector/Cargo.toml`
- Ensure `[lib]` target; remove `dora-node-api` dependency
- Keep: ort, image, ndarray, serde, serde_json, tracing, eyre, robo_rover_lib

### Public API

```rust
pub struct DetectorConfig {
    pub model_path: String,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub target_classes: Vec<String>,
}

pub struct YoloDetector { /* session, config, frame_counter, input_size */ }

impl YoloDetector {
    pub fn new(env: Arc<ort::Environment>, config: DetectorConfig) -> Result<Self>;
    pub fn detect(&mut self, frame_data: &[u8], width: u32, height: u32) -> Result<DetectionFrame>;
}
```

---

## 1.2 — reid_extractor as lib

**File:** `rover-kiwi/reid_extractor/src/lib.rs` (NEW, ~170 lines)
- Move `ReidExtractor` struct + methods from main.rs
- Public API: `pub struct ReIdExtractor`, `pub fn new(env: Arc<Environment>, ...)`, `pub fn process_detections()`
- Constructor takes `Arc<Environment>` — shared with detector
- **Key change:** Remove `current_frame` field. Pass frame data directly to `process_detections()` — zero-copy benefit
- Add `pub struct ReIdConfig`

**File:** `rover-kiwi/reid_extractor/src/main.rs` (DELETE)

**File:** `rover-kiwi/reid_extractor/Cargo.toml`
- `[lib]` crate, remove `dora-node-api`
- Keep: ort, image, ndarray, serde, serde_json, tracing, eyre, robo_rover_lib

### Public API

```rust
pub struct ReIdConfig {
    pub model_path: String,
    pub min_bbox_size: u32,
}

pub struct ReIdExtractor { /* session, feature_dim, min_bbox_size */ }

impl ReIdExtractor {
    pub fn new(env: Arc<ort::Environment>, config: ReIdConfig) -> Result<Self>;
    pub fn process_detections(
        &mut self,
        frame_data: &[u8],
        width: u32,
        height: u32,
        detection_frame: DetectionFrame,
    ) -> Result<DetectionFrame>;
}
```

Note: `process_detections` takes `frame_data: &[u8]` directly instead of storing frame internally. Zero-copy from kornia_capture's grabbed frame.

---

## 1.3 — object_tracker as lib (split into 3 files)

### kalman.rs (NEW, ~130 lines)

```rust
pub struct KalmanFilter {
    state: Vector4<f32>,
    covariance: Matrix4<f32>,
    process_noise: Matrix4<f32>,
    measurement_noise: Matrix2<f32>,
    transition: Matrix4<f32>,
    measurement: Matrix2x4<f32>,
}

impl KalmanFilter {
    pub fn new(initial_x: f32, initial_y: f32) -> Self;
    pub fn predict(&mut self);
    pub fn update(&mut self, measurement_x: f32, measurement_y: f32);
    pub fn get_position(&self) -> (f32, f32);
    pub fn get_velocity(&self) -> (f32, f32);
}
```

### tracked_object.rs (NEW, ~170 lines)

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum InternalTrackState { New, Tracked, Lost }

pub struct TrackedObject {
    pub id: u32,
    pub class_name: String,
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub(crate) kalman: KalmanFilter,
    pub frames_since_update: u32,
    pub total_frames: u32,
    pub last_seen: u64,
    pub reid_features: Option<Vec<f32>>,
    pub state: InternalTrackState,
    pub hits: u32,
}

impl TrackedObject {
    pub fn new(id: u32, detection: &DetectionResult) -> Self;
    pub fn predict(&mut self);
    pub fn update(&mut self, detection: &DetectionResult, min_hits: u32);
    pub fn apply_camera_motion(&mut self, transform: &Matrix3<f32>);
    pub fn reid_similarity(&self, detection: &DetectionResult) -> Option<f32>;
    pub fn get_predicted_bbox(&self) -> BoundingBox;
    pub fn to_tracking_target(&self) -> TrackingTarget;
}
```

### lib.rs (NEW, ~250 lines)

```rust
pub mod cmc;
pub mod kalman;
pub mod tracked_object;

pub struct TrackerConfig {
    pub max_age: u32,
    pub min_hits: u32,
    pub iou_threshold: f32,
    pub reid_weight: f32,
    pub reid_threshold: f32,
    pub enable_cmc: bool,
}

pub struct ObjectTracker { /* tracks, next_id, config, cmc */ }

impl ObjectTracker {
    pub fn new(config: TrackerConfig) -> Self;
    pub fn process_frame(&mut self, frame_data: &[u8], width: u32, height: u32);
    pub fn update(&mut self, detections: Vec<DetectionResult>);
    pub fn handle_tracking_command(&mut self, command: TrackingCommand);
    pub fn get_tracking_telemetry(&self) -> TrackingTelemetry;
    pub fn get_all_tracks(&self) -> Vec<DetectionResult>;
}
```

### Files unchanged
- `rover-kiwi/object_tracker/src/cmc.rs` — no changes needed

### Files deleted
- `rover-kiwi/object_tracker/src/main.rs` — DELETE

### Cargo.toml changes
- `[lib]` crate, remove `dora-node-api`
- Keep: nalgebra, image, imageproc, serde, serde_json, tracing, eyre, robo_rover_lib, rand, pathfinding

---

## Verification

After Phase 1, all 3 crates compile as libraries:

```bash
cargo build -p object_detector   # lib only, no binary produced
cargo build -p reid_extractor    # lib only
cargo build -p object_tracker    # lib only
```

No dataflow changes yet — rover dataflow will NOT work until Phase 2+3 complete (old nodes reference deleted binaries). This is expected.
