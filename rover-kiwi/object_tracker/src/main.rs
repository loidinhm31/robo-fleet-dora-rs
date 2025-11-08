mod cmc;

use cmc::CameraMotionCompensator;
use dora_node_api::{
    dora_core::config::DataId,
    DoraNode,
    Event,
    arrow::array::{BinaryArray, UInt8Array},
};
use eyre::{Context, Result};
use nalgebra as na;
use robo_rover_lib::{init_tracing, types::{
    BoundingBox, DetectionFrame, DetectionResult, TrackingCommand, TrackingState,
    TrackingTarget, TrackingTelemetry,
}};
use std::collections::HashMap;
use std::env;
use tracing::{debug, error, info, warn};

/// Internal track state for lifecycle management (BoTSORT)
#[derive(Debug, Clone, PartialEq)]
enum InternalTrackState {
    New,        // Just created, not yet confirmed
    Tracked,    // Active and confirmed
    Lost,       // Lost but still searchable
}

impl InternalTrackState {
    fn is_active(&self) -> bool {
        matches!(self, InternalTrackState::Tracked)
    }
}

/// Kalman filter for tracking bounding box center (x, y) and velocity (vx, vy)
struct KalmanFilter {
    // State: [x, y, vx, vy]
    state: na::Vector4<f32>,
    // Covariance matrix
    covariance: na::Matrix4<f32>,
    // Process noise covariance
    process_noise: na::Matrix4<f32>,
    // Measurement noise covariance
    measurement_noise: na::Matrix2<f32>,
    // State transition matrix
    transition: na::Matrix4<f32>,
    // Measurement matrix (we only measure position, not velocity)
    measurement: na::Matrix2x4<f32>,
}

impl KalmanFilter {
    fn new(initial_x: f32, initial_y: f32) -> Self {
        let state = na::Vector4::new(initial_x, initial_y, 0.0, 0.0);

        // Initialize with moderate uncertainty
        let covariance = na::Matrix4::from_diagonal(&na::Vector4::new(1.0, 1.0, 10.0, 10.0));

        // Process noise (model uncertainty)
        let process_noise = na::Matrix4::from_diagonal(&na::Vector4::new(0.01, 0.01, 0.1, 0.1));

        // Measurement noise (sensor uncertainty)
        let measurement_noise = na::Matrix2::from_diagonal(&na::Vector2::new(0.1, 0.1));

        // State transition (constant velocity model): x_k = x_{k-1} + vx * dt
        // Assuming dt = 1 frame
        #[rustfmt::skip]
        let transition = na::Matrix4::new(
            1.0, 0.0, 1.0, 0.0,  // x = x + vx
            0.0, 1.0, 0.0, 1.0,  // y = y + vy
            0.0, 0.0, 1.0, 0.0,  // vx = vx
            0.0, 0.0, 0.0, 1.0,  // vy = vy
        );

        // Measurement matrix (we observe only position)
        #[rustfmt::skip]
        let measurement = na::Matrix2x4::new(
            1.0, 0.0, 0.0, 0.0,  // Measure x
            0.0, 1.0, 0.0, 0.0,  // Measure y
        );

        Self {
            state,
            covariance,
            process_noise,
            measurement_noise,
            transition,
            measurement,
        }
    }

    fn predict(&mut self) {
        // Predict state: x_k = F * x_{k-1}
        self.state = self.transition * self.state;

        // Predict covariance: P_k = F * P_{k-1} * F^T + Q
        self.covariance = self.transition * self.covariance * self.transition.transpose() + self.process_noise;
    }

    fn update(&mut self, measurement_x: f32, measurement_y: f32) {
        let measurement = na::Vector2::new(measurement_x, measurement_y);

        // Innovation: y = z - H * x
        let innovation = measurement - self.measurement * self.state;

        // Innovation covariance: S = H * P * H^T + R
        let innovation_cov = self.measurement * self.covariance * self.measurement.transpose() + self.measurement_noise;

        // Kalman gain: K = P * H^T * S^{-1}
        if let Some(inv_innovation_cov) = innovation_cov.try_inverse() {
            let kalman_gain = self.covariance * self.measurement.transpose() * inv_innovation_cov;

            // Update state: x = x + K * y
            self.state += kalman_gain * innovation;

            // Update covariance: P = (I - K * H) * P
            let identity = na::Matrix4::identity();
            self.covariance = (identity - kalman_gain * self.measurement) * self.covariance;
        }
    }

    fn get_position(&self) -> (f32, f32) {
        (self.state[0], self.state[1])
    }

    fn get_velocity(&self) -> (f32, f32) {
        (self.state[2], self.state[3])
    }
}

/// Tracked object with Kalman filter
struct TrackedObject {
    id: u32,
    class_name: String,
    bbox: BoundingBox,
    confidence: f32,
    kalman: KalmanFilter,
    frames_since_update: u32,
    total_frames: u32,
    last_seen: u64,
    reid_features: Option<Vec<f32>>,  // Last known ReID features for re-identification
    state: InternalTrackState,  // Track lifecycle state (BoTSORT)
    hits: u32,  // Number of consecutive hits for confirming track
}

impl TrackedObject {
    fn new(id: u32, detection: &DetectionResult) -> Self {
        let (cx, cy) = detection.bbox.center();
        let kalman = KalmanFilter::new(cx, cy);

        Self {
            id,
            class_name: detection.class_name.clone(),
            bbox: detection.bbox.clone(),
            confidence: detection.confidence,
            kalman,
            frames_since_update: 0,
            total_frames: 1,
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            reid_features: detection.reid_features.clone(),
            state: InternalTrackState::New,
            hits: 1,
        }
    }

    fn predict(&mut self) {
        self.kalman.predict();
        self.frames_since_update += 1;

        // Update state based on frames since last update
        if self.frames_since_update > 1 {
            self.state = InternalTrackState::Lost;
        }
    }

    fn update(&mut self, detection: &DetectionResult, min_hits: u32) {
        let (cx, cy) = detection.bbox.center();
        self.kalman.update(cx, cy);

        self.bbox = detection.bbox.clone();
        self.confidence = detection.confidence;
        self.frames_since_update = 0;
        self.total_frames += 1;
        self.hits += 1;
        self.last_seen = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Update ReID features if available
        if let Some(ref features) = detection.reid_features {
            self.reid_features = Some(features.clone());
        }

        // Update state: track is active if it has enough hits
        if self.hits >= min_hits {
            self.state = InternalTrackState::Tracked;
        }
    }

    /// Apply camera motion compensation to predicted bbox
    fn apply_camera_motion(&mut self, transform: &na::Matrix3<f32>) {
        let (cx, cy) = self.kalman.get_position();

        // Transform center point
        let p = na::Vector3::new(cx, cy, 1.0);
        let p_transformed = transform * p;

        // Update Kalman state with compensated position
        self.kalman.state[0] = p_transformed[0];
        self.kalman.state[1] = p_transformed[1];

        // Also transform bbox for visualization
        let w = self.bbox.width();
        let h = self.bbox.height();
        let new_cx = p_transformed[0];
        let new_cy = p_transformed[1];

        self.bbox = BoundingBox::new(
            (new_cx - w / 2.0).clamp(0.0, 1.0),
            (new_cy - h / 2.0).clamp(0.0, 1.0),
            (new_cx + w / 2.0).clamp(0.0, 1.0),
            (new_cy + h / 2.0).clamp(0.0, 1.0),
        );
    }

    /// Compute ReID similarity with a detection (cosine similarity)
    fn reid_similarity(&self, detection: &DetectionResult) -> Option<f32> {
        match (&self.reid_features, &detection.reid_features) {
            (Some(f1), Some(f2)) if f1.len() == f2.len() => {
                let dot: f32 = f1.iter().zip(f2.iter()).map(|(a, b)| a * b).sum();
                let norm1: f32 = f1.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm2: f32 = f2.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm1 > 0.0 && norm2 > 0.0 {
                    Some(dot / (norm1 * norm2))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn get_predicted_bbox(&self) -> BoundingBox {
        let (cx, cy) = self.kalman.get_position();
        let w = self.bbox.width();
        let h = self.bbox.height();

        BoundingBox::new(
            (cx - w / 2.0).clamp(0.0, 1.0),
            (cy - h / 2.0).clamp(0.0, 1.0),
            (cx + w / 2.0).clamp(0.0, 1.0),
            (cy + h / 2.0).clamp(0.0, 1.0),
        )
    }

    fn to_tracking_target(&self) -> TrackingTarget {
        TrackingTarget {
            tracking_id: self.id,
            class_name: self.class_name.clone(),
            bbox: self.bbox.clone(),
            last_seen: self.last_seen,
            confidence: self.confidence,
            lost_frames: self.frames_since_update,
        }
    }
}

/// BoTSORT object tracker with ReID and Camera Motion Compensation
struct ObjectTracker {
    tracks: HashMap<u32, TrackedObject>,
    next_id: u32,
    max_age: u32,
    min_hits: u32,
    iou_threshold: f32,
    reid_weight: f32,  // Weight for ReID similarity in matching (0.0 = IoU only, 1.0 = ReID only)
    reid_threshold: f32,  // Minimum ReID similarity for matching
    selected_target_id: Option<u32>,
    tracking_enabled: bool,
    cmc: Option<CameraMotionCompensator>,  // Camera motion compensator (BoTSORT)
    high_conf_threshold: f32,  // Threshold for high-confidence detections (two-stage matching)
}

impl ObjectTracker {
    fn new(
        max_age: u32,
        min_hits: u32,
        iou_threshold: f32,
        reid_weight: f32,
        reid_threshold: f32,
        enable_cmc: bool,
    ) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_age,
            min_hits,
            iou_threshold,
            reid_weight,
            reid_threshold,
            selected_target_id: None,
            tracking_enabled: false,
            cmc: if enable_cmc {
                Some(CameraMotionCompensator::new())
            } else {
                None
            },
            high_conf_threshold: 0.6,  // Detections above this are "high confidence"
        }
    }

    /// Process camera frame for motion compensation
    fn process_frame(&mut self, frame_data: &[u8], width: u32, height: u32) {
        if let Some(ref mut cmc) = self.cmc {
            // Convert to grayscale
            let gray_frame = CameraMotionCompensator::rgb_to_gray(frame_data, width, height);

            // Estimate camera motion
            if let Some(transform) = cmc.estimate_motion(&gray_frame) {
                // Apply motion compensation to all tracks
                for track in self.tracks.values_mut() {
                    track.apply_camera_motion(&transform);
                }
                debug!("CMC: Applied camera motion compensation");
            }
        }
    }

    fn update(&mut self, detections: Vec<DetectionResult>) {
        // Predict all existing tracks
        for track in self.tracks.values_mut() {
            track.predict();
        }

        // Two-stage matching (BoTSORT)
        // Stage 1: Match high-confidence detections with active tracks
        // Stage 2: Match remaining low-confidence detections

        let (high_conf_dets, low_conf_dets): (Vec<_>, Vec<_>) = detections
            .iter()
            .enumerate()
            .partition(|(_, d)| d.confidence >= self.high_conf_threshold);

        debug!("Two-stage matching: {} high-conf, {} low-conf detections",
               high_conf_dets.len(), low_conf_dets.len());

        // Stage 1: High-confidence detections with IoU + ReID
        let mut matched_tracks = std::collections::HashSet::new();
        let mut matched_detections = std::collections::HashSet::new();

        if !high_conf_dets.is_empty() {
            let high_conf_indices: Vec<usize> = high_conf_dets.iter().map(|(idx, _)| *idx).collect();
            let high_conf_subset: Vec<DetectionResult> = high_conf_indices
                .iter()
                .map(|&idx| detections[idx].clone())
                .collect();

            let matches_stage1 = self.associate_detections_to_tracks(&high_conf_subset, true);

            for (subset_idx, track_id) in matches_stage1 {
                let detection_idx = high_conf_indices[subset_idx];
                if let Some(track) = self.tracks.get_mut(&track_id) {
                    track.update(&detections[detection_idx], self.min_hits);
                    matched_tracks.insert(track_id);
                    matched_detections.insert(detection_idx);
                }
            }

            debug!("Stage 1: Matched {} high-confidence detections", matched_detections.len());
        }

        // Stage 2: Low-confidence detections with IoU only
        if !low_conf_dets.is_empty() {
            let low_conf_indices: Vec<usize> = low_conf_dets
                .iter()
                .map(|(idx, _)| *idx)
                .filter(|idx| !matched_detections.contains(idx))
                .collect();

            let low_conf_subset: Vec<DetectionResult> = low_conf_indices
                .iter()
                .map(|&idx| detections[idx].clone())
                .collect();

            let matches_stage2 = self.associate_detections_to_tracks(&low_conf_subset, false);

            for (subset_idx, track_id) in matches_stage2 {
                let detection_idx = low_conf_indices[subset_idx];
                if !matched_tracks.contains(&track_id) {
                    if let Some(track) = self.tracks.get_mut(&track_id) {
                        track.update(&detections[detection_idx], self.min_hits);
                        matched_tracks.insert(track_id);
                        matched_detections.insert(detection_idx);
                    }
                }
            }

            debug!("Stage 2: Matched {} low-confidence detections",
                   matched_detections.len() - high_conf_dets.len());
        }

        // Create new tracks for unmatched high-confidence detections
        for (idx, _) in high_conf_dets {
            if !matched_detections.contains(&idx) {
                let new_track = TrackedObject::new(self.next_id, &detections[idx]);
                self.tracks.insert(self.next_id, new_track);
                self.next_id += 1;
            }
        }

        // Remove old tracks
        let tracks_to_remove: Vec<u32> = self.tracks.iter()
            .filter(|(_, track)| track.frames_since_update > self.max_age)
            .map(|(id, _)| *id)
            .collect();

        for track_id in tracks_to_remove {
            self.tracks.remove(&track_id);

            // Clear selected target if it was removed
            if self.selected_target_id == Some(track_id) {
                self.selected_target_id = None;
                info!("Selected target lost");
            }
        }

        debug!("Active tracks: {} (confirmed: {})",
               self.tracks.len(),
               self.tracks.values().filter(|t| t.state.is_active()).count());
    }

    fn associate_detections_to_tracks(&self, detections: &[DetectionResult], use_reid: bool) -> Vec<(usize, u32)> {
        let mut matches = Vec::new();

        if detections.is_empty() || self.tracks.is_empty() {
            return matches;
        }

        // Compute combined similarity matrix (IoU + ReID)
        let mut similarity_matrix: Vec<Vec<(f32, u32)>> = Vec::new();

        for detection in detections {
            let mut row = Vec::new();
            for (track_id, track) in &self.tracks {
                // Check class match first
                if detection.class_name != track.class_name {
                    row.push((0.0, *track_id));
                    continue;
                }

                // Compute IoU
                let predicted_bbox = track.get_predicted_bbox();
                let iou = detection.bbox.iou(&predicted_bbox);

                // Compute combined similarity based on matching stage
                let similarity = if use_reid && self.reid_weight > 0.0 {
                    // High-confidence stage: use IoU + ReID
                    if let Some(reid_sim) = track.reid_similarity(detection) {
                        // Use weighted combination of IoU and ReID similarity
                        let combined = (1.0 - self.reid_weight) * iou + self.reid_weight * reid_sim;

                        debug!(
                            "Track {} <-> Detection: IoU={:.3}, ReID={:.3}, Combined={:.3}",
                            track_id, iou, reid_sim, combined
                        );

                        combined
                    } else {
                        // No ReID features, fallback to IoU
                        iou
                    }
                } else {
                    // Low-confidence stage: use IoU only
                    iou
                };

                row.push((similarity, *track_id));
            }
            similarity_matrix.push(row);
        }

        // Greedy matching: match highest similarity first
        let mut used_tracks = std::collections::HashSet::new();
        let mut used_detections = std::collections::HashSet::new();

        loop {
            // Determine threshold based on matching stage
            let threshold = if use_reid && self.reid_weight > 0.0 {
                let has_reid = detections.iter().any(|d| d.reid_features.is_some());
                if has_reid {
                    // Use combined threshold for high-confidence + ReID
                    (1.0 - self.reid_weight) * self.iou_threshold + self.reid_weight * self.reid_threshold
                } else {
                    // Fallback to IoU threshold
                    self.iou_threshold
                }
            } else {
                // Use IoU threshold only for low-confidence
                self.iou_threshold * 0.8  // Slightly lower threshold for second stage
            };

            let mut best_similarity = threshold;
            let mut best_detection = None;
            let mut best_track = None;

            for (det_idx, row) in similarity_matrix.iter().enumerate() {
                if used_detections.contains(&det_idx) {
                    continue;
                }

                for (similarity, track_id) in row {
                    if used_tracks.contains(track_id) {
                        continue;
                    }

                    if *similarity > best_similarity {
                        best_similarity = *similarity;
                        best_detection = Some(det_idx);
                        best_track = Some(*track_id);
                    }
                }
            }

            if let (Some(det_idx), Some(track_id)) = (best_detection, best_track) {
                matches.push((det_idx, track_id));
                used_detections.insert(det_idx);
                used_tracks.insert(track_id);
            } else {
                break;
            }
        }

        matches
    }

    fn handle_tracking_command(&mut self, command: TrackingCommand) {
        match command {
            TrackingCommand::Enable { timestamp } => {
                info!("Tracking enabled at {}", timestamp);
                self.tracking_enabled = true;
            }
            TrackingCommand::Disable { timestamp } => {
                info!("Tracking disabled at {}", timestamp);
                self.tracking_enabled = false;
                self.selected_target_id = None;
            }
            TrackingCommand::SelectTarget { detection_index, timestamp } => {
                warn!("SelectTarget by index not yet supported (idx: {}, ts: {})", detection_index, timestamp);
            }
            TrackingCommand::SelectTargetById { tracking_id, timestamp } => {
                if self.tracks.contains_key(&tracking_id) {
                    info!("Selected target ID {} at {}", tracking_id, timestamp);
                    self.selected_target_id = Some(tracking_id);
                    self.tracking_enabled = true;
                } else {
                    warn!("Cannot select target ID {}: not found", tracking_id);
                }
            }
            TrackingCommand::ClearTarget { timestamp } => {
                info!("Cleared target at {}", timestamp);
                self.selected_target_id = None;
            }
        }
    }

    fn get_tracking_telemetry(&self) -> TrackingTelemetry {
        let state = if !self.tracking_enabled {
            TrackingState::Disabled
        } else if let Some(target_id) = self.selected_target_id {
            if let Some(track) = self.tracks.get(&target_id) {
                if track.frames_since_update > self.max_age / 2 {
                    TrackingState::TargetLost
                } else {
                    TrackingState::Tracking
                }
            } else {
                TrackingState::TargetLost
            }
        } else {
            TrackingState::Enabled
        };

        let target = self.selected_target_id
            .and_then(|id| self.tracks.get(&id))
            .map(|track| track.to_tracking_target());

        TrackingTelemetry::new(state, target)
    }

    fn get_all_tracks(&self) -> Vec<DetectionResult> {
        self.tracks.values()
            // Only return confirmed/tracked objects (BoTSORT)
            .filter(|track| track.state.is_active())
            .map(|track| {
                let mut detection = DetectionResult::new(
                    track.bbox.clone(),
                    0, // class_id not used
                    track.class_name.clone(),
                    track.confidence,
                );
                detection.tracking_id = Some(track.id);
                detection.reid_features = track.reid_features.clone();
                detection
            })
            .collect()
    }
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    info!("Starting BoTSORT object_tracker node (with ReID + CMC support)");

    // Read configuration from environment
    let max_age = env::var("MAX_TRACKING_AGE")
        .unwrap_or_else(|_| "30".to_string())
        .parse::<u32>()
        .context("Invalid MAX_TRACKING_AGE")?;

    let min_hits = env::var("MIN_HITS")
        .unwrap_or_else(|_| "3".to_string())
        .parse::<u32>()
        .context("Invalid MIN_HITS")?;

    let iou_threshold = env::var("IOU_THRESHOLD")
        .unwrap_or_else(|_| "0.3".to_string())
        .parse::<f32>()
        .context("Invalid IOU_THRESHOLD")?;

    let reid_weight = env::var("REID_WEIGHT")
        .unwrap_or_else(|_| "0.5".to_string())
        .parse::<f32>()
        .context("Invalid REID_WEIGHT")?;

    let reid_threshold = env::var("REID_THRESHOLD")
        .unwrap_or_else(|_| "0.5".to_string())
        .parse::<f32>()
        .context("Invalid REID_THRESHOLD")?;

    let enable_cmc = env::var("ENABLE_CMC")
        .unwrap_or_else(|_| "true".to_string())
        .parse::<bool>()
        .unwrap_or(true);

    info!("Configuration:");
    info!("  Max tracking age: {} frames", max_age);
    info!("  Min hits: {} frames", min_hits);
    info!("  IoU threshold: {}", iou_threshold);
    info!("  ReID weight: {} (0.0=IoU only, 1.0=ReID only)", reid_weight);
    info!("  ReID threshold: {}", reid_threshold);
    info!("  Camera Motion Compensation: {}", if enable_cmc { "enabled" } else { "disabled" });

    // Initialize tracker
    let mut tracker = ObjectTracker::new(max_age, min_hits, iou_threshold, reid_weight, reid_threshold, enable_cmc);

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;
    info!("Dora node initialized");

    // Main event loop
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata, .. } => {
                match id.as_str() {
                    "frame" => {
                        // Process frame for camera motion compensation
                        if enable_cmc {
                            let width = metadata.parameters.get("width")
                                .and_then(|v| match v {
                                    dora_node_api::Parameter::Integer(i) => Some(*i as u32),
                                    _ => None,
                                })
                                .unwrap_or(640);

                            let height = metadata.parameters.get("height")
                                .and_then(|v| match v {
                                    dora_node_api::Parameter::Integer(i) => Some(*i as u32),
                                    _ => None,
                                })
                                .unwrap_or(480);

                            let frame_data = if let Some(array) = data.as_any().downcast_ref::<UInt8Array>() {
                                array.values().as_ref()
                            } else {
                                error!("Failed to cast frame data to UInt8Array");
                                continue;
                            };

                            tracker.process_frame(frame_data, width, height);
                        }
                    }
                    "detections" => {
                        // Deserialize detection frame
                        let binary_data = if let Some(array) = data.as_any().downcast_ref::<BinaryArray>() {
                            array.value(0)
                        } else {
                            error!("Failed to cast detections to BinaryArray");
                            continue;
                        };

                        let detection_frame: DetectionFrame = match serde_json::from_slice(binary_data) {
                            Ok(frame) => frame,
                            Err(e) => {
                                error!("Failed to deserialize detection frame: {:?}", e);
                                continue;
                            }
                        };

                        debug!("Received {} detections", detection_frame.detections.len());

                        // Update tracker with detections
                        tracker.update(detection_frame.detections.clone());

                        // Send tracking telemetry
                        let telemetry = tracker.get_tracking_telemetry();
                        let telemetry_json = serde_json::to_vec(&telemetry)?;
                        let telemetry_data = BinaryArray::from_vec(vec![telemetry_json.as_slice()]);
                        node.send_output(
                            DataId::from("tracking_telemetry".to_owned()),
                            Default::default(),
                            telemetry_data,
                        )?;

                        // Send updated detection frame with tracking IDs
                        let mut updated_frame = detection_frame;
                        updated_frame.detections = tracker.get_all_tracks();
                        let frame_json = serde_json::to_vec(&updated_frame)?;
                        let frame_data = BinaryArray::from_vec(vec![frame_json.as_slice()]);
                        node.send_output(
                            DataId::from("tracked_detections".to_owned()),
                            Default::default(),
                            frame_data,
                        )?;

                        debug!("Sent tracking update");
                    }
                    "tracking_command" | "tracking_command_voice" => {
                        // Deserialize tracking command
                        let binary_data = if let Some(array) = data.as_any().downcast_ref::<BinaryArray>() {
                            array.value(0)
                        } else {
                            error!("Failed to cast tracking_command to BinaryArray");
                            continue;
                        };

                        let command: TrackingCommand = match serde_json::from_slice(binary_data) {
                            Ok(cmd) => cmd,
                            Err(e) => {
                                error!("Failed to deserialize tracking command: {:?}", e);
                                continue;
                            }
                        };

                        let source = match id.as_str() {
                            "tracking_command_voice" => "voice",
                            "tracking_command_reid" => "re-id",
                            _ => "web",
                        };
                        debug!("Received {} tracking command: {:?}", source, command);
                        tracker.handle_tracking_command(command);

                        // Send updated tracking telemetry immediately after command
                        let telemetry = tracker.get_tracking_telemetry();
                        let telemetry_json = serde_json::to_vec(&telemetry)?;
                        let telemetry_data = BinaryArray::from_vec(vec![telemetry_json.as_slice()]);
                        node.send_output(
                            DataId::from("tracking_telemetry".to_owned()),
                            Default::default(),
                            telemetry_data,
                        )?;
                        debug!("Sent tracking telemetry after {} command", source);
                    }
                    other => {
                        warn!("Received unexpected input: {}", other);
                    }
                }
            }
            Event::InputClosed { id } => {
                info!("Input {} closed", id);
                break;
            }
            Event::Stop(_) => {
                info!("Received stop signal");
                break;
            }
            other => {
                debug!("Received other event: {:?}", other);
            }
        }
    }

    info!("Object tracker node shutting down");
    Ok(())
}
