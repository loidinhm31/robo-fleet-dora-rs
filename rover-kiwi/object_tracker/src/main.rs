use dora_node_api::{
    dora_core::config::DataId,
    DoraNode,
    Event,
    arrow::array::BinaryArray,
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
        }
    }

    fn predict(&mut self) {
        self.kalman.predict();
        self.frames_since_update += 1;
    }

    fn update(&mut self, detection: &DetectionResult) {
        let (cx, cy) = detection.bbox.center();
        self.kalman.update(cx, cy);

        self.bbox = detection.bbox.clone();
        self.confidence = detection.confidence;
        self.frames_since_update = 0;
        self.total_frames += 1;
        self.last_seen = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
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

/// SORT-based object tracker
struct ObjectTracker {
    tracks: HashMap<u32, TrackedObject>,
    next_id: u32,
    max_age: u32,
    min_hits: u32,
    iou_threshold: f32,
    selected_target_id: Option<u32>,
    tracking_enabled: bool,
}

impl ObjectTracker {
    fn new(max_age: u32, min_hits: u32, iou_threshold: f32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_age,
            min_hits,
            iou_threshold,
            selected_target_id: None,
            tracking_enabled: false,
        }
    }

    fn update(&mut self, detections: Vec<DetectionResult>) {
        // Predict all existing tracks
        for track in self.tracks.values_mut() {
            track.predict();
        }

        // Match detections to tracks using Hungarian algorithm (simplified with greedy matching)
        let matched_pairs = self.associate_detections_to_tracks(&detections);

        // Update matched tracks
        let mut unmatched_detections: Vec<usize> = (0..detections.len()).collect();
        let mut matched_tracks = std::collections::HashSet::new();

        for (detection_idx, track_id) in matched_pairs {
            if let Some(track) = self.tracks.get_mut(&track_id) {
                track.update(&detections[detection_idx]);
                matched_tracks.insert(track_id);
                unmatched_detections.retain(|&idx| idx != detection_idx);
            }
        }

        // Create new tracks for unmatched detections
        for detection_idx in unmatched_detections {
            let new_track = TrackedObject::new(self.next_id, &detections[detection_idx]);
            self.tracks.insert(self.next_id, new_track);
            self.next_id += 1;
        }

        // Remove old tracks
        let tracks_to_remove: Vec<u32> = self.tracks.iter()
            .filter(|(_, track)| track.frames_since_update > self.max_age)
            .map(|(id, _)| *id)
            .collect();

        for track_id in tracks_to_remove {
            // info!("Removing track {} (lost for {} frames)", track_id, self.max_age);
            self.tracks.remove(&track_id);

            // Clear selected target if it was removed
            if self.selected_target_id == Some(track_id) {
                self.selected_target_id = None;
                info!("Selected target lost");
            }
        }

        debug!("Active tracks: {}", self.tracks.len());
    }

    fn associate_detections_to_tracks(&self, detections: &[DetectionResult]) -> Vec<(usize, u32)> {
        let mut matches = Vec::new();

        if detections.is_empty() || self.tracks.is_empty() {
            return matches;
        }

        // Compute IoU matrix
        let mut iou_matrix: Vec<Vec<(f32, u32)>> = Vec::new();

        for detection in detections {
            let mut row = Vec::new();
            for (track_id, track) in &self.tracks {
                let predicted_bbox = track.get_predicted_bbox();
                let iou = detection.bbox.iou(&predicted_bbox);

                // Also check class match
                if detection.class_name == track.class_name {
                    row.push((iou, *track_id));
                } else {
                    row.push((0.0, *track_id));
                }
            }
            iou_matrix.push(row);
        }

        // Greedy matching: match highest IoU first
        let mut used_tracks = std::collections::HashSet::new();
        let mut used_detections = std::collections::HashSet::new();

        loop {
            let mut best_iou = self.iou_threshold;
            let mut best_detection = None;
            let mut best_track = None;

            for (det_idx, row) in iou_matrix.iter().enumerate() {
                if used_detections.contains(&det_idx) {
                    continue;
                }

                for (iou, track_id) in row {
                    if used_tracks.contains(track_id) {
                        continue;
                    }

                    if *iou > best_iou {
                        best_iou = *iou;
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
            .filter(|track| track.total_frames >= self.min_hits)
            .map(|track| {
                let mut detection = DetectionResult::new(
                    track.bbox.clone(),
                    0, // class_id not used
                    track.class_name.clone(),
                    track.confidence,
                );
                detection.tracking_id = Some(track.id);
                detection
            })
            .collect()
    }
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    info!("Starting object_tracker node");

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

    info!("Configuration:");
    info!("  Max tracking age: {} frames", max_age);
    info!("  Min hits: {} frames", min_hits);
    info!("  IoU threshold: {}", iou_threshold);

    // Initialize tracker
    let mut tracker = ObjectTracker::new(max_age, min_hits, iou_threshold);

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;
    info!("Dora node initialized");

    // Main event loop
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } => {
                match id.as_str() {
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
