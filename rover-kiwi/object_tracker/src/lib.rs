pub mod cmc;
pub mod kalman;
pub mod tracked_object;

use cmc::CameraMotionCompensator;
use std::collections::{HashMap, HashSet};
use tracked_object::TrackedObject;
use tracing::{debug, info, warn};

use robo_rover_lib::types::{DetectionResult, TrackingCommand, TrackingState, TrackingTelemetry};

pub struct TrackerConfig {
    pub max_age: u32,
    pub min_hits: u32,
    pub iou_threshold: f32,
    pub reid_weight: f32,
    pub reid_threshold: f32,
    pub enable_cmc: bool,
}

pub struct ObjectTracker {
    tracks: HashMap<u32, TrackedObject>,
    next_id: u32,
    max_age: u32,
    min_hits: u32,
    iou_threshold: f32,
    reid_weight: f32,
    reid_threshold: f32,
    selected_target_id: Option<u32>,
    tracking_enabled: bool,
    cmc: Option<CameraMotionCompensator>,
    high_conf_threshold: f32,
}

impl ObjectTracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_age: config.max_age,
            min_hits: config.min_hits,
            iou_threshold: config.iou_threshold,
            reid_weight: config.reid_weight,
            reid_threshold: config.reid_threshold,
            selected_target_id: None,
            tracking_enabled: false,
            cmc: if config.enable_cmc {
                Some(CameraMotionCompensator::new())
            } else {
                None
            },
            high_conf_threshold: 0.6,
        }
    }

    /// Run camera motion compensation on the raw frame.
    pub fn process_frame(&mut self, frame_data: &[u8], width: u32, height: u32) {
        if let Some(ref mut cmc) = self.cmc {
            let gray = CameraMotionCompensator::rgb_to_gray(frame_data, width, height);
            if let Some(transform) = cmc.estimate_motion(&gray) {
                for track in self.tracks.values_mut() {
                    track.apply_camera_motion(&transform);
                }
                debug!("CMC: applied camera motion compensation");
            }
        }
    }

    pub fn update(&mut self, detections: Vec<DetectionResult>) {
        for track in self.tracks.values_mut() {
            track.predict();
        }

        let (high_conf_dets, low_conf_dets): (Vec<_>, Vec<_>) = detections
            .iter()
            .enumerate()
            .partition(|(_, d)| d.confidence >= self.high_conf_threshold);

        debug!(
            "Two-stage matching: {} high-conf, {} low-conf",
            high_conf_dets.len(), low_conf_dets.len()
        );

        let mut matched_tracks: HashSet<u32> = HashSet::new();
        let mut matched_detections: HashSet<usize> = HashSet::new();

        // Stage 1: high-confidence detections with IoU + ReID
        if !high_conf_dets.is_empty() {
            let hc_indices: Vec<usize> = high_conf_dets.iter().map(|(idx, _)| *idx).collect();
            let hc_subset: Vec<DetectionResult> =
                hc_indices.iter().map(|&idx| detections[idx].clone()).collect();

            for (subset_idx, track_id) in self.associate_detections_to_tracks(&hc_subset, true) {
                let det_idx = hc_indices[subset_idx];
                if let Some(track) = self.tracks.get_mut(&track_id) {
                    track.update(&detections[det_idx], self.min_hits);
                    matched_tracks.insert(track_id);
                    matched_detections.insert(det_idx);
                }
            }
        }

        // Stage 2: low-confidence detections with IoU only
        if !low_conf_dets.is_empty() {
            let lc_indices: Vec<usize> = low_conf_dets
                .iter()
                .map(|(idx, _)| *idx)
                .filter(|idx| !matched_detections.contains(idx))
                .collect();

            let lc_subset: Vec<DetectionResult> =
                lc_indices.iter().map(|&idx| detections[idx].clone()).collect();

            for (subset_idx, track_id) in self.associate_detections_to_tracks(&lc_subset, false) {
                let det_idx = lc_indices[subset_idx];
                if !matched_tracks.contains(&track_id) {
                    if let Some(track) = self.tracks.get_mut(&track_id) {
                        track.update(&detections[det_idx], self.min_hits);
                        matched_tracks.insert(track_id);
                        matched_detections.insert(det_idx);
                    }
                }
            }
        }

        // New tracks for unmatched high-confidence detections
        for (idx, _) in high_conf_dets {
            if !matched_detections.contains(&idx) {
                let new_track = TrackedObject::new(self.next_id, &detections[idx]);
                self.tracks.insert(self.next_id, new_track);
                self.next_id += 1;
            }
        }

        // Evict stale tracks
        let stale: Vec<u32> = self
            .tracks
            .iter()
            .filter(|(_, t)| t.frames_since_update > self.max_age)
            .map(|(id, _)| *id)
            .collect();

        for track_id in stale {
            self.tracks.remove(&track_id);
            if self.selected_target_id == Some(track_id) {
                self.selected_target_id = None;
                info!("Selected target {} lost (evicted)", track_id);
            }
        }

        debug!(
            "Active tracks: {} (confirmed: {})",
            self.tracks.len(),
            self.tracks.values().filter(|t| t.state.is_active()).count()
        );
    }

    pub fn handle_tracking_command(&mut self, command: TrackingCommand) {
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
                warn!("SelectTarget by index not supported (idx: {}, ts: {})", detection_index, timestamp);
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

    pub fn get_tracking_telemetry(&self) -> TrackingTelemetry {
        let state = if !self.tracking_enabled {
            TrackingState::Disabled
        } else if let Some(target_id) = self.selected_target_id {
            match self.tracks.get(&target_id) {
                Some(track) if track.frames_since_update > self.max_age / 2 => {
                    TrackingState::TargetLost
                }
                Some(_) => TrackingState::Tracking,
                None => TrackingState::TargetLost,
            }
        } else {
            TrackingState::Enabled
        };

        let target = self
            .selected_target_id
            .and_then(|id| self.tracks.get(&id))
            .map(|t| t.to_tracking_target());

        TrackingTelemetry::new(state, target)
    }

    pub fn get_all_tracks(&self) -> Vec<DetectionResult> {
        self.tracks
            .values()
            .filter(|t| t.state.is_active())
            .map(|track| {
                let mut det = DetectionResult::new(
                    track.bbox.clone(),
                    0,
                    track.class_name.clone(),
                    track.confidence,
                );
                det.tracking_id = Some(track.id);
                det.reid_features = track.reid_features.clone();
                det
            })
            .collect()
    }

    fn associate_detections_to_tracks(
        &self,
        detections: &[DetectionResult],
        use_reid: bool,
    ) -> Vec<(usize, u32)> {
        if detections.is_empty() || self.tracks.is_empty() {
            return Vec::new();
        }

        // Build similarity matrix: [detection_idx] -> Vec<(similarity, track_id)>
        let similarity_matrix: Vec<Vec<(f32, u32)>> = detections
            .iter()
            .map(|detection| {
                self.tracks
                    .iter()
                    .map(|(track_id, track)| {
                        if detection.class_name != track.class_name {
                            return (0.0, *track_id);
                        }

                        let iou = detection.bbox.iou(&track.get_predicted_bbox());

                        let similarity = if use_reid && self.reid_weight > 0.0 {
                            if let Some(reid_sim) = track.reid_similarity(detection) {
                                (1.0 - self.reid_weight) * iou + self.reid_weight * reid_sim
                            } else {
                                iou
                            }
                        } else {
                            iou
                        };

                        (similarity, *track_id)
                    })
                    .collect()
            })
            .collect();

        let threshold = if use_reid && self.reid_weight > 0.0 {
            let has_reid = detections.iter().any(|d| d.reid_features.is_some());
            if has_reid {
                (1.0 - self.reid_weight) * self.iou_threshold
                    + self.reid_weight * self.reid_threshold
            } else {
                self.iou_threshold
            }
        } else {
            self.iou_threshold * 0.8
        };

        let mut matches = Vec::new();
        let mut used_tracks: HashSet<u32> = HashSet::new();
        let mut used_detections: HashSet<usize> = HashSet::new();

        loop {
            let mut best_sim = threshold;
            let mut best_det = None;
            let mut best_track = None;

            for (det_idx, row) in similarity_matrix.iter().enumerate() {
                if used_detections.contains(&det_idx) {
                    continue;
                }
                for (sim, track_id) in row {
                    if used_tracks.contains(track_id) {
                        continue;
                    }
                    if *sim > best_sim {
                        best_sim = *sim;
                        best_det = Some(det_idx);
                        best_track = Some(*track_id);
                    }
                }
            }

            match (best_det, best_track) {
                (Some(det_idx), Some(track_id)) => {
                    matches.push((det_idx, track_id));
                    used_detections.insert(det_idx);
                    used_tracks.insert(track_id);
                }
                _ => break,
            }
        }

        matches
    }
}
