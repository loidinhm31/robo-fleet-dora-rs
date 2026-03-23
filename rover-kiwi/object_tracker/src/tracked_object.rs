use crate::kalman::KalmanFilter;
use nalgebra as na;
use robo_rover_lib::types::{cosine_similarity, BoundingBox, DetectionResult, TrackingTarget};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq)]
pub enum InternalTrackState {
    New,
    Tracked,
    Lost,
}

impl InternalTrackState {
    pub fn is_active(&self) -> bool {
        matches!(self, InternalTrackState::Tracked)
    }
}

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
    pub fn new(id: u32, detection: &DetectionResult) -> Self {
        let (cx, cy) = detection.bbox.center();

        Self {
            id,
            class_name: detection.class_name.clone(),
            bbox: detection.bbox.clone(),
            confidence: detection.confidence,
            kalman: KalmanFilter::new(cx, cy),
            frames_since_update: 0,
            total_frames: 1,
            last_seen: now_millis(),
            reid_features: detection.reid_features.clone(),
            state: InternalTrackState::New,
            hits: 1,
        }
    }

    pub fn predict(&mut self) {
        self.kalman.predict();
        self.frames_since_update += 1;

        if self.frames_since_update > 1 {
            self.state = InternalTrackState::Lost;
        }
    }

    pub fn update(&mut self, detection: &DetectionResult, min_hits: u32) {
        let (cx, cy) = detection.bbox.center();
        self.kalman.update(cx, cy);

        self.bbox = detection.bbox.clone();
        self.confidence = detection.confidence;
        self.frames_since_update = 0;
        self.total_frames += 1;
        self.hits += 1;
        self.last_seen = now_millis();

        if let Some(ref features) = detection.reid_features {
            self.reid_features = Some(features.clone());
        }

        if self.hits >= min_hits {
            self.state = InternalTrackState::Tracked;
        }
    }

    pub fn apply_camera_motion(&mut self, transform: &na::Matrix3<f32>) {
        let (cx, cy) = self.kalman.get_position();
        let p = na::Vector3::new(cx, cy, 1.0);
        let p_t = transform * p;

        self.kalman.state[0] = p_t[0];
        self.kalman.state[1] = p_t[1];

        let w = self.bbox.width();
        let h = self.bbox.height();

        self.bbox = BoundingBox::new(
            (p_t[0] - w / 2.0).clamp(0.0, 1.0),
            (p_t[1] - h / 2.0).clamp(0.0, 1.0),
            (p_t[0] + w / 2.0).clamp(0.0, 1.0),
            (p_t[1] + h / 2.0).clamp(0.0, 1.0),
        );
    }

    pub fn reid_similarity(&self, detection: &DetectionResult) -> Option<f32> {
        cosine_similarity(
            self.reid_features.as_deref()?,
            detection.reid_features.as_deref()?,
        )
    }

    pub fn get_predicted_bbox(&self) -> BoundingBox {
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

    pub fn to_tracking_target(&self) -> TrackingTarget {
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

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
