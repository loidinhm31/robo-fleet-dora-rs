use eyre::Result;
use object_detector::{DetectorConfig, YoloDetector};
use object_tracker::{ObjectTracker, TrackerConfig};
use reid_extractor::{ReIdConfig, ReIdExtractor};
use robo_rover_lib::types::{DetectionFrame, TrackingCommand, TrackingState, TrackingTelemetry};
use std::sync::Arc;
use tracing::info;

pub enum PipelineOutput {
    /// Camera running, ML pipeline disabled. Carries telemetry with state=Disabled
    /// so web UI badge updates immediately when detection is toggled off.
    CameraOnly { tracking_telemetry: TrackingTelemetry },
    /// YOLO detections without tracking IDs. Carries telemetry with state=DetectionOnly
    /// so web UI badge reflects current pipeline mode.
    DetectionOnly {
        detections: DetectionFrame,
        tracking_telemetry: TrackingTelemetry,
    },
    FullTracking {
        tracked_detections: DetectionFrame,
        tracking_telemetry: TrackingTelemetry,
    },
}

pub struct VisionPipeline {
    ort_env: Option<Arc<ort::Environment>>,
    detector: Option<YoloDetector>,
    reid: Option<ReIdExtractor>,
    tracker: ObjectTracker,

    detection_enabled: bool,
    tracking_enabled: bool,

    detector_config: DetectorConfig,
    reid_config: ReIdConfig,
}

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
            detection_enabled: false,
            tracking_enabled: false,
            detector_config,
            reid_config,
        }
    }

    /// Process one camera frame. State machine:
    /// - !detection && !tracking → CameraOnly
    /// - detection && !tracking  → YOLO only → DetectionOnly
    /// - tracking (any)          → YOLO + ReID + BoTSORT → FullTracking
    pub fn process_frame(&mut self, frame: &[u8], w: u32, h: u32) -> Result<PipelineOutput> {
        if !self.detection_enabled {
            return Ok(PipelineOutput::CameraOnly {
                tracking_telemetry: self.tracker.get_tracking_telemetry(),
            });
        }

        if let Err(e) = self.ensure_detector_loaded() {
            tracing::error!("Failed to load YOLO: {:?} — disabling detection", e);
            self.detection_enabled = false;
            self.tracking_enabled = false;
            return Ok(PipelineOutput::CameraOnly {
                tracking_telemetry: self.tracker.get_tracking_telemetry(),
            });
        }

        let detector = self.detector.as_mut().unwrap();
        let detection_frame = match detector.detect(frame, w, h) {
            Ok(f) => f,
            Err(e) => {
                tracing::error!("YOLO detect failed: {:?} — disabling detection", e);
                self.detection_enabled = false;
                self.tracking_enabled = false;
                return Ok(PipelineOutput::CameraOnly {
                    tracking_telemetry: self.tracker.get_tracking_telemetry(),
                });
            }
        };

        if !self.tracking_enabled {
            return Ok(PipelineOutput::DetectionOnly {
                detections: detection_frame,
                tracking_telemetry: self.detection_only_telemetry(),
            });
        }

        // Full pipeline: ReID + BoTSORT
        if let Err(e) = self.ensure_reid_loaded() {
            tracing::error!("Failed to load ReID: {:?} — falling back to detection-only", e);
            self.tracking_enabled = false;
            // Return already-computed YOLO frame — no re-detection
            return Ok(PipelineOutput::DetectionOnly {
                detections: detection_frame,
                tracking_telemetry: self.detection_only_telemetry(),
            });
        }

        let reid = self.reid.as_mut().unwrap();
        let enriched = match reid.process_detections(frame, w, h, detection_frame) {
            Ok(f) => f,
            Err(e) => {
                tracing::error!("ReID failed: {:?} — falling back to detection-only", e);
                self.tracking_enabled = false;
                // Telemetry reflects degradation; detections lost for this frame (ReID consumed it).
                // Acceptable trade-off: avoids re-running YOLO under persistent ReID failure.
                return Ok(PipelineOutput::DetectionOnly {
                    detections: DetectionFrame::new(0, w, h, vec![]),
                    tracking_telemetry: self.detection_only_telemetry(),
                });
            }
        };

        self.tracker.process_frame(frame, w, h); // CMC
        self.tracker.update(enriched.detections.clone());

        let tracking_telemetry = self.tracker.get_tracking_telemetry();
        let mut tracked_frame = enriched;
        tracked_frame.detections = self.tracker.get_all_tracks();

        Ok(PipelineOutput::FullTracking {
            tracked_detections: tracked_frame,
            tracking_telemetry,
        })
    }

    pub fn handle_tracking_command(&mut self, cmd: TrackingCommand) {
        match &cmd {
            TrackingCommand::EnableDetection { .. } => {
                info!("Detection enabled (detection-only mode)");
                self.detection_enabled = true;
                // tracking stays as-is
            }
            // Disables both detection AND tracking (full pipeline off → camera-only).
            // Named DisableDetection because detection is the prerequisite for tracking;
            // disabling detection implicitly disables tracking.
            TrackingCommand::DisableDetection { .. } => {
                info!("Detection disabled → camera-only");
                self.detection_enabled = false;
                self.tracking_enabled = false;
            }
            TrackingCommand::Enable { .. } => {
                info!("Full tracking enabled");
                self.detection_enabled = true;
                self.tracking_enabled = true;
            }
            TrackingCommand::Disable { .. } => {
                info!("Tracking disabled → detection-only");
                // detection stays on; only tracking disabled (progressive degradation)
                self.tracking_enabled = false;
            }
            _ => {} // SelectTarget / ClearTarget / SelectTargetById — delegated to tracker
        }
        self.tracker.handle_tracking_command(cmd);
    }

    fn detection_only_telemetry(&self) -> TrackingTelemetry {
        let mut telemetry = self.tracker.get_tracking_telemetry();
        // Override state: tracker returns Disabled when tracking_enabled=false,
        // but pipeline is in detection-only mode (YOLO active, ReID/tracker off).
        telemetry.state = TrackingState::DetectionOnly;
        telemetry
    }

    fn ensure_detector_loaded(&mut self) -> Result<()> {
        if self.detector.is_some() {
            return Ok(());
        }

        let env = self.get_or_init_ort_env()?;
        info!("Loading YOLO detector (first enable)...");
        self.detector = Some(YoloDetector::new(env, self.detector_config.clone())?);
        info!("YOLO detector loaded");
        Ok(())
    }

    fn ensure_reid_loaded(&mut self) -> Result<()> {
        if self.reid.is_some() {
            return Ok(());
        }

        let env = self.get_or_init_ort_env()?;
        info!("Loading ReID extractor...");
        self.reid = Some(ReIdExtractor::new(env, self.reid_config.clone())?);
        info!("ReID extractor loaded");
        Ok(())
    }

    fn get_or_init_ort_env(&mut self) -> Result<Arc<ort::Environment>> {
        match &self.ort_env {
            Some(env) => Ok(env.clone()),
            None => {
                let env = ort::Environment::builder()
                    .with_name("vision_pipeline")
                    .with_execution_providers([ort::ExecutionProvider::CPU(Default::default())])
                    .build()?
                    .into_arc();
                self.ort_env = Some(env.clone());
                Ok(env)
            }
        }
    }
}
