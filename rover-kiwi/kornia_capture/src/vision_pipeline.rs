use eyre::Result;
use object_detector::{DetectorConfig, YoloDetector};
use object_tracker::{ObjectTracker, TrackerConfig};
use reid_extractor::{ReIdConfig, ReIdExtractor};
use robo_rover_lib::types::{DetectionFrame, TrackingCommand, TrackingTelemetry};
use std::sync::Arc;
use tracing::info;

pub enum PipelineOutput {
    CameraOnly,
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

    pipeline_enabled: bool,

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
            pipeline_enabled: false,
            detector_config,
            reid_config,
        }
    }

    /// Process one camera frame. Returns CameraOnly when pipeline is disabled.
    /// On model error: auto-disables pipeline, returns CameraOnly (camera keeps streaming).
    pub fn process_frame(&mut self, frame: &[u8], w: u32, h: u32) -> Result<PipelineOutput> {
        if !self.pipeline_enabled {
            return Ok(PipelineOutput::CameraOnly);
        }

        if let Err(e) = self.ensure_models_loaded() {
            tracing::error!("Failed to load ML models: {:?} — disabling pipeline", e);
            self.pipeline_enabled = false;
            return Ok(PipelineOutput::CameraOnly);
        }

        let detector = self.detector.as_mut().unwrap();
        let reid = self.reid.as_mut().unwrap();

        let detection_frame = match detector.detect(frame, w, h) {
            Ok(f) => f,
            Err(e) => {
                tracing::error!("YOLO detect failed: {:?} — disabling pipeline", e);
                self.pipeline_enabled = false;
                return Ok(PipelineOutput::CameraOnly);
            }
        };

        let enriched = match reid.process_detections(frame, w, h, detection_frame) {
            Ok(f) => f,
            Err(e) => {
                tracing::error!("ReID failed: {:?} — disabling pipeline", e);
                self.pipeline_enabled = false;
                return Ok(PipelineOutput::CameraOnly);
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
            TrackingCommand::Enable { .. } => {
                info!("Vision pipeline enabled (tracking on)");
                self.pipeline_enabled = true;
            }
            TrackingCommand::Disable { .. } => {
                info!("Vision pipeline disabled (tracking off)");
                self.pipeline_enabled = false;
                // Models stay loaded; tracker state (tracks HashMap) preserved.
                // Intentional: avoids cold start + stale-track-ID churn on re-enable.
            }
            _ => {} // SelectTarget / ClearTarget — delegated to tracker below
        }
        self.tracker.handle_tracking_command(cmd);
    }

    fn ensure_models_loaded(&mut self) -> Result<()> {
        if self.detector.is_some() && self.reid.is_some() {
            return Ok(());
        }

        info!("Loading ML models (first enable)...");

        let env = match &self.ort_env {
            Some(env) => env.clone(),
            None => {
                let env = ort::Environment::builder()
                    .with_name("vision_pipeline")
                    .with_execution_providers([ort::ExecutionProvider::CPU(Default::default())])
                    .build()?
                    .into_arc();
                self.ort_env = Some(env.clone());
                env
            }
        };

        if self.detector.is_none() {
            self.detector =
                Some(YoloDetector::new(env.clone(), self.detector_config.clone())?);
            info!("YOLO detector loaded");
        }

        if self.reid.is_none() {
            self.reid = Some(ReIdExtractor::new(env, self.reid_config.clone())?);
            info!("ReID extractor loaded");
        }

        Ok(())
    }
}
