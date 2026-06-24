mod pipeline_metrics;
mod vision_pipeline;

use dora_node_api::{
    self,
    arrow::array::{Array as ArrowArray, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event, Parameter,
};
use kornia_io::gstreamer::{CameraCapture, RTSPCameraConfig, V4L2CameraConfig};
use object_detector::DetectorConfig;
use object_tracker::TrackerConfig;
use pipeline_metrics::PipelineMetricWindows;
use reid_extractor::ReIdConfig;
use robo_rover_lib::{
    init_tracing, types::TrackingCommand, CameraAction, CameraControl, MetricWindow, StreamCommand,
    StreamControl,
};
use std::{
    env,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use vision_pipeline::{PipelineOutput, ProcessedPipelineOutput, VisionPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing();
    tracing::info!("Starting kornia_capture node");

    let source_type = env::var("SOURCE_TYPE").map_err(|e| format!("SOURCE_TYPE: {e}"))?;
    let source_uri = env::var("SOURCE_URI").map_err(|e| format!("SOURCE_URI: {e}"))?;

    // Build camera based on source type
    let mut camera_opt: Option<CameraCapture> = Some(match source_type.as_str() {
        "webcam" => {
            let cols = env::var("IMAGE_COLS")
                .map_err(|e| format!("IMAGE_COLS: {e}"))?
                .parse::<usize>()?;
            let rows = env::var("IMAGE_ROWS")
                .map_err(|e| format!("IMAGE_ROWS: {e}"))?
                .parse::<usize>()?;
            let fps = env::var("SOURCE_FPS")
                .map_err(|e| format!("SOURCE_FPS: {e}"))?
                .parse::<u32>()?;

            V4L2CameraConfig::new()
                .with_size([cols, rows].into())
                .with_fps(fps)
                .with_device(&source_uri)
                .build()?
        }
        "rtsp" => RTSPCameraConfig::new().with_url(&source_uri).build()?,
        _ => return Err(format!("Invalid SOURCE_TYPE: {source_type}").into()),
    });

    camera_opt.as_ref().unwrap().start()?;
    tracing::info!("{source_type} camera started");

    let mut pipeline = build_pipeline()?;

    let frame_output = DataId::from("frame".to_owned());
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut frame_id = 0u64;
    let mut last_capture = None;
    let view_stream_fps = env::var("VIEW_STREAM_FPS")
        .unwrap_or_else(|_| "15".into())
        .parse::<u32>()?
        .clamp(1, 120);
    let view_source_fps = env::var("SOURCE_FPS")
        .unwrap_or_else(|_| "30".into())
        .parse::<u32>()?
        .clamp(1, 240);
    let mut view_frame_cadence =
        ViewFrameCadence::new(view_stream_fps, &source_type, view_source_fps);
    let mut view_output = ViewOutputGate::default();
    let mut capture_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut capture_interval_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut vision_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut view_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut pipeline_metrics = PipelineMetricWindows::new();

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => match id.as_str() {
                "tick" => {
                    let capture_started = Instant::now();
                    let Some(ref mut camera) = camera_opt else {
                        capture_metrics.record_drop();
                        continue;
                    };
                    let Some(frame) = camera.grab_rgb8()? else {
                        capture_metrics.record_drop();
                        continue;
                    };
                    let capture_duration = capture_started.elapsed();

                    let width = frame.size().width as u32;
                    let height = frame.size().height as u32;
                    frame_id = frame_id.saturating_add(1);
                    let capture_timestamp_ms = unix_timestamp_ms()?;
                    let capture_interval = last_capture
                        .replace(Instant::now())
                        .map(|previous| previous.elapsed())
                        .unwrap_or_default();
                    if !capture_interval.is_zero() {
                        capture_interval_metrics.record(capture_interval, 0);
                    }

                    let mut params = metadata.parameters;
                    params.insert("encoding".into(), Parameter::String("RGB8".into()));
                    params.insert("height".into(), Parameter::Integer(height as i64));
                    params.insert("width".into(), Parameter::Integer(width as i64));
                    params.insert("frame_id".into(), Parameter::Integer(frame_id as i64));
                    params.insert(
                        "capture_timestamp_ms".into(),
                        Parameter::Integer(capture_timestamp_ms as i64),
                    );

                    if view_output.is_enabled() && view_frame_cadence.should_emit_at(Instant::now())
                    {
                        node.send_output_bytes(
                            frame_output.clone(),
                            params.clone(),
                            frame.numel(),
                            frame.as_slice(),
                        )?;
                        view_metrics.record(capture_started.elapsed(), frame.numel());
                    } else {
                        view_metrics.record_drop();
                    }

                    // Vision pipeline — conditionally sends tracked_detections + tracking_telemetry
                    let vision_started = Instant::now();
                    let timings = send_pipeline_output(
                        &mut pipeline,
                        frame_id,
                        frame.as_slice(),
                        width,
                        height,
                        &mut node,
                    )?;
                    pipeline_metrics.record(frame_id, timings);
                    vision_metrics.record(vision_started.elapsed(), frame.numel());
                    capture_metrics.record(capture_duration, frame.numel());
                    if let Some(snapshot) = capture_metrics.snapshot_if_due() {
                        tracing::info!(
                            metric = "video_pipeline",
                            stage = "capture",
                            frame_id,
                            count = snapshot.count,
                            bytes = snapshot.bytes,
                            drops = snapshot.drops,
                            errors = snapshot.errors,
                            p50_us = snapshot.p50_us,
                            p95_us = snapshot.p95_us,
                            p99_us = snapshot.p99_us,
                            max_us = snapshot.max_us
                        );
                    }
                    if let Some(snapshot) = capture_interval_metrics.snapshot_if_due() {
                        tracing::info!(
                            metric = "video_pipeline",
                            stage = "capture_interval",
                            frame_id,
                            count = snapshot.count,
                            bytes = snapshot.bytes,
                            drops = snapshot.drops,
                            errors = snapshot.errors,
                            p50_us = snapshot.p50_us,
                            p95_us = snapshot.p95_us,
                            p99_us = snapshot.p99_us,
                            max_us = snapshot.max_us
                        );
                    }
                    if let Some(snapshot) = vision_metrics.snapshot_if_due() {
                        tracing::info!(
                            metric = "video_pipeline",
                            stage = "vision_total",
                            frame_id,
                            count = snapshot.count,
                            bytes = snapshot.bytes,
                            drops = snapshot.drops,
                            errors = snapshot.errors,
                            p50_us = snapshot.p50_us,
                            p95_us = snapshot.p95_us,
                            p99_us = snapshot.p99_us,
                            max_us = snapshot.max_us
                        );
                    }
                    if let Some(snapshot) = view_metrics.snapshot_if_due() {
                        tracing::info!(
                            metric = "video_pipeline",
                            stage = "view_branch_emit",
                            target_fps = view_stream_fps,
                            frame_id,
                            count = snapshot.count,
                            bytes = snapshot.bytes,
                            drops = snapshot.drops,
                            errors = snapshot.errors,
                            p50_us = snapshot.p50_us,
                            p95_us = snapshot.p95_us,
                            p99_us = snapshot.p99_us,
                            max_us = snapshot.max_us
                        );
                    }
                }
                "camera_control" | "camera_control_voice" => {
                    let source = if id.as_str() == "camera_control_voice" {
                        "voice"
                    } else {
                        "web"
                    };

                    if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                        if binary_array.len() > 0 {
                            match serde_json::from_slice::<CameraControl>(binary_array.value(0)) {
                                Ok(ctrl) => {
                                    tracing::info!(
                                        "Camera control from {source}: {:?}",
                                        ctrl.command
                                    );
                                    match ctrl.command {
                                        CameraAction::Start => {
                                            if camera_opt.is_none() {
                                                let cam = build_camera(&source_type, &source_uri)?;
                                                cam.start()?;
                                                camera_opt = Some(cam);
                                                tracing::info!("Camera restarted");
                                            }
                                        }
                                        CameraAction::Stop => {
                                            if let Some(cam) = camera_opt.take() {
                                                cam.close()?;
                                                tracing::info!("Camera stopped");
                                            }
                                        }
                                    }
                                }
                                Err(e) => tracing::error!("Failed to parse CameraControl: {e}"),
                            }
                        }
                    }
                }
                "tracking_command" => {
                    if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                        if binary_array.len() > 0 {
                            match serde_json::from_slice::<TrackingCommand>(binary_array.value(0)) {
                                Ok(cmd) => {
                                    tracing::info!("Tracking command: {:?}", cmd);
                                    pipeline.handle_tracking_command(cmd);
                                }
                                Err(e) => tracing::error!("Failed to parse TrackingCommand: {e}"),
                            }
                        }
                    }
                }
                "stream_control" => {
                    if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                        if binary_array.len() > 0 {
                            match serde_json::from_slice::<StreamControl>(binary_array.value(0)) {
                                Ok(ctrl) => {
                                    view_output.apply(&ctrl);
                                    tracing::info!(
                                        "Stream control: {:?}, view_enabled={}",
                                        ctrl.command,
                                        view_output.is_enabled()
                                    );
                                }
                                Err(e) => tracing::error!("Failed to parse StreamControl: {e}"),
                            }
                        }
                    }
                }
                other => tracing::warn!("Ignoring unexpected input: {other}"),
            },
            Event::Stop(_) => {
                tracing::info!("Stop received, closing camera");
                if let Some(cam) = camera_opt.take() {
                    cam.close()?;
                }
                break;
            }
            other => tracing::debug!("Unexpected event: {:?}", other),
        }
    }

    Ok(())
}

fn build_camera(
    source_type: &str,
    source_uri: &str,
) -> Result<CameraCapture, Box<dyn std::error::Error>> {
    match source_type {
        "webcam" => {
            let cols = env::var("IMAGE_COLS")?.parse::<usize>()?;
            let rows = env::var("IMAGE_ROWS")?.parse::<usize>()?;
            let fps = env::var("SOURCE_FPS")?.parse::<u32>()?;

            Ok(V4L2CameraConfig::new()
                .with_size([cols, rows].into())
                .with_fps(fps)
                .with_device(source_uri)
                .build()?)
        }
        "rtsp" => Ok(RTSPCameraConfig::new().with_url(source_uri).build()?),
        _ => Err(format!("Invalid SOURCE_TYPE: {source_type}").into()),
    }
}

fn unix_timestamp_ms() -> Result<u64, Box<dyn std::error::Error>> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

#[derive(Debug, Default)]
struct ViewOutputGate {
    enabled: bool,
}

impl ViewOutputGate {
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn apply(&mut self, control: &StreamControl) {
        self.enabled = match control.command {
            StreamCommand::Start | StreamCommand::Resume => control.video_enabled,
            StreamCommand::Stop | StreamCommand::Pause => false,
            StreamCommand::Configure => control.video_enabled,
        };
    }
}

#[derive(Debug)]
struct ViewFrameCadence {
    mode: ViewFrameCadenceMode,
}

#[derive(Debug)]
enum ViewFrameCadenceMode {
    FrameRatio {
        target_fps: u32,
        source_fps: u32,
        credit: u32,
    },
    TokenBucket {
        target_interval: Duration,
        max_credit: Duration,
        credit: Duration,
        last_seen_at: Option<Instant>,
    },
}

impl ViewFrameCadence {
    fn new(target_fps: u32, source_type: &str, source_fps: u32) -> Self {
        let target_fps = target_fps.max(1);
        if source_type == "webcam" {
            let source_fps = source_fps.max(1);
            return Self {
                mode: ViewFrameCadenceMode::FrameRatio {
                    target_fps,
                    source_fps,
                    credit: source_fps.saturating_sub(target_fps),
                },
            };
        }

        let target_interval = Duration::from_nanos(1_000_000_000 / target_fps as u64);
        let max_credit = target_interval.saturating_mul(target_fps);
        Self {
            mode: ViewFrameCadenceMode::TokenBucket {
                target_interval,
                max_credit,
                credit: target_interval,
                last_seen_at: None,
            },
        }
    }

    fn should_emit_at(&mut self, now: Instant) -> bool {
        match &mut self.mode {
            ViewFrameCadenceMode::FrameRatio {
                target_fps,
                source_fps,
                credit,
            } => {
                if *target_fps >= *source_fps {
                    return true;
                }

                *credit = credit.saturating_add(*target_fps);
                if *credit >= *source_fps {
                    *credit -= *source_fps;
                    true
                } else {
                    false
                }
            }
            ViewFrameCadenceMode::TokenBucket {
                target_interval,
                max_credit,
                credit,
                last_seen_at,
            } => {
                if let Some(previous) = *last_seen_at {
                    if now > previous {
                        *credit = credit
                            .saturating_add(now.duration_since(previous))
                            .min(*max_credit);
                    }
                }
                *last_seen_at = Some(now);

                if *credit < *target_interval {
                    return false;
                }

                *credit -= *target_interval;
                true
            }
        }
    }
}

#[cfg(test)]
impl Default for ViewFrameCadence {
    fn default() -> Self {
        Self::new(15, "webcam", 30)
    }
}

fn build_pipeline() -> Result<VisionPipeline, Box<dyn std::error::Error>> {
    let home = env::var("HOME").unwrap_or_default();

    let confidence_threshold = env::var("CONFIDENCE_THRESHOLD")
        .unwrap_or_else(|_| "0.5".into())
        .parse::<f32>()?;

    let nms_threshold = env::var("NMS_THRESHOLD")
        .unwrap_or_else(|_| "0.4".into())
        .parse::<f32>()?;

    let target_classes = env::var("TARGET_CLASSES")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();

    let max_age = env::var("MAX_TRACKING_AGE")
        .unwrap_or_else(|_| "50".into())
        .parse::<u32>()?;

    let min_hits = env::var("MIN_HITS")
        .unwrap_or_else(|_| "3".into())
        .parse::<u32>()?;

    let iou_threshold = env::var("IOU_THRESHOLD")
        .unwrap_or_else(|_| "0.3".into())
        .parse::<f32>()?;

    let reid_weight = env::var("REID_WEIGHT")
        .unwrap_or_else(|_| "0.8".into())
        .parse::<f32>()?;

    let reid_threshold = env::var("REID_THRESHOLD")
        .unwrap_or_else(|_| "0.5".into())
        .parse::<f32>()?;

    let enable_cmc = env::var("ENABLE_CMC")
        .unwrap_or_else(|_| "true".into())
        .parse::<bool>()?;

    Ok(VisionPipeline::new(
        DetectorConfig {
            model_path: env::var("MODEL_PATH")
                .unwrap_or_else(|_| format!("{home}/.cache/yolo/yolo12n.onnx")),
            confidence_threshold,
            nms_threshold,
            target_classes,
        },
        ReIdConfig {
            model_path: env::var("REID_MODEL_PATH")
                .unwrap_or_else(|_| format!("{home}/.cache/reid/osnet_x0_25.onnx")),
            min_bbox_size: env::var("MIN_BBOX_SIZE")
                .unwrap_or_else(|_| "32".into())
                .parse::<u32>()?,
        },
        TrackerConfig {
            max_age,
            min_hits,
            iou_threshold,
            reid_weight,
            reid_threshold,
            enable_cmc,
        },
    ))
}

fn send_pipeline_output(
    pipeline: &mut VisionPipeline,
    frame_id: u64,
    frame_data: &[u8],
    width: u32,
    height: u32,
    node: &mut DoraNode,
) -> Result<vision_pipeline::PipelineTimings, Box<dyn std::error::Error>> {
    let ProcessedPipelineOutput {
        output,
        mut timings,
    } = pipeline.process_frame(frame_id, frame_data, width, height)?;
    let serialization_started = Instant::now();
    match output {
        PipelineOutput::DetectionOnly {
            detections,
            tracking_telemetry,
        } => {
            let det_json = serde_json::to_vec(&detections)?;
            node.send_output(
                DataId::from("detections".to_owned()),
                Default::default(),
                BinaryArray::from_vec(vec![det_json.as_slice()]),
            )?;
            // Emit telemetry with state=DetectionOnly so web UI badge reflects pipeline mode
            let tel_json = serde_json::to_vec(&tracking_telemetry)?;
            node.send_output(
                DataId::from("tracking_telemetry".to_owned()),
                Default::default(),
                BinaryArray::from_vec(vec![tel_json.as_slice()]),
            )?;
        }
        PipelineOutput::FullTracking {
            tracked_detections,
            tracking_telemetry,
        } => {
            let det_json = serde_json::to_vec(&tracked_detections)?;
            node.send_output(
                DataId::from("tracked_detections".to_owned()),
                Default::default(),
                BinaryArray::from_vec(vec![det_json.as_slice()]),
            )?;

            let tel_json = serde_json::to_vec(&tracking_telemetry)?;
            node.send_output(
                DataId::from("tracking_telemetry".to_owned()),
                Default::default(),
                BinaryArray::from_vec(vec![tel_json.as_slice()]),
            )?;
        }
        PipelineOutput::CameraOnly { tracking_telemetry } => {
            // Emit state=Disabled so the web UI badge updates immediately
            let tel_json = serde_json::to_vec(&tracking_telemetry)?;
            node.send_output(
                DataId::from("tracking_telemetry".to_owned()),
                Default::default(),
                BinaryArray::from_vec(vec![tel_json.as_slice()]),
            )?;
        }
    }
    timings.serialization = serialization_started.elapsed();
    Ok(timings)
}

#[cfg(test)]
mod tests {
    use super::{ViewFrameCadence, ViewOutputGate};
    use robo_rover_lib::{StreamCommand, StreamControl};
    use std::time::{Duration, Instant};

    fn emitted_frames_for_capture_count(
        capture_frames: u64,
        target_fps: u32,
        duration: Duration,
    ) -> u64 {
        let mut cadence = ViewFrameCadence::new(target_fps, "rtsp", 30);
        let mut emitted = 0;
        let start = Instant::now();

        for frame_index in 0..capture_frames {
            let frame_offset_nanos =
                duration.as_nanos() * frame_index as u128 / capture_frames.max(1) as u128;
            let frame_time = start + Duration::from_nanos(frame_offset_nanos as u64);
            if cadence.should_emit_at(frame_time) {
                emitted += 1;
            }
        }
        emitted
    }

    fn capture_frames_for_duration(duration_ms: u64, source_fps: u32) -> u64 {
        duration_ms * source_fps as u64 / 1_000
    }

    fn stream_control(command: StreamCommand, video_enabled: bool) -> StreamControl {
        StreamControl {
            command,
            video_enabled,
            audio_enabled: false,
            quality: None,
            target_fps: Some(15),
        }
    }

    #[test]
    fn view_output_gate_defaults_to_no_view_work() {
        assert!(!ViewOutputGate::default().is_enabled());
    }

    #[test]
    fn view_output_gate_follows_stream_demand_without_affecting_capture() {
        let mut gate = ViewOutputGate::default();

        gate.apply(&stream_control(StreamCommand::Start, true));
        assert!(gate.is_enabled());

        gate.apply(&stream_control(StreamCommand::Configure, true));
        assert!(gate.is_enabled());

        gate.apply(&stream_control(StreamCommand::Stop, false));
        assert!(!gate.is_enabled());
    }

    #[test]
    fn view_cadence_meets_phase_two_gate_for_roughly_thirty_fps_input() {
        let emitted = emitted_frames_for_capture_count(
            capture_frames_for_duration(600_000, 30),
            15,
            Duration::from_secs(600),
        );

        assert!(
            emitted >= 8_700,
            "expected at least 14.5 FPS over 600s, got {emitted} frames"
        );
    }

    #[test]
    fn view_cadence_meets_phase_two_gate_for_observed_capture_count() {
        let mut cadence = ViewFrameCadence::new(15, "webcam", 30);
        let mut emitted = 0;
        for _ in 0..17_914 {
            if cadence.should_emit_at(Instant::now()) {
                emitted += 1;
            }
        }

        assert!(
            emitted >= 8_700,
            "expected observed 10-minute capture count to keep at least 14.5 FPS, got {emitted} frames"
        );
    }

    #[test]
    fn view_cadence_caps_fast_input_to_target_rate() {
        let emitted = emitted_frames_for_capture_count(
            capture_frames_for_duration(600_000, 60),
            15,
            Duration::from_secs(600),
        );

        assert!(
            (8_999..=9_001).contains(&emitted),
            "expected roughly 15 FPS over 600s for fast input, got {emitted} frames"
        );
    }

    #[test]
    fn view_cadence_handles_non_even_target_rate() {
        let emitted = emitted_frames_for_capture_count(
            capture_frames_for_duration(600_000, 30),
            17,
            Duration::from_secs(600),
        );

        assert!(
            (10_199..=10_201).contains(&emitted),
            "expected roughly 17 FPS over 600s, got {emitted} frames"
        );
    }

    #[test]
    fn view_cadence_does_not_backfill_after_pause() {
        let start = Instant::now();
        let mut cadence = ViewFrameCadence::new(15, "rtsp", 30);

        assert!(cadence.should_emit_at(start));
        assert!(!cadence.should_emit_at(start + Duration::from_millis(10)));
        assert!(cadence.should_emit_at(start + Duration::from_secs(5)));
        assert!(cadence.should_emit_at(start + Duration::from_secs(5) + Duration::from_millis(10)));
    }

    #[test]
    fn view_cadence_recovers_after_short_stalls() {
        let start = Instant::now();
        let mut cadence = ViewFrameCadence::new(15, "rtsp", 30);
        let mut emitted = 0;
        let mut elapsed = Duration::ZERO;

        for frame_index in 0..18_000 {
            if frame_index % 900 == 0 && frame_index != 0 {
                elapsed += Duration::from_millis(250);
            } else {
                elapsed += Duration::from_micros(33_333);
            }

            if cadence.should_emit_at(start + elapsed) {
                emitted += 1;
            }
        }

        assert!(
            emitted >= 8_700,
            "expected short stalls to remain above Phase 2 gate, got {emitted} frames"
        );
    }

    #[test]
    fn view_cadence_caps_recovery_burst() {
        let start = Instant::now();
        let mut cadence = ViewFrameCadence::new(15, "rtsp", 30);

        assert!(cadence.should_emit_at(start));

        let mut emitted_after_pause = 0;
        for frame_index in 0..100 {
            if cadence
                .should_emit_at(start + Duration::from_secs(5) + Duration::from_millis(frame_index))
            {
                emitted_after_pause += 1;
            }
        }

        assert!(emitted_after_pause <= 16);
    }

    #[test]
    fn webcam_view_cadence_honors_configured_source_ratio() {
        let mut cadence = ViewFrameCadence::new(15, "webcam", 30);
        let mut emitted = 0;

        for _ in 0..120 {
            if cadence.should_emit_at(Instant::now()) {
                emitted += 1;
            }
        }

        assert_eq!(emitted, 60);
    }
}
