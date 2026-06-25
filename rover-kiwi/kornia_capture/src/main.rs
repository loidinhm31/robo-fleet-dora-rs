mod latest_frame;
mod pipeline_metrics;
mod vision_pipeline;
mod vision_worker;

use dora_node_api::{
    self,
    arrow::array::{Array as ArrowArray, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event, Parameter,
};
use kornia_io::gstreamer::{CameraCapture, RTSPCameraConfig, V4L2CameraConfig};
use latest_frame::CapturedFrame;
use object_detector::DetectorConfig;
use object_tracker::TrackerConfig;
use pipeline_metrics::PipelineMetricWindows;
use reid_extractor::ReIdConfig;
use robo_rover_lib::{
    init_tracing,
    types::{TrackingCommand, TrackingState, TrackingTelemetry},
    CameraAction, CameraControl, MetricWindow, StreamCommand, StreamControl,
};
use std::{
    env,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use vision_pipeline::{PipelineOutput, VisionPipelineConfig};
use vision_worker::{
    CommandSubmitStatus, DrainStatus, VisionWorker, WorkerMessage, WorkerPipelineResult,
};

const MAX_SERVO_FRAME_AGE: Duration = Duration::from_millis(150);

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

    let pipeline_config = build_pipeline_config()?;

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
    let mut worker_submit_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut worker_result_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut worker_stale_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut worker_error_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut vision_output_log_state = VisionOutputLogState::default();
    let mut vision_worker = Some(VisionWorker::start(pipeline_config));
    let mut vision_submission = VisionSubmissionGate::default();

    while let Some(event) = events.recv() {
        if let Some(worker) = &vision_worker {
            let drain_status = drain_worker_results(
                worker,
                &mut node,
                &mut pipeline_metrics,
                &mut worker_result_metrics,
                &mut worker_stale_metrics,
                &mut worker_error_metrics,
                &mut vision_output_log_state,
            )?;
            if drain_status == DrainStatus::Disconnected {
                handle_worker_disconnected(
                    &mut vision_worker,
                    &mut node,
                    &mut worker_error_metrics,
                    "result channel disconnected",
                )?;
                vision_submission.disable();
            }
        }

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

                    if vision_submission.should_submit_frames() {
                        let expected_len = width as usize * height as usize * 3;
                        if frame.numel() == expected_len {
                            if let Some(worker) = &vision_worker {
                                let replaced = worker.submit_frame(CapturedFrame::new(
                                    frame_id,
                                    capture_started,
                                    capture_timestamp_ms,
                                    width,
                                    height,
                                    frame.as_slice().to_vec(),
                                ));
                                worker_submit_metrics.record(Duration::ZERO, frame.numel());
                                if replaced {
                                    worker_submit_metrics.record_drop();
                                }
                            }
                        } else {
                            tracing::error!(
                                "Invalid RGB frame length for worker: frame_id={frame_id}, expected={expected_len}, actual={}",
                                frame.numel()
                            );
                            worker_error_metrics.record_error();
                        }
                    }
                    vision_metrics.record(capture_started.elapsed(), frame.numel());
                    if let Some(worker) = &vision_worker {
                        let drain_status = drain_worker_results(
                            worker,
                            &mut node,
                            &mut pipeline_metrics,
                            &mut worker_result_metrics,
                            &mut worker_stale_metrics,
                            &mut worker_error_metrics,
                            &mut vision_output_log_state,
                        )?;
                        if drain_status == DrainStatus::Disconnected {
                            handle_worker_disconnected(
                                &mut vision_worker,
                                &mut node,
                                &mut worker_error_metrics,
                                "result channel disconnected",
                            )?;
                            vision_submission.disable();
                        }
                    }
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
                            stage = "vision_submit",
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
                    log_metric_snapshot(
                        "vision_worker_submit",
                        frame_id,
                        worker_submit_metrics.snapshot_if_due(),
                    );
                    log_metric_snapshot(
                        "vision_worker_result",
                        frame_id,
                        worker_result_metrics.snapshot_if_due(),
                    );
                    log_metric_snapshot(
                        "vision_worker_stale_drop",
                        frame_id,
                        worker_stale_metrics.snapshot_if_due(),
                    );
                    log_metric_snapshot(
                        "vision_worker_error",
                        frame_id,
                        worker_error_metrics.snapshot_if_due(),
                    );
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
                                    if let Some(worker) = &vision_worker {
                                        match worker.submit_command(cmd.clone()) {
                                            CommandSubmitStatus::Accepted => {
                                                vision_submission.apply_tracking_command(&cmd);
                                                if matches!(
                                                    cmd,
                                                    TrackingCommand::DisableDetection { .. }
                                                ) {
                                                    send_disabled_tracking_telemetry(&mut node)?;
                                                }
                                            }
                                            CommandSubmitStatus::Full => {
                                                worker_error_metrics.record_error();
                                            }
                                            CommandSubmitStatus::Disconnected => {
                                                handle_worker_disconnected(
                                                    &mut vision_worker,
                                                    &mut node,
                                                    &mut worker_error_metrics,
                                                    "command channel disconnected",
                                                )?;
                                                vision_submission.disable();
                                            }
                                        }
                                    }
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
                shutdown_worker(&mut vision_worker);
                break;
            }
            other => tracing::debug!("Unexpected event: {:?}", other),
        }
    }

    shutdown_worker(&mut vision_worker);

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

#[derive(Debug, Default)]
struct VisionSubmissionGate {
    submit_frames: bool,
}

impl VisionSubmissionGate {
    fn should_submit_frames(&self) -> bool {
        self.submit_frames
    }

    fn disable(&mut self) {
        self.submit_frames = false;
    }

    fn apply_tracking_command(&mut self, command: &TrackingCommand) {
        match command {
            // This gate only avoids the camera-only RGB copy on the Dora hot path.
            // The worker's VisionPipeline remains the authoritative tracking state.
            TrackingCommand::EnableDetection { .. } | TrackingCommand::Enable { .. } => {
                self.submit_frames = true;
            }
            TrackingCommand::DisableDetection { .. } => {
                self.submit_frames = false;
            }
            TrackingCommand::Disable { .. }
            | TrackingCommand::SelectTarget { .. }
            | TrackingCommand::SelectTargetById { .. }
            | TrackingCommand::ClearTarget { .. } => {}
        }
    }
}

fn build_pipeline_config() -> Result<VisionPipelineConfig, Box<dyn std::error::Error>> {
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

    let detector_model_path = resolve_runtime_path(
        "MODEL_PATH",
        &env::var("MODEL_PATH").unwrap_or_else(|_| format!("{home}/.cache/yolo/yolo12n.onnx")),
    );
    let reid_model_path = resolve_runtime_path(
        "REID_MODEL_PATH",
        &env::var("REID_MODEL_PATH")
            .unwrap_or_else(|_| format!("{home}/.cache/reid/osnet_x0_25.onnx")),
    );

    Ok(VisionPipelineConfig {
        detector: DetectorConfig {
            model_path: detector_model_path,
            confidence_threshold,
            nms_threshold,
            target_classes,
            intra_threads: env::var("DETECTOR_INTRA_THREADS")
                .unwrap_or_else(|_| "4".into())
                .parse::<i16>()?,
        },
        reid: ReIdConfig {
            model_path: reid_model_path,
            min_bbox_size: env::var("MIN_BBOX_SIZE")
                .unwrap_or_else(|_| "32".into())
                .parse::<u32>()?,
            intra_threads: env::var("REID_INTRA_THREADS")
                .unwrap_or_else(|_| "2".into())
                .parse::<i16>()?,
        },
        tracker: TrackerConfig {
            max_age,
            min_hits,
            iou_threshold,
            reid_weight,
            reid_threshold,
            enable_cmc,
        },
    })
}

fn resolve_runtime_path(label: &str, raw_path: &str) -> String {
    let raw = Path::new(raw_path);
    if raw.is_absolute() || raw.exists() {
        return raw_path.to_string();
    }

    let mut candidate_bases = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        candidate_bases.push(cwd);
    }
    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidate_bases.extend(exe_dir.ancestors().map(Path::to_path_buf));
        }
    }

    if let Some(resolved) =
        resolve_path_from_bases(raw, candidate_bases.iter().map(PathBuf::as_path))
    {
        tracing::info!(
            path_label = label,
            configured_path = raw_path,
            resolved_path = %resolved.display(),
            "Resolved runtime asset path"
        );
        return resolved.display().to_string();
    }

    tracing::warn!(
        path_label = label,
        configured_path = raw_path,
        "Runtime asset path could not be resolved; using configured value"
    );
    raw_path.to_string()
}

fn resolve_path_from_bases<'a>(
    raw: &Path,
    bases: impl IntoIterator<Item = &'a Path>,
) -> Option<PathBuf> {
    for base in bases {
        let candidate = base.join(raw);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn drain_worker_results(
    worker: &VisionWorker,
    node: &mut DoraNode,
    pipeline_metrics: &mut PipelineMetricWindows,
    result_metrics: &mut MetricWindow,
    stale_metrics: &mut MetricWindow,
    error_metrics: &mut MetricWindow,
    vision_output_log_state: &mut VisionOutputLogState,
) -> Result<DrainStatus, Box<dyn std::error::Error>> {
    let mut result = Ok(());
    let drain_status = worker.drain_results(|message| {
        if result.is_err() {
            return;
        }

        match message {
            WorkerMessage::Result(worker_result) => {
                if let Err(error) = handle_worker_result(
                    worker_result,
                    node,
                    pipeline_metrics,
                    result_metrics,
                    stale_metrics,
                    vision_output_log_state,
                ) {
                    result = Err(error);
                }
            }
            WorkerMessage::Error(error) => {
                tracing::error!(
                    frame_id = error.frame_id,
                    "Vision worker failed: {}",
                    error.message
                );
                error_metrics.record_error();
                if let Err(send_error) = send_disabled_tracking_telemetry(node) {
                    result = Err(send_error);
                }
            }
        }
    });
    result.map(|_| drain_status)
}

fn handle_worker_disconnected(
    worker: &mut Option<VisionWorker>,
    node: &mut DoraNode,
    error_metrics: &mut MetricWindow,
    reason: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::error!("Vision worker stopped: {reason}; disabling tracking telemetry");
    error_metrics.record_error();
    send_disabled_tracking_telemetry(node)?;
    shutdown_worker(worker);
    Ok(())
}

fn shutdown_worker(worker: &mut Option<VisionWorker>) {
    if let Some(worker) = worker.take() {
        let counters = worker.frame_slot().counters();
        tracing::info!(
            metric = "video_pipeline",
            stage = "latest_frame_slot_final",
            submitted = counters.submitted,
            replaced = counters.replaced,
            taken = counters.taken
        );
        worker.shutdown();
    }
}

fn handle_worker_result(
    WorkerPipelineResult {
        frame_id,
        captured_at,
        capture_timestamp_ms,
        output,
        mut timings,
    }: WorkerPipelineResult,
    node: &mut DoraNode,
    pipeline_metrics: &mut PipelineMetricWindows,
    result_metrics: &mut MetricWindow,
    stale_metrics: &mut MetricWindow,
    vision_output_log_state: &mut VisionOutputLogState,
) -> Result<(), Box<dyn std::error::Error>> {
    let age = captured_at.elapsed();
    if should_drop_stale_output(&output, age) {
        tracing::warn!(
            metric = "video_pipeline",
            stage = "vision_stale_drop",
            frame_id,
            capture_timestamp_ms,
            age_ms = age.as_millis() as u64,
            max_age_ms = MAX_SERVO_FRAME_AGE.as_millis() as u64
        );
        stale_metrics.record_drop();
        return Ok(());
    }

    result_metrics.record(age, 0);
    timings = send_pipeline_output(output, timings, node, vision_output_log_state)?;
    pipeline_metrics.record(frame_id, timings);
    Ok(())
}

#[derive(Debug, Default)]
struct VisionOutputLogState {
    last_signature: Option<String>,
}

impl VisionOutputLogState {
    fn log_transition(&mut self, signature: String, message: impl FnOnce()) {
        if self.last_signature.as_ref() == Some(&signature) {
            return;
        }
        self.last_signature = Some(signature);
        message();
    }
}

fn should_drop_stale_output(output: &PipelineOutput, age: Duration) -> bool {
    matches!(output, PipelineOutput::FullTracking { .. }) && age > MAX_SERVO_FRAME_AGE
}

fn send_pipeline_output(
    output: PipelineOutput,
    mut timings: vision_pipeline::PipelineTimings,
    node: &mut DoraNode,
    vision_output_log_state: &mut VisionOutputLogState,
) -> Result<vision_pipeline::PipelineTimings, Box<dyn std::error::Error>> {
    let serialization_started = Instant::now();
    match output {
        PipelineOutput::DetectionOnly {
            detections,
            tracking_telemetry,
        } => {
            let signature = format!("detection_only::{:?}", tracking_telemetry.state);
            vision_output_log_state.log_transition(signature, || {
                tracing::info!(
                    event = "vision_output_forwarded",
                    variant = "detection_only",
                    frame_id = detections.frame_id,
                    object_count = detections.detections.len(),
                    state = ?tracking_telemetry.state
                );
            });
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
            let signature = format!("full_tracking::{:?}", tracking_telemetry.state);
            vision_output_log_state.log_transition(signature, || {
                tracing::info!(
                    event = "vision_output_forwarded",
                    variant = "full_tracking",
                    frame_id = tracked_detections.frame_id,
                    object_count = tracked_detections.detections.len(),
                    state = ?tracking_telemetry.state
                );
            });
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
            let signature = format!("camera_only::{:?}", tracking_telemetry.state);
            vision_output_log_state.log_transition(signature, || {
                tracing::info!(
                    event = "vision_output_forwarded",
                    variant = "camera_only",
                    state = ?tracking_telemetry.state
                );
            });
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

fn send_disabled_tracking_telemetry(node: &mut DoraNode) -> Result<(), Box<dyn std::error::Error>> {
    let telemetry = TrackingTelemetry::new(TrackingState::Disabled, None);
    let tel_json = serde_json::to_vec(&telemetry)?;
    node.send_output(
        DataId::from("tracking_telemetry".to_owned()),
        Default::default(),
        BinaryArray::from_vec(vec![tel_json.as_slice()]),
    )?;
    Ok(())
}

fn log_metric_snapshot(
    stage: &str,
    frame_id: u64,
    snapshot: Option<robo_rover_lib::MetricSnapshot>,
) {
    if let Some(snapshot) = snapshot {
        tracing::info!(
            metric = "video_pipeline",
            stage,
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

#[cfg(test)]
mod tests {
    use super::{
        resolve_path_from_bases, ViewFrameCadence, ViewOutputGate, VisionSubmissionGate,
        MAX_SERVO_FRAME_AGE,
    };
    use robo_rover_lib::{StreamCommand, StreamControl, TrackingCommand};
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{Duration, Instant},
    };

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
    fn vision_submission_gate_skips_frames_until_detection_or_tracking_enabled() {
        let mut gate = VisionSubmissionGate::default();

        assert!(!gate.should_submit_frames());

        gate.apply_tracking_command(&TrackingCommand::EnableDetection { timestamp: 1 });
        assert!(gate.should_submit_frames());

        gate.apply_tracking_command(&TrackingCommand::Disable { timestamp: 2 });
        assert!(gate.should_submit_frames());

        gate.apply_tracking_command(&TrackingCommand::DisableDetection { timestamp: 3 });
        assert!(!gate.should_submit_frames());

        gate.apply_tracking_command(&TrackingCommand::Enable { timestamp: 4 });
        assert!(gate.should_submit_frames());
    }

    #[test]
    fn stale_detection_only_output_is_not_dropped_by_servo_age_gate() {
        use crate::vision_pipeline::PipelineOutput;
        use robo_rover_lib::types::{DetectionFrame, TrackingState, TrackingTelemetry};

        let output = PipelineOutput::DetectionOnly {
            detections: DetectionFrame::new(1, 640, 480, Vec::new()),
            tracking_telemetry: TrackingTelemetry::new(TrackingState::DetectionOnly, None),
        };

        assert!(!super::should_drop_stale_output(
            &output,
            MAX_SERVO_FRAME_AGE + Duration::from_millis(1),
        ));
    }

    #[test]
    fn stale_full_tracking_output_is_dropped_by_servo_age_gate() {
        use crate::vision_pipeline::PipelineOutput;
        use robo_rover_lib::types::{DetectionFrame, TrackingState, TrackingTelemetry};

        let output = PipelineOutput::FullTracking {
            tracked_detections: DetectionFrame::new(1, 640, 480, Vec::new()),
            tracking_telemetry: TrackingTelemetry::new(TrackingState::Tracking, None),
        };

        assert!(super::should_drop_stale_output(
            &output,
            MAX_SERVO_FRAME_AGE + Duration::from_millis(1),
        ));
    }

    #[test]
    fn resolve_path_from_bases_finds_relative_asset_under_base() {
        let root = std::env::temp_dir().join(format!("kornia-capture-test-{}", std::process::id()));
        let repo_root = root.join("repo");
        let model_path = repo_root.join("models/.cache/yolo/yolo12n.onnx");
        fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        fs::write(&model_path, b"test").unwrap();

        let resolved = resolve_path_from_bases(
            Path::new("models/.cache/yolo/yolo12n.onnx"),
            [root.join("miss"), repo_root.clone()]
                .iter()
                .map(PathBuf::as_path),
        );

        assert_eq!(resolved, Some(model_path));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_path_from_bases_returns_none_when_asset_missing() {
        let resolved = resolve_path_from_bases(
            Path::new("models/.cache/yolo/yolo12n.onnx"),
            [PathBuf::from("/tmp/miss"), PathBuf::from("/also-miss")]
                .iter()
                .map(PathBuf::as_path),
        );

        assert!(resolved.is_none());
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
