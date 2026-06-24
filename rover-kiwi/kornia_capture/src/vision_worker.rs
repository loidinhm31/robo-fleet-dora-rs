use crate::{
    latest_frame::{CapturedFrame, LatestFrameSlot, TakeOutcome},
    vision_pipeline::{
        PipelineOutput, PipelineTimings, ProcessedPipelineOutput, VisionPipeline,
        VisionPipelineConfig,
    },
};
use robo_rover_lib::types::TrackingCommand;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

// Control traffic is bounded so a stalled worker cannot grow memory without limit.
// Overflow is counted and logged; the main loop treats disconnect as a safe-disable event.
const COMMAND_CAPACITY: usize = 32;
// Results are freshness-biased: if the main Dora loop falls behind, older finished
// results are dropped instead of queueing stale servo inputs.
const RESULT_CAPACITY: usize = 4;

pub struct VisionWorker {
    frames: LatestFrameSlot,
    commands: SyncSender<TrackingCommand>,
    results: Receiver<WorkerMessage>,
    accepted_commands: Arc<AtomicU64>,
    command_drops: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

#[derive(Debug)]
pub struct WorkerPipelineResult {
    pub frame_id: u64,
    pub captured_at: Instant,
    pub capture_timestamp_ms: u64,
    pub output: PipelineOutput,
    pub timings: PipelineTimings,
}

#[derive(Debug)]
pub enum WorkerMessage {
    Result(WorkerPipelineResult),
    Error(WorkerError),
}

#[derive(Debug, Clone)]
pub struct WorkerError {
    pub frame_id: Option<u64>,
    pub message: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct WorkerCounters {
    pub result_drops: u64,
    pub errors: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandSubmitStatus {
    Accepted,
    Full,
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainStatus {
    Open,
    Disconnected,
}

impl VisionWorker {
    pub fn start(config: VisionPipelineConfig) -> Self {
        let frames = LatestFrameSlot::default();
        let worker_frames = frames.clone();
        let (command_tx, command_rx) = sync_channel(COMMAND_CAPACITY);
        let (result_tx, result_rx) = sync_channel(RESULT_CAPACITY);
        let accepted_commands = Arc::new(AtomicU64::new(0));
        let command_drops = Arc::new(AtomicU64::new(0));

        let handle = thread::Builder::new()
            .name("vision-pipeline-worker".into())
            .spawn(move || run_worker(config, worker_frames, command_rx, result_tx))
            .expect("failed to spawn vision worker");

        Self {
            frames,
            commands: command_tx,
            results: result_rx,
            accepted_commands,
            command_drops,
            handle: Some(handle),
        }
    }

    pub fn submit_frame(&self, frame: CapturedFrame) -> bool {
        self.frames.submit(frame)
    }

    pub fn submit_command(&self, command: TrackingCommand) -> CommandSubmitStatus {
        match self.commands.try_send(command) {
            Ok(()) => {
                self.accepted_commands.fetch_add(1, Ordering::Relaxed);
                self.frames.wake();
                CommandSubmitStatus::Accepted
            }
            Err(TrySendError::Full(command)) => {
                self.command_drops.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("Dropping tracking command because worker command channel is full");
                drop(command);
                CommandSubmitStatus::Full
            }
            Err(TrySendError::Disconnected(command)) => {
                self.command_drops.fetch_add(1, Ordering::Relaxed);
                tracing::error!("Dropping tracking command because vision worker has stopped");
                drop(command);
                CommandSubmitStatus::Disconnected
            }
        }
    }

    pub fn drain_results(&self, mut on_message: impl FnMut(WorkerMessage)) -> DrainStatus {
        loop {
            match self.results.try_recv() {
                Ok(message) => on_message(message),
                Err(TryRecvError::Empty) => return DrainStatus::Open,
                Err(TryRecvError::Disconnected) => return DrainStatus::Disconnected,
            }
        }
    }

    pub fn frame_slot(&self) -> &LatestFrameSlot {
        &self.frames
    }

    pub fn shutdown(mut self) {
        self.frames.close();
        drop(self.commands);
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                tracing::error!("Vision worker panicked during shutdown");
            }
        }
        tracing::info!(
            metric = "video_pipeline",
            stage = "vision_worker_command_submit",
            accepted_commands = self.accepted_commands.load(Ordering::Relaxed),
            command_drops = self.command_drops.load(Ordering::Relaxed)
        );
    }
}

fn run_worker(
    config: VisionPipelineConfig,
    frames: LatestFrameSlot,
    commands: Receiver<TrackingCommand>,
    results: SyncSender<WorkerMessage>,
) {
    let mut pipeline = VisionPipeline::from_config(config);
    let mut wake_generation = 0;
    let mut counters = WorkerCounters::default();

    loop {
        apply_pending_commands(&mut pipeline, &commands, &mut counters);

        match frames.take_next(&mut wake_generation) {
            (TakeOutcome::Frame, Some(frame)) => {
                apply_pending_commands(&mut pipeline, &commands, &mut counters);
                process_frame(&mut pipeline, frame, &results, &mut counters);
            }
            (TakeOutcome::Woken, None) => continue,
            (TakeOutcome::Closed, None) => break,
            _ => continue,
        }
    }

    tracing::info!(
        metric = "video_pipeline",
        stage = "vision_worker_shutdown",
        result_drops = counters.result_drops,
        errors = counters.errors
    );
}

fn process_frame(
    pipeline: &mut VisionPipeline,
    frame: CapturedFrame,
    results: &SyncSender<WorkerMessage>,
    counters: &mut WorkerCounters,
) {
    let frame_id = frame.frame_id;
    let captured_at = frame.captured_at;
    let capture_timestamp_ms = frame.capture_timestamp_ms;
    let processed = pipeline.process_frame(frame_id, &frame.rgb, frame.width, frame.height);
    let message = match processed {
        Ok(ProcessedPipelineOutput { output, timings }) => {
            WorkerMessage::Result(WorkerPipelineResult {
                frame_id,
                captured_at,
                capture_timestamp_ms,
                output,
                timings,
            })
        }
        Err(error) => {
            counters.errors = counters.errors.saturating_add(1);
            WorkerMessage::Error(WorkerError {
                frame_id: Some(frame_id),
                message: error.to_string(),
            })
        }
    };

    if let Err(error) = results.try_send(message) {
        counters.result_drops = counters.result_drops.saturating_add(1);
        tracing::warn!("Dropping vision worker result: {error}");
    }
}

fn apply_pending_commands(
    pipeline: &mut VisionPipeline,
    commands: &Receiver<TrackingCommand>,
    counters: &mut WorkerCounters,
) {
    for command in coalesce_commands(drain_commands(commands, counters)) {
        pipeline.handle_tracking_command(command);
    }
}

fn drain_commands(
    commands: &Receiver<TrackingCommand>,
    _counters: &mut WorkerCounters,
) -> Vec<TrackingCommand> {
    let mut drained = Vec::new();
    loop {
        match commands.try_recv() {
            Ok(command) => drained.push(command),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
        }
    }
    drained
}

fn coalesce_commands(commands: Vec<TrackingCommand>) -> Vec<TrackingCommand> {
    let mut coalesced = Vec::new();
    let mut latest_mode = None;

    for command in commands {
        if is_mode_command(&command) {
            latest_mode = Some(command);
        } else {
            if let Some(mode) = latest_mode.take() {
                coalesced.push(mode);
            }
            coalesced.push(command);
        }
    }

    if let Some(mode) = latest_mode {
        coalesced.push(mode);
    }

    coalesced
}

fn is_mode_command(command: &TrackingCommand) -> bool {
    matches!(
        command,
        TrackingCommand::EnableDetection { .. }
            | TrackingCommand::DisableDetection { .. }
            | TrackingCommand::Enable { .. }
            | TrackingCommand::Disable { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::{
        coalesce_commands, sync_channel, Arc, AtomicU64, CommandSubmitStatus, LatestFrameSlot,
        Ordering, VisionWorker,
    };
    use crate::vision_pipeline::VisionPipelineConfig;
    use object_detector::DetectorConfig;
    use object_tracker::TrackerConfig;
    use reid_extractor::ReIdConfig;
    use robo_rover_lib::types::TrackingCommand;

    fn test_config() -> VisionPipelineConfig {
        VisionPipelineConfig {
            detector: DetectorConfig {
                model_path: "unused-yolo.onnx".into(),
                confidence_threshold: 0.5,
                nms_threshold: 0.4,
                target_classes: Vec::new(),
                intra_threads: 1,
            },
            reid: ReIdConfig {
                model_path: "unused-reid.onnx".into(),
                min_bbox_size: 32,
                intra_threads: 1,
            },
            tracker: TrackerConfig {
                max_age: 50,
                min_hits: 3,
                iou_threshold: 0.3,
                reid_weight: 0.8,
                reid_threshold: 0.5,
                enable_cmc: false,
            },
        }
    }

    #[test]
    fn coalesces_tracking_mode_commands_before_frame_work() {
        let commands = coalesce_commands(vec![
            TrackingCommand::EnableDetection { timestamp: 1 },
            TrackingCommand::DisableDetection { timestamp: 2 },
            TrackingCommand::Enable { timestamp: 3 },
        ]);

        assert_eq!(commands.len(), 1);
        assert!(matches!(
            commands[0],
            TrackingCommand::Enable { timestamp: 3 }
        ));
    }

    #[test]
    fn preserves_selection_command_order_with_latest_preceding_mode() {
        let commands = coalesce_commands(vec![
            TrackingCommand::EnableDetection { timestamp: 1 },
            TrackingCommand::Enable { timestamp: 2 },
            TrackingCommand::SelectTargetById {
                tracking_id: 7,
                timestamp: 3,
            },
            TrackingCommand::Disable { timestamp: 4 },
            TrackingCommand::ClearTarget { timestamp: 5 },
        ]);

        assert_eq!(commands.len(), 4);
        assert!(matches!(
            commands[0],
            TrackingCommand::Enable { timestamp: 2 }
        ));
        assert!(matches!(
            commands[1],
            TrackingCommand::SelectTargetById {
                tracking_id: 7,
                timestamp: 3
            }
        ));
        assert!(matches!(
            commands[2],
            TrackingCommand::Disable { timestamp: 4 }
        ));
        assert!(matches!(
            commands[3],
            TrackingCommand::ClearTarget { timestamp: 5 }
        ));
    }

    #[test]
    fn worker_shutdown_closes_waiting_thread() {
        let worker = VisionWorker::start(test_config());
        worker.shutdown();
    }

    #[test]
    fn disconnected_command_submission_is_counted_and_reported() {
        let (command_tx, command_rx) = sync_channel(1);
        drop(command_rx);
        let (_result_tx, result_rx) = sync_channel(1);
        let accepted_commands = Arc::new(AtomicU64::new(0));
        let command_drops = Arc::new(AtomicU64::new(0));
        let worker = VisionWorker {
            frames: LatestFrameSlot::default(),
            commands: command_tx,
            results: result_rx,
            accepted_commands: accepted_commands.clone(),
            command_drops: command_drops.clone(),
            handle: None,
        };

        assert_eq!(
            worker.submit_command(TrackingCommand::Enable { timestamp: 1 }),
            CommandSubmitStatus::Disconnected
        );
        assert_eq!(accepted_commands.load(Ordering::Relaxed), 0);
        assert_eq!(command_drops.load(Ordering::Relaxed), 1);
    }
}
