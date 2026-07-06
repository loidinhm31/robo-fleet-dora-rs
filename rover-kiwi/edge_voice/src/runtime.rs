use std::{
    sync::mpsc::Receiver,
    time::{Duration, Instant},
};

use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::{eyre, Result};
use robo_rover_lib::{
    init_tracing, TtsCommand, TtsCommandResult, TtsConfigCommand, TtsResultState, TtsRuntimeConfig,
    VoiceReasonCode, VoiceState,
};
use serde_json::json;

use crate::{
    config::DeploymentConfig,
    protocol::{
        audio_metadata, command_result, parse_config_command, parse_playback_state,
        parse_tts_command, sanitized_error, to_binary, validate_result, validate_voice_status,
        voice_status, walkie_is_active,
    },
    queue::{EnqueueStatus, VoiceQueue},
    tts_pacer::{InstantClock, PacingSnapshot, TtsAudioChunk, TtsPacer},
    worker::{self, WorkerEvent, WorkerHandle},
};

const INPUT_POLL_CEILING: Duration = Duration::from_millis(20);
const METRICS_INTERVAL: Duration = Duration::from_secs(5);
const MAX_WORKER_EVENTS_PER_TICK: usize = worker::WORKER_EVENT_CAPACITY + 1;

pub fn run() -> Result<()> {
    let _guard = init_tracing();
    let deployment = DeploymentConfig::from_env()?;

    let (mut node, mut events) = DoraNode::init_from_env()?;
    let outputs = Outputs::default();
    let mut runtime = RuntimeState::new(deployment.clone());
    emit_status(
        &mut node,
        &outputs,
        runtime.status(VoiceState::Loading, None, None),
    )?;

    let (worker, worker_rx) = worker::spawn(deployment);
    tracing::info!("edge_voice node initialized; model loading on worker");

    loop {
        emit_due_paced_chunk(&mut node, &outputs, &mut runtime, &worker)?;
        drain_worker_events(&mut node, &outputs, &mut runtime, &worker, &worker_rx)?;
        emit_periodic_metrics(&mut node, &outputs, &mut runtime)?;

        match events.recv_timeout(runtime.input_timeout()) {
            Some(Event::Input { id, data, .. }) => {
                if let Err(error) = handle_input(
                    id.as_str(),
                    data.as_ref(),
                    &mut node,
                    &outputs,
                    &mut runtime,
                    &worker,
                ) {
                    tracing::warn!(%error, input = %id, "edge_voice rejected input");
                }
                dispatch_if_idle(&mut runtime, &worker, &mut node, &outputs)?;
            }
            Some(Event::Stop(_)) => {
                worker.stop();
                drain_worker_events_until(
                    &mut node,
                    &outputs,
                    &mut runtime,
                    &worker,
                    &worker_rx,
                    Duration::from_secs(2),
                )?;
                break;
            }
            Some(_) => {}
            None => {}
        }
    }

    emit_due_paced_chunk(&mut node, &outputs, &mut runtime, &worker)?;
    drain_worker_events(&mut node, &outputs, &mut runtime, &worker, &worker_rx)?;
    emit_pacing_metrics(&mut node, &outputs, &runtime, "shutdown")?;
    tracing::info!("edge_voice node stopped");
    Ok(())
}

#[derive(Debug)]
struct Outputs {
    tts_audio: DataId,
    voice_status: DataId,
    tts_command_result: DataId,
    tts_synthesis_state: DataId,
    metrics: DataId,
}

impl Default for Outputs {
    fn default() -> Self {
        Self {
            tts_audio: DataId::from("tts_audio".to_string()),
            voice_status: DataId::from("voice_status".to_string()),
            tts_command_result: DataId::from("tts_command_result".to_string()),
            tts_synthesis_state: DataId::from("tts_synthesis_state".to_string()),
            metrics: DataId::from("metrics".to_string()),
        }
    }
}

#[derive(Debug)]
struct PendingCompletion {
    command_id: String,
    elapsed_ms: u64,
    samples: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RuntimeCounters {
    worker_backpressure_count: u64,
    cancellation_count: u64,
    terminal_reason: Option<String>,
}

#[derive(Debug)]
struct RuntimeState {
    entity_id: String,
    queue: VoiceQueue,
    current_config: TtsRuntimeConfig,
    current_revision: u64,
    ready: bool,
    load_failed: bool,
    busy: bool,
    walkie_active: bool,
    active_command_id: Option<String>,
    active_config: Option<TtsRuntimeConfig>,
    active_revision: Option<u64>,
    synthesis_finished: bool,
    playback_failure_pending: bool,
    pacer: Option<TtsPacer<InstantClock>>,
    pending_completion: Option<PendingCompletion>,
    last_metrics_emit: Instant,
    last_metrics_snapshot: Option<PacingSnapshot>,
    counters: RuntimeCounters,
}

impl RuntimeState {
    fn new(deployment: DeploymentConfig) -> Self {
        Self {
            entity_id: deployment.entity_id,
            queue: VoiceQueue::new(deployment.queue_capacity),
            current_config: deployment.default_runtime,
            current_revision: 0,
            ready: false,
            load_failed: false,
            busy: false,
            walkie_active: false,
            active_command_id: None,
            active_config: None,
            active_revision: None,
            synthesis_finished: false,
            playback_failure_pending: false,
            pacer: None,
            pending_completion: None,
            last_metrics_emit: Instant::now(),
            last_metrics_snapshot: None,
            counters: RuntimeCounters::default(),
        }
    }

    fn status(
        &self,
        state: VoiceState,
        reason: Option<VoiceReasonCode>,
        detail: Option<String>,
    ) -> robo_rover_lib::VoiceStatus {
        let applied_revision = self
            .active_revision
            .filter(|_| self.busy)
            .unwrap_or(self.current_revision);
        let applied_config = self
            .active_config
            .as_ref()
            .filter(|_| self.busy)
            .cloned()
            .unwrap_or_else(|| self.current_config.clone());
        voice_status(
            &self.entity_id,
            state,
            applied_revision,
            applied_config,
            self.active_command_id.clone(),
            reason,
            detail,
        )
    }

    fn apply_config(&mut self, command: TtsConfigCommand) -> bool {
        if command.revision < self.current_revision {
            return false;
        }
        self.current_revision = command.revision;
        self.current_config = command.config;
        true
    }

    fn input_timeout(&self) -> Duration {
        self.pacer
            .as_ref()
            .map(|pacer| pacer.timeout_until_due(INPUT_POLL_CEILING))
            .unwrap_or(INPUT_POLL_CEILING)
    }

    fn clear_pacing_for_active_command(&mut self) {
        if let Some(pacer) = self.pacer.as_mut() {
            pacer.clear_pending();
        }
        if let Some(pacer) = self.pacer.take() {
            self.last_metrics_snapshot = Some(pacer.snapshot());
        }
        self.pending_completion = None;
    }

    fn record_terminal(&mut self, state: TtsResultState, reason: Option<VoiceReasonCode>) {
        self.counters.terminal_reason = Some(terminal_reason_label(state, reason));
    }
}

fn handle_input(
    id: &str,
    data: &dyn Array,
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
    worker: &WorkerHandle,
) -> Result<()> {
    match id {
        "tts_command" | "tts_command_web" => {
            let command = parse_tts_command(binary_payload(data)?)?;
            tracing::info!(
                command_id = %command.command_id,
                text_len = command.text.len(),
                priority = ?command.priority,
                "edge_voice received TTS command"
            );
            handle_tts_command(command, node, outputs, runtime)?;
        }
        "tts_config_command" | "tts_config" => {
            let command = parse_config_command(binary_payload(data)?)?;
            tracing::info!(
                revision = command.revision,
                language = ?command.config.language,
                speaker_id = command.config.speaker_id,
                speed = command.config.speed,
                num_steps = command.config.num_steps,
                volume = command.config.volume,
                "edge_voice received TTS config command"
            );
            let stale_revision = command.revision;
            if !runtime.apply_config(command) {
                tracing::warn!(
                    current_revision = runtime.current_revision,
                    stale_revision,
                    "ignored stale edge voice config revision"
                );
            }
            emit_status(
                node,
                outputs,
                runtime.status(current_voice_state(runtime), None, None),
            )?;
        }
        "playback_state" => {
            let state = parse_playback_state(binary_payload(data)?)?;
            if walkie_is_active(&state) {
                runtime.walkie_active = true;
                if let Some(command_id) = runtime.active_command_id.clone() {
                    if runtime.synthesis_finished {
                        finish_active_command(
                            node,
                            outputs,
                            runtime,
                            &command_id,
                            TtsResultState::Interrupted,
                            Some(VoiceReasonCode::InterruptedByWalkie),
                            None,
                        )?;
                    } else {
                        request_worker_cancel(
                            runtime,
                            worker,
                            VoiceReasonCode::InterruptedByWalkie,
                        );
                        runtime.clear_pacing_for_active_command();
                        emit_synthesis_state(
                            node,
                            outputs,
                            command_result(
                                &runtime.entity_id,
                                &command_id,
                                TtsResultState::Interrupted,
                                Some(VoiceReasonCode::InterruptedByWalkie),
                                None,
                            ),
                        )?;
                        finish_active_command(
                            node,
                            outputs,
                            runtime,
                            &command_id,
                            TtsResultState::Interrupted,
                            Some(VoiceReasonCode::InterruptedByWalkie),
                            None,
                        )?;
                    }
                }
                for command_id in runtime.queue.clear_ids() {
                    emit_result(
                        node,
                        outputs,
                        command_result(
                            &runtime.entity_id,
                            &command_id,
                            TtsResultState::Interrupted,
                            Some(VoiceReasonCode::InterruptedByWalkie),
                            None,
                        ),
                    )?;
                }
            } else {
                runtime.walkie_active = false;
            }
        }
        "playback_result" => {
            let result = serde_json::from_slice::<TtsCommandResult>(binary_payload(data)?)?;
            validate_result(&result)?;
            if result.entity_id != runtime.entity_id
                || runtime.active_command_id.as_deref() != Some(&result.command_id)
            {
                return Err(eyre!("playback result does not match active command"));
            }
            match result.state {
                TtsResultState::Completed if runtime.synthesis_finished => {
                    finish_active_command(
                        node,
                        outputs,
                        runtime,
                        &result.command_id,
                        TtsResultState::Completed,
                        None,
                        None,
                    )?;
                }
                TtsResultState::Failed => {
                    if runtime.synthesis_finished {
                        finish_active_command(
                            node,
                            outputs,
                            runtime,
                            &result.command_id,
                            TtsResultState::Failed,
                            Some(VoiceReasonCode::PlaybackFailed),
                            None,
                        )?;
                    } else {
                        runtime.playback_failure_pending = true;
                        runtime.clear_pacing_for_active_command();
                        request_worker_cancel(runtime, worker, VoiceReasonCode::PlaybackFailed);
                    }
                }
                _ => return Err(eyre!("invalid playback result state or ordering")),
            }
        }
        other => return Err(eyre!("unexpected edge_voice input: {other}")),
    }
    Ok(())
}

fn handle_tts_command(
    command: TtsCommand,
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
) -> Result<()> {
    if runtime.walkie_active {
        emit_result(
            node,
            outputs,
            command_result(
                &runtime.entity_id,
                &command.command_id,
                TtsResultState::Rejected,
                Some(VoiceReasonCode::WalkieActive),
                None,
            ),
        )?;
        return Ok(());
    }
    if runtime.load_failed {
        emit_result(
            node,
            outputs,
            command_result(
                &runtime.entity_id,
                &command.command_id,
                TtsResultState::Rejected,
                Some(VoiceReasonCode::VoiceNotReady),
                Some("voice engine is not ready".to_string()),
            ),
        )?;
        return Ok(());
    }

    let outcome = runtime.queue.enqueue(command.clone());
    match outcome.status {
        EnqueueStatus::Accepted => {
            for interrupted_id in outcome.interrupted_command_ids {
                emit_result(
                    node,
                    outputs,
                    command_result(
                        &runtime.entity_id,
                        &interrupted_id,
                        TtsResultState::Interrupted,
                        Some(VoiceReasonCode::Cancelled),
                        Some("interrupted by higher priority command".to_string()),
                    ),
                )?;
            }
        }
        EnqueueStatus::Rejected(reason) => {
            emit_result(
                node,
                outputs,
                command_result(
                    &runtime.entity_id,
                    &command.command_id,
                    TtsResultState::Rejected,
                    Some(reason),
                    None,
                ),
            )?;
        }
    }
    Ok(())
}

fn dispatch_if_idle(
    runtime: &mut RuntimeState,
    worker: &WorkerHandle,
    node: &mut DoraNode,
    outputs: &Outputs,
) -> Result<()> {
    if !runtime.ready || runtime.busy || runtime.walkie_active {
        return Ok(());
    }
    let Some(command) = runtime.queue.pop_next() else {
        return Ok(());
    };
    if let Err(error) = worker.synthesize(
        command.clone(),
        runtime.current_config.clone(),
        runtime.current_revision,
    ) {
        emit_result(
            node,
            outputs,
            command_result(
                &runtime.entity_id,
                &command.command_id,
                TtsResultState::Failed,
                Some(VoiceReasonCode::InternalError),
                Some(sanitized_error(error)),
            ),
        )?;
    } else {
        runtime.busy = true;
    }
    Ok(())
}

fn drain_worker_events(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
    worker: &WorkerHandle,
    worker_rx: &Receiver<WorkerEvent>,
) -> Result<()> {
    if runtime
        .pacer
        .as_ref()
        .map(|pacer| pacer.is_full())
        .unwrap_or(false)
    {
        runtime.counters.worker_backpressure_count =
            runtime.counters.worker_backpressure_count.saturating_add(1);
        return Ok(());
    }
    for _ in 0..MAX_WORKER_EVENTS_PER_TICK {
        if runtime
            .pacer
            .as_ref()
            .map(|pacer| pacer.is_full())
            .unwrap_or(false)
        {
            runtime.counters.worker_backpressure_count =
                runtime.counters.worker_backpressure_count.saturating_add(1);
            break;
        }
        let Ok(event) = worker_rx.try_recv() else {
            break;
        };
        if discard_stale_worker_audio(runtime, &event) {
            continue;
        }
        handle_worker_event(node, outputs, runtime, worker, event)?;
    }
    Ok(())
}

fn drain_worker_events_until(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
    worker: &WorkerHandle,
    worker_rx: &Receiver<WorkerEvent>,
    timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        emit_due_paced_chunk(node, outputs, runtime, worker)?;
        if runtime
            .pacer
            .as_ref()
            .map(|pacer| pacer.is_full())
            .unwrap_or(false)
        {
            runtime.counters.worker_backpressure_count =
                runtime.counters.worker_backpressure_count.saturating_add(1);
            std::thread::sleep(runtime.input_timeout());
            if Instant::now() >= deadline {
                return Ok(());
            }
            continue;
        }
        match worker_rx.recv_timeout(runtime.input_timeout()) {
            Ok(event) => {
                let stopped = matches!(event, WorkerEvent::Stopped);
                if !discard_stale_worker_audio(runtime, &event) {
                    handle_worker_event(node, outputs, runtime, worker, event)?;
                }
                if stopped {
                    return Ok(());
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) if Instant::now() < deadline => {}
            Err(_) => return Ok(()),
        }
    }
}

fn handle_worker_event(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
    worker: &WorkerHandle,
    event: WorkerEvent,
) -> Result<()> {
    match event {
        WorkerEvent::Loaded {
            sample_rate,
            speakers,
        } => {
            tracing::info!(
                sample_rate,
                speakers,
                "edge_voice worker loaded Supertonic model"
            );
            runtime.ready = true;
            emit_status(node, outputs, runtime.status(VoiceState::Ready, None, None))?;
            emit_metric(
                node,
                outputs,
                json!({"event":"loaded","sample_rate":sample_rate,"speakers":speakers}),
            )?;
            dispatch_if_idle(runtime, worker, node, outputs)?;
        }
        WorkerEvent::LoadFailed { detail } => {
            tracing::error!(%detail, "edge_voice worker failed to load Supertonic model");
            runtime.load_failed = true;
            for command_id in runtime.queue.clear_ids() {
                emit_result(
                    node,
                    outputs,
                    command_result(
                        &runtime.entity_id,
                        &command_id,
                        TtsResultState::Rejected,
                        Some(VoiceReasonCode::VoiceNotReady),
                        Some("voice engine failed to load".to_string()),
                    ),
                )?;
            }
            emit_status(
                node,
                outputs,
                runtime.status(
                    VoiceState::Error,
                    Some(VoiceReasonCode::SynthesisFailed),
                    Some(detail),
                ),
            )?;
        }
        WorkerEvent::Started {
            command_id,
            revision,
            config,
        } => {
            tracing::info!(
                %command_id,
                revision,
                language = ?config.language,
                speaker_id = config.speaker_id,
                "edge_voice started synthesis"
            );
            start_active_command(runtime, command_id.clone(), revision, config);
            emit_status(
                node,
                outputs,
                runtime.status(VoiceState::Speaking, None, None),
            )?;
        }
        WorkerEvent::AudioChunk {
            command_id,
            priority,
            frame_id,
            timestamp_ms,
            samples,
        } => {
            if runtime.active_command_id.as_deref() != Some(&command_id) {
                tracing::debug!(%command_id, frame_id, "ignored stale TTS chunk");
                return Ok(());
            }
            let Some(pacer) = runtime.pacer.as_mut() else {
                tracing::debug!(%command_id, frame_id, "ignored TTS chunk without active pacer");
                return Ok(());
            };
            if let Err(error) = pacer.accept(TtsAudioChunk {
                command_id: command_id.clone(),
                priority,
                frame_id,
                timestamp_ms,
                samples,
            }) {
                request_worker_cancel(runtime, worker, VoiceReasonCode::SynthesisFailed);
                fail_active_command(
                    node,
                    outputs,
                    runtime,
                    &command_id,
                    VoiceReasonCode::SynthesisFailed,
                    Some(sanitized_error(error)),
                )?;
                dispatch_if_idle(runtime, worker, node, outputs)?;
            }
        }
        WorkerEvent::Completed {
            command_id,
            elapsed_ms,
            samples,
        } => {
            tracing::info!(%command_id, elapsed_ms, samples, "edge_voice synthesis completed");
            if runtime.active_command_id.as_deref() != Some(&command_id) {
                return Ok(());
            }
            emit_metric(
                node,
                outputs,
                json!({"event":"synthesis_completed","command_id":command_id.clone(),"elapsed_ms":elapsed_ms,"samples":samples}),
            )?;
            if runtime.playback_failure_pending {
                finish_active_command(
                    node,
                    outputs,
                    runtime,
                    &command_id,
                    TtsResultState::Failed,
                    Some(VoiceReasonCode::PlaybackFailed),
                    None,
                )?;
                dispatch_if_idle(runtime, worker, node, outputs)?;
            } else {
                runtime.pending_completion = Some(PendingCompletion {
                    command_id,
                    elapsed_ms,
                    samples,
                });
                complete_synthesis_if_pacer_empty(node, outputs, runtime)?;
            }
        }
        WorkerEvent::Interrupted { command_id, reason } => {
            tracing::warn!(%command_id, ?reason, "edge_voice synthesis interrupted");
            if runtime.active_command_id.as_deref() == Some(&command_id) {
                runtime.clear_pacing_for_active_command();
                let playback_failed =
                    runtime.playback_failure_pending || reason == VoiceReasonCode::PlaybackFailed;
                let state = if playback_failed {
                    TtsResultState::Failed
                } else {
                    TtsResultState::Interrupted
                };
                let terminal_reason = if playback_failed {
                    VoiceReasonCode::PlaybackFailed
                } else {
                    reason
                };
                emit_synthesis_state(
                    node,
                    outputs,
                    command_result(
                        &runtime.entity_id,
                        &command_id,
                        state,
                        Some(terminal_reason),
                        None,
                    ),
                )?;
                finish_active_command(
                    node,
                    outputs,
                    runtime,
                    &command_id,
                    state,
                    Some(terminal_reason),
                    None,
                )?;
                dispatch_if_idle(runtime, worker, node, outputs)?;
            }
        }
        WorkerEvent::Failed {
            command_id,
            reason,
            detail,
        } => {
            tracing::error!(%command_id, ?reason, %detail, "edge_voice synthesis failed");
            if runtime.active_command_id.as_deref() == Some(&command_id) {
                runtime.clear_pacing_for_active_command();
                emit_synthesis_state(
                    node,
                    outputs,
                    command_result(
                        &runtime.entity_id,
                        &command_id,
                        TtsResultState::Failed,
                        Some(reason),
                        Some(detail.clone()),
                    ),
                )?;
                finish_active_command(
                    node,
                    outputs,
                    runtime,
                    &command_id,
                    TtsResultState::Failed,
                    Some(reason),
                    Some(detail),
                )?;
                dispatch_if_idle(runtime, worker, node, outputs)?;
            }
        }
        WorkerEvent::Stopped => {}
    }
    Ok(())
}

fn start_active_command(
    runtime: &mut RuntimeState,
    command_id: String,
    revision: u64,
    config: TtsRuntimeConfig,
) {
    runtime.active_command_id = Some(command_id.clone());
    runtime.active_revision = Some(revision);
    runtime.active_config = Some(config);
    runtime.synthesis_finished = false;
    runtime.playback_failure_pending = false;
    runtime.pacer = Some(TtsPacer::new(command_id, InstantClock::default()));
    runtime.pending_completion = None;
    runtime.last_metrics_snapshot = None;
    runtime.counters = RuntimeCounters::default();
}

fn discard_stale_worker_audio(runtime: &RuntimeState, event: &WorkerEvent) -> bool {
    if let WorkerEvent::AudioChunk {
        command_id,
        frame_id,
        ..
    } = event
    {
        if runtime.active_command_id.as_deref() != Some(command_id) {
            tracing::debug!(%command_id, frame_id, "discarded stale TTS chunk before pacing");
            return true;
        }
    }
    false
}

fn emit_due_paced_chunk(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
    worker: &WorkerHandle,
) -> Result<()> {
    let due = runtime.pacer.as_mut().and_then(TtsPacer::pop_due);
    let Some(due) = due else {
        return Ok(());
    };

    let command_id = due.chunk.command_id.clone();
    let metadata = audio_metadata(
        &due.chunk.command_id,
        due.chunk.frame_id,
        due.chunk.timestamp_ms,
        due.chunk.samples.len(),
        due.chunk.priority,
    )?;
    if let Err(error) = node.send_output(
        outputs.tts_audio.clone(),
        metadata,
        Float32Array::from(due.chunk.samples),
    ) {
        request_worker_cancel(runtime, worker, VoiceReasonCode::SynthesisFailed);
        fail_active_command(
            node,
            outputs,
            runtime,
            &command_id,
            VoiceReasonCode::SynthesisFailed,
            Some(sanitized_error(error)),
        )?;
        dispatch_if_idle(runtime, worker, node, outputs)?;
        return Ok(());
    }

    if let Some(pacer) = runtime.pacer.as_ref() {
        runtime.last_metrics_snapshot = Some(pacer.snapshot());
    }
    if due.lag > Duration::ZERO {
        tracing::debug!(
            command_id = %command_id,
            lag_ms = due.lag.as_millis(),
            "edge_voice paced TTS chunk emitted after deadline"
        );
    }
    complete_synthesis_if_pacer_empty(node, outputs, runtime)?;
    Ok(())
}

fn complete_synthesis_if_pacer_empty(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
) -> Result<()> {
    if runtime
        .pacer
        .as_ref()
        .map(|pacer| pacer.has_pending())
        .unwrap_or(false)
    {
        return Ok(());
    }
    let Some(completion) = runtime.pending_completion.take() else {
        return Ok(());
    };

    if let Some(pacer) = runtime.pacer.take() {
        runtime.last_metrics_snapshot = Some(pacer.snapshot());
    }
    runtime.synthesis_finished = true;
    emit_synthesis_state(
        node,
        outputs,
        command_result(
            &runtime.entity_id,
            &completion.command_id,
            TtsResultState::Completed,
            None,
            None,
        ),
    )?;
    emit_metric(
        node,
        outputs,
        json!({
            "event":"synthesis_paced_completed",
            "command_id":completion.command_id,
            "elapsed_ms":completion.elapsed_ms,
            "samples":completion.samples,
            "pacing":runtime.last_metrics_snapshot,
        }),
    )?;
    Ok(())
}

fn fail_active_command(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
    command_id: &str,
    reason: VoiceReasonCode,
    detail: Option<String>,
) -> Result<()> {
    runtime.clear_pacing_for_active_command();
    emit_synthesis_state(
        node,
        outputs,
        command_result(
            &runtime.entity_id,
            command_id,
            TtsResultState::Failed,
            Some(reason),
            detail.clone(),
        ),
    )?;
    finish_active_command(
        node,
        outputs,
        runtime,
        command_id,
        TtsResultState::Failed,
        Some(reason),
        detail,
    )
}

fn request_worker_cancel(
    runtime: &mut RuntimeState,
    worker: &WorkerHandle,
    reason: VoiceReasonCode,
) {
    runtime.counters.cancellation_count = runtime.counters.cancellation_count.saturating_add(1);
    worker.cancel(reason);
}

fn emit_periodic_metrics(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
) -> Result<()> {
    if runtime.last_metrics_emit.elapsed() < METRICS_INTERVAL {
        return Ok(());
    }
    runtime.last_metrics_emit = Instant::now();
    emit_pacing_metrics(node, outputs, runtime, "periodic")
}

fn emit_pacing_metrics(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &RuntimeState,
    event: &str,
) -> Result<()> {
    let Some(value) = pacing_metrics_payload(runtime, event) else {
        return Ok(());
    };
    emit_metric(node, outputs, value)
}

fn pacing_metrics_payload(runtime: &RuntimeState, event: &str) -> Option<serde_json::Value> {
    let snapshot = runtime
        .pacer
        .as_ref()
        .map(TtsPacer::snapshot)
        .or(runtime.last_metrics_snapshot)
        .unwrap_or_default();
    if snapshot == PacingSnapshot::default()
        && runtime.counters == RuntimeCounters::default()
        && runtime.active_command_id.is_none()
    {
        return None;
    }
    Some(json!({
        "event": "tts_pacing_metrics",
        "phase": event,
        "active_command_id": runtime.active_command_id.clone(),
        "generated_frames": snapshot.generated_frames,
        "generated_samples": snapshot.generated_samples,
        "emitted_frames": snapshot.emitted_frames,
        "emitted_samples": snapshot.emitted_samples,
        "pending_depth": snapshot.pending_depth,
        "pacing_lag_ms": snapshot.pacing_lag_ms,
        "worker_backpressure_count": runtime.counters.worker_backpressure_count,
        "cancellation_count": runtime.counters.cancellation_count,
        "terminal_reason": runtime.counters.terminal_reason.clone(),
    }))
}

fn emit_status(
    node: &mut DoraNode,
    outputs: &Outputs,
    status: robo_rover_lib::VoiceStatus,
) -> Result<()> {
    validate_voice_status(&status)?;
    node.send_output(
        outputs.voice_status.clone(),
        Default::default(),
        to_binary(&status)?,
    )?;
    Ok(())
}

fn emit_result(
    node: &mut DoraNode,
    outputs: &Outputs,
    result: robo_rover_lib::TtsCommandResult,
) -> Result<()> {
    validate_result(&result)?;
    node.send_output(
        outputs.tts_command_result.clone(),
        Default::default(),
        to_binary(&result)?,
    )?;
    Ok(())
}

fn emit_synthesis_state(
    node: &mut DoraNode,
    outputs: &Outputs,
    result: TtsCommandResult,
) -> Result<()> {
    validate_result(&result)?;
    node.send_output(
        outputs.tts_synthesis_state.clone(),
        Default::default(),
        to_binary(&result)?,
    )?;
    Ok(())
}

fn finish_active_command(
    node: &mut DoraNode,
    outputs: &Outputs,
    runtime: &mut RuntimeState,
    command_id: &str,
    state: TtsResultState,
    reason: Option<VoiceReasonCode>,
    detail: Option<String>,
) -> Result<()> {
    runtime.record_terminal(state, reason);
    let terminal_metrics = pacing_metrics_payload(runtime, "terminal");
    runtime.busy = false;
    runtime.active_command_id = None;
    runtime.active_revision = None;
    runtime.active_config = None;
    runtime.synthesis_finished = false;
    runtime.playback_failure_pending = false;
    runtime.clear_pacing_for_active_command();
    emit_result(
        node,
        outputs,
        command_result(&runtime.entity_id, command_id, state, reason, detail),
    )?;
    emit_status(node, outputs, runtime.status(VoiceState::Ready, None, None))?;
    if let Some(value) = terminal_metrics {
        if let Err(error) = emit_metric(node, outputs, value) {
            tracing::warn!(%error, command_id, "failed to emit terminal TTS pacing metrics");
        }
    }
    Ok(())
}

fn emit_metric(node: &mut DoraNode, outputs: &Outputs, value: serde_json::Value) -> Result<()> {
    node.send_output(
        outputs.metrics.clone(),
        Default::default(),
        to_binary(&value)?,
    )?;
    Ok(())
}

fn current_voice_state(runtime: &RuntimeState) -> VoiceState {
    if runtime.busy {
        VoiceState::Speaking
    } else if runtime.ready {
        VoiceState::Ready
    } else if runtime.load_failed {
        VoiceState::Error
    } else {
        VoiceState::Loading
    }
}

fn binary_payload(data: &dyn Array) -> Result<&[u8]> {
    let array = data
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| eyre!("expected BinaryArray payload"))?;
    if array.is_empty() {
        return Err(eyre!("empty BinaryArray payload"));
    }
    Ok(array.value(0))
}

fn terminal_reason_label(state: TtsResultState, reason: Option<VoiceReasonCode>) -> String {
    if state == TtsResultState::Completed {
        return "completed".to_string();
    }
    reason
        .and_then(|value| {
            serde_json::to_value(value)
                .ok()
                .and_then(|json| json.as_str().map(str::to_string))
        })
        .unwrap_or_else(|| format!("{state:?}").to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DEFAULT_QUEUE_CAPACITY;
    use robo_rover_lib::TtsLanguage;

    fn deployment() -> DeploymentConfig {
        DeploymentConfig {
            entity_id: "rover-kiwi".to_string(),
            model_dir: "unused".into(),
            num_threads: 1,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            debug: false,
            default_runtime: TtsRuntimeConfig::default(),
        }
    }

    #[test]
    fn speaking_status_uses_active_config_snapshot() {
        let mut runtime = RuntimeState::new(deployment());
        runtime.busy = true;
        runtime.active_revision = Some(1);
        runtime.active_config = Some(TtsRuntimeConfig {
            language: TtsLanguage::En,
            speaker_id: 5,
            speed: 1.0,
            num_steps: 8,
            volume: 0.8,
        });
        runtime.current_revision = 2;
        runtime.current_config = TtsRuntimeConfig {
            language: TtsLanguage::Vi,
            speaker_id: 1,
            speed: 1.2,
            num_steps: 10,
            volume: 0.4,
        };

        let status = runtime.status(VoiceState::Speaking, None, None);
        assert_eq!(status.applied_revision, 1);
        assert_eq!(status.applied_config.language, TtsLanguage::En);
        assert_eq!(status.applied_config.speaker_id, 5);
    }

    #[test]
    fn load_failure_clears_pending_queue() {
        let mut runtime = RuntimeState::new(deployment());
        runtime.queue.enqueue(TtsCommand {
            command_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            text: "hello".to_string(),
            timestamp: 1,
            priority: robo_rover_lib::TtsPriority::Normal,
        });

        let cleared = runtime.queue.clear_ids();
        runtime.load_failed = true;
        assert_eq!(cleared, vec!["550e8400-e29b-41d4-a716-446655440000"]);
        assert_eq!(runtime.queue.len(), 0);
    }

    #[test]
    fn stale_config_does_not_mutate_current_state() {
        let mut runtime = RuntimeState::new(deployment());
        let newer = TtsRuntimeConfig {
            language: TtsLanguage::Vi,
            speaker_id: 2,
            speed: 1.1,
            num_steps: 9,
            volume: 0.7,
        };
        assert!(runtime.apply_config(TtsConfigCommand {
            revision: 3,
            config: newer.clone(),
        }));
        assert!(!runtime.apply_config(TtsConfigCommand {
            revision: 2,
            config: TtsRuntimeConfig::default(),
        }));
        assert_eq!(runtime.current_revision, 3);
        assert_eq!(runtime.current_config, newer);
    }

    #[test]
    fn clear_pacing_preserves_snapshot_for_terminal_metrics() {
        let mut runtime = RuntimeState::new(deployment());
        let mut pacer = TtsPacer::new("cmd", InstantClock::default());
        pacer
            .accept(TtsAudioChunk {
                command_id: "cmd".to_string(),
                priority: robo_rover_lib::TtsPriority::Normal,
                frame_id: 0,
                timestamp_ms: 1,
                samples: vec![0.0; 882],
            })
            .unwrap();
        runtime.pacer = Some(pacer);

        runtime.clear_pacing_for_active_command();

        let snapshot = runtime.last_metrics_snapshot.unwrap();
        assert_eq!(snapshot.generated_frames, 1);
        assert_eq!(snapshot.pending_depth, 0);
    }

    #[test]
    fn pacing_metrics_payload_includes_backpressure_cancellation_and_terminal_reason() {
        let mut runtime = RuntimeState::new(deployment());
        runtime.active_command_id = Some("cmd".to_string());
        runtime.last_metrics_snapshot = Some(PacingSnapshot {
            generated_frames: 2,
            generated_samples: 1764,
            emitted_frames: 1,
            emitted_samples: 882,
            pending_depth: 1,
            pacing_lag_ms: 7,
        });
        runtime.counters.worker_backpressure_count = 3;
        runtime.counters.cancellation_count = 1;
        runtime.record_terminal(
            TtsResultState::Interrupted,
            Some(VoiceReasonCode::InterruptedByWalkie),
        );

        let payload = pacing_metrics_payload(&runtime, "periodic").unwrap();

        assert_eq!(payload["event"], "tts_pacing_metrics");
        assert_eq!(payload["phase"], "periodic");
        assert_eq!(payload["active_command_id"], "cmd");
        assert_eq!(payload["generated_frames"], 2);
        assert_eq!(payload["emitted_frames"], 1);
        assert_eq!(payload["worker_backpressure_count"], 3);
        assert_eq!(payload["cancellation_count"], 1);
        assert_eq!(payload["terminal_reason"], "interrupted_by_walkie");
    }

    #[test]
    fn terminal_reason_uses_completed_without_reason() {
        assert_eq!(
            terminal_reason_label(TtsResultState::Completed, None),
            "completed"
        );
        assert_eq!(
            terminal_reason_label(
                TtsResultState::Failed,
                Some(VoiceReasonCode::PlaybackFailed)
            ),
            "playback_failed"
        );
    }

    #[test]
    fn command_start_resets_per_command_metrics() {
        let mut runtime = RuntimeState::new(deployment());
        runtime.last_metrics_snapshot = Some(PacingSnapshot {
            generated_frames: 9,
            generated_samples: 9,
            emitted_frames: 9,
            emitted_samples: 9,
            pending_depth: 0,
            pacing_lag_ms: 2,
        });
        runtime.counters.worker_backpressure_count = 7;
        runtime.counters.cancellation_count = 3;
        runtime.record_terminal(
            TtsResultState::Failed,
            Some(VoiceReasonCode::PlaybackFailed),
        );

        start_active_command(
            &mut runtime,
            "next".to_string(),
            4,
            TtsRuntimeConfig::default(),
        );

        assert_eq!(runtime.counters, RuntimeCounters::default());
        assert!(runtime.last_metrics_snapshot.is_none());
        assert_eq!(runtime.active_command_id.as_deref(), Some("next"));
        assert_eq!(runtime.active_revision, Some(4));
    }

    #[test]
    fn stale_worker_audio_is_discarded_before_dora_handling() {
        let mut runtime = RuntimeState::new(deployment());
        runtime.active_command_id = Some("current".to_string());
        let stale = WorkerEvent::AudioChunk {
            command_id: "stale".to_string(),
            priority: robo_rover_lib::TtsPriority::Normal,
            frame_id: 42,
            timestamp_ms: 1,
            samples: vec![0.0; 882],
        };
        let current = WorkerEvent::AudioChunk {
            command_id: "current".to_string(),
            priority: robo_rover_lib::TtsPriority::Normal,
            frame_id: 0,
            timestamp_ms: 1,
            samples: vec![0.0; 882],
        };

        assert!(discard_stale_worker_audio(&runtime, &stale));
        assert!(!discard_stale_worker_audio(&runtime, &current));
    }

    #[test]
    fn worker_drain_budget_covers_large_stale_prefix_plus_lifecycle_event() {
        let stale_prefix = 129;
        assert!(stale_prefix > 128);
        assert!(MAX_WORKER_EVENTS_PER_TICK > stale_prefix + 1);
        assert_eq!(
            MAX_WORKER_EVENTS_PER_TICK,
            crate::worker::WORKER_EVENT_CAPACITY + 1
        );
    }

    #[test]
    fn terminal_metrics_survive_until_emitted_before_next_command_reset() {
        let mut runtime = RuntimeState::new(deployment());
        start_active_command(
            &mut runtime,
            "first".to_string(),
            1,
            TtsRuntimeConfig::default(),
        );
        runtime.counters.cancellation_count = 1;
        runtime.record_terminal(
            TtsResultState::Interrupted,
            Some(VoiceReasonCode::InterruptedByWalkie),
        );

        let terminal_payload = pacing_metrics_payload(&runtime, "terminal").unwrap();

        start_active_command(
            &mut runtime,
            "second".to_string(),
            2,
            TtsRuntimeConfig::default(),
        );
        let next_payload = pacing_metrics_payload(&runtime, "periodic").unwrap();

        assert_eq!(terminal_payload["phase"], "terminal");
        assert_eq!(terminal_payload["active_command_id"], "first");
        assert_eq!(terminal_payload["cancellation_count"], 1);
        assert_eq!(terminal_payload["terminal_reason"], "interrupted_by_walkie");
        assert_eq!(next_payload["active_command_id"], "second");
        assert_eq!(next_payload["cancellation_count"], 0);
        assert!(next_payload["terminal_reason"].is_null());
    }
}
