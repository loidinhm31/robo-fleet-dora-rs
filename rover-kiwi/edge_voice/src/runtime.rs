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
    worker::{self, WorkerEvent, WorkerHandle},
};

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
        drain_worker_events(&mut node, &outputs, &mut runtime, &worker, &worker_rx)?;

        match events.recv_timeout(Duration::from_millis(20)) {
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

    drain_worker_events(&mut node, &outputs, &mut runtime, &worker, &worker_rx)?;
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
            handle_tts_command(command, node, outputs, runtime)?;
        }
        "tts_config_command" | "tts_config" => {
            let command = parse_config_command(binary_payload(data)?)?;
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
                if runtime.synthesis_finished {
                    if let Some(command_id) = runtime.active_command_id.clone() {
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
                } else {
                    worker.cancel(VoiceReasonCode::InterruptedByWalkie);
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
                        worker.cancel(VoiceReasonCode::PlaybackFailed);
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
    while let Ok(event) = worker_rx.try_recv() {
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
        match worker_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(event) => {
                let stopped = matches!(event, WorkerEvent::Stopped);
                handle_worker_event(node, outputs, runtime, worker, event)?;
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
            runtime.active_command_id = Some(command_id);
            runtime.active_revision = Some(revision);
            runtime.active_config = Some(config);
            runtime.synthesis_finished = false;
            runtime.playback_failure_pending = false;
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
            let metadata =
                audio_metadata(&command_id, frame_id, timestamp_ms, samples.len(), priority)?;
            node.send_output(
                outputs.tts_audio.clone(),
                metadata,
                Float32Array::from(samples),
            )?;
        }
        WorkerEvent::Completed {
            command_id,
            elapsed_ms,
            samples,
        } => {
            if runtime.active_command_id.as_deref() != Some(&command_id) {
                return Ok(());
            }
            emit_metric(
                node,
                outputs,
                json!({"event":"synthesis_completed","command_id":command_id,"elapsed_ms":elapsed_ms,"samples":samples}),
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
                runtime.synthesis_finished = true;
                emit_synthesis_state(
                    node,
                    outputs,
                    command_result(
                        &runtime.entity_id,
                        &command_id,
                        TtsResultState::Completed,
                        None,
                        None,
                    ),
                )?;
            }
        }
        WorkerEvent::Interrupted { command_id, reason } => {
            if runtime.active_command_id.as_deref() == Some(&command_id) {
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
            if runtime.active_command_id.as_deref() == Some(&command_id) {
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
    runtime.busy = false;
    runtime.active_command_id = None;
    runtime.active_revision = None;
    runtime.active_config = None;
    runtime.synthesis_finished = false;
    runtime.playback_failure_pending = false;
    emit_result(
        node,
        outputs,
        command_result(&runtime.entity_id, command_id, state, reason, detail),
    )?;
    emit_status(node, outputs, runtime.status(VoiceState::Ready, None, None))?;
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
}
