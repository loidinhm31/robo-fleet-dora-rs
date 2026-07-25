use crate::audio_input::{parse_browser, parse_rover};
use crate::browser_control::handle_control;
use crate::config::SttConfig;
use crate::decoder::{self, DecodeSubmitter, SharedNode, SherpaDecoder, SubmitResult};
use crate::metrics::RuntimeMetrics;
use crate::segmenter::sherpa_factory;
use crate::session::{DecodeJob, SessionManager};
use crate::startup::wait_for_initialization;
use crate::status::{build_status, emit_status, sanitize_startup_error, startup_profile};
use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::{eyre, Result};
use robo_rover_lib::{
    LifecycleCommand, LifecycleComponentState, LifecycleGate, LifecycleReasonCode, LifecycleRole,
    LifecycleTarget, LifecycleTransition, SttState, SttStatus,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn run() -> Result<()> {
    let (node, mut events) = DoraNode::init_from_env()?;
    let node = Arc::new(Mutex::new(node));
    let profile = startup_profile();
    let loading = build_status(SttState::Loading, profile, None);
    emit_status(&node, &loading)?;
    let lifecycle_status_output = DataId::from("lifecycle_component_status".to_owned());
    let mut lifecycle_gate = LifecycleGate::new(lifecycle_target());

    let initialized = wait_for_initialization(
        &node,
        &mut events,
        &loading,
        &mut lifecycle_gate,
        &lifecycle_status_output,
    )?;
    let Some(initialized) = initialized else {
        return Ok(());
    };
    match initialized {
        Ok((config, models)) => {
            if lifecycle_gate.desired_state()
                == Some(robo_rover_lib::LifecycleDesiredState::Quiesced)
            {
                drop(models.vad);
                drop(models.recognizer);
                lifecycle_gate.complete(LifecycleTransition::Quiesce);
                send_lifecycle_status(
                    &node,
                    &lifecycle_status_output,
                    &lifecycle_gate,
                    LifecycleComponentState::Quiesced,
                    None,
                )?;
                return run_quiesced(
                    node,
                    &mut events,
                    &mut lifecycle_gate,
                    &lifecycle_status_output,
                );
            }
            drop(models.vad);
            let ready = build_status(SttState::Ready, config.models.profile, None);
            if lifecycle_gate.desired_state()
                == Some(robo_rover_lib::LifecycleDesiredState::Running)
            {
                lifecycle_gate.complete(LifecycleTransition::Resume);
                send_lifecycle_status(
                    &node,
                    &lifecycle_status_output,
                    &lifecycle_gate,
                    LifecycleComponentState::Running,
                    None,
                )?;
            }
            if let Err(error) = run_ready(
                node.clone(),
                &mut events,
                config,
                models.recognizer,
                ready,
                &mut lifecycle_gate,
                &lifecycle_status_output,
            ) {
                enter_error_state(node, &mut events, profile, error)
            } else {
                Ok(())
            }
        }
        Err(error) => {
            if lifecycle_gate.desired_state().is_some() {
                send_lifecycle_status(
                    &node,
                    &lifecycle_status_output,
                    &lifecycle_gate,
                    LifecycleComponentState::Failed,
                    Some(LifecycleReasonCode::Internal),
                )?;
            }
            enter_error_state(node, &mut events, profile, error)
        }
    }
}

fn lifecycle_target() -> LifecycleTarget {
    LifecycleTarget {
        role: LifecycleRole::Orchestra,
        entity_id: "orchestra".into(),
        node_id: "central-speech-recognizer".into(),
    }
}

fn enter_error_state(
    node: SharedNode,
    events: &mut dora_node_api::EventStream,
    profile: robo_rover_lib::SttProfile,
    error: eyre::Report,
) -> Result<()> {
    let message = sanitize_startup_error(&error);
    tracing::error!(error = %message, "central speech recognizer unavailable");
    let status = build_status(SttState::Error, profile, Some(message));
    emit_status(&node, &status)?;
    run_unavailable(node, events, status)
}

fn run_ready(
    node: SharedNode,
    events: &mut dora_node_api::EventStream,
    config: SttConfig,
    recognizer: sherpa_onnx::OfflineRecognizer,
    status: SttStatus,
    lifecycle_gate: &mut LifecycleGate,
    lifecycle_status_output: &DataId,
) -> Result<()> {
    let factory = sherpa_factory(config.clone());
    let mut sessions = SessionManager::new(factory);
    let (submitter, worker) = decoder::spawn(
        config.decode_queue_capacity,
        Box::new(SherpaDecoder::new(recognizer)),
        config.models.profile,
        node.clone(),
    )?;
    emit_status(&node, &status)?;
    let mut metrics = RuntimeMetrics::new();

    while let Some(event) = events.recv_timeout(Duration::from_secs(1)) {
        match event {
            Event::Input { id, metadata, data } => {
                let result = match id.as_str() {
                    "audio_rover" => parse_rover(&metadata.parameters, data.as_ref())
                        .and_then(|input| sessions.accept_rover(input))
                        .map(|outcome| record_outcome(outcome, &submitter, &mut metrics)),
                    "audio_browser" => parse_browser(&metadata.parameters, data.as_ref())
                        .and_then(|input| sessions.accept_browser(input))
                        .map(|outcome| record_outcome(outcome, &submitter, &mut metrics)),
                    "browser_control" => handle_control(data.as_ref(), &mut sessions)
                        .map(|outcome| record_non_frame_outcome(outcome, &submitter, &mut metrics)),
                    "stt_status_request" => emit_status(&node, &status),
                    "lifecycle_command" => {
                        match lifecycle_transition(data.as_ref(), lifecycle_gate) {
                            Ok(Some(LifecycleTransition::Quiesce)) => {
                                send_lifecycle_status(
                                    &node,
                                    lifecycle_status_output,
                                    lifecycle_gate,
                                    LifecycleComponentState::Cancelling,
                                    Some(LifecycleReasonCode::InterruptedByLifecycle),
                                )?;
                                let cancelled = sessions.cancel_all_for_lifecycle();
                                // Session-owned VAD/resampler resources and the
                                // VAD factory are no longer reachable before
                                // we acknowledge quiesce. The recognizer
                                // belongs to the decoder worker and is dropped
                                // by its successful join below.
                                drop(sessions);
                                submitter.close_admission();
                                drop(submitter);
                                if worker.join().is_err() {
                                    tracing::error!(
                                        "Sherpa decode worker panicked during lifecycle quiesce"
                                    );
                                    send_lifecycle_status(
                                        &node,
                                        lifecycle_status_output,
                                        lifecycle_gate,
                                        LifecycleComponentState::Failed,
                                        Some(LifecycleReasonCode::Internal),
                                    )?;
                                    return Ok(());
                                }
                                tracing::info!(
                                    cancelled_streams = cancelled.len(),
                                    "discarded active STT streams for lifecycle quiesce"
                                );
                                lifecycle_gate.complete(LifecycleTransition::Quiesce);
                                send_lifecycle_status(
                                    &node,
                                    lifecycle_status_output,
                                    lifecycle_gate,
                                    LifecycleComponentState::Quiesced,
                                    None,
                                )?;
                                return run_quiesced(
                                    node,
                                    events,
                                    lifecycle_gate,
                                    lifecycle_status_output,
                                );
                            }
                            Ok(Some(LifecycleTransition::Resume)) => {
                                send_lifecycle_status(
                                    &node,
                                    lifecycle_status_output,
                                    lifecycle_gate,
                                    LifecycleComponentState::Resuming,
                                    None,
                                )?;
                                lifecycle_gate.complete(LifecycleTransition::Resume);
                                send_lifecycle_status(
                                    &node,
                                    lifecycle_status_output,
                                    lifecycle_gate,
                                    LifecycleComponentState::Running,
                                    None,
                                )?;
                                Ok(())
                            }
                            Ok(None) => Ok(()),
                            Err(error) => {
                                tracing::warn!(%error, "rejected central STT lifecycle command");
                                Ok(())
                            }
                        }
                    }
                    other => Err(eyre!("unexpected central STT input: {other}")),
                };
                if let Err(error) = result {
                    metrics.validation_errors += 1;
                    tracing::warn!(input = %id, %error, "rejected central STT input");
                }
            }
            Event::Stop(_) => break,
            Event::InputClosed { id } => {
                if closes_browser_sessions(id.as_str()) {
                    let jobs = sessions.flush_all_browsers();
                    submit_jobs(jobs, &submitter, &mut metrics);
                }
                tracing::debug!(input = %id, "central STT input closed");
            }
            Event::Error(error) if is_timeout_error(&error) => {}
            Event::Error(error) => tracing::warn!(%error, "central STT event stream error"),
            _ => {}
        }
        metrics.log_if_due();
    }
    submit_jobs(sessions.flush_all_browsers(), &submitter, &mut metrics);
    drop(submitter);
    if worker.join().is_err() {
        tracing::error!("Sherpa decode worker panicked");
    }
    metrics.log_shutdown();
    Ok(())
}

/// Waits with admission closed until a newer Running transition has rebuilt the
/// native model resources. Incoming audio is deliberately discarded here.
fn run_quiesced(
    node: SharedNode,
    events: &mut dora_node_api::EventStream,
    lifecycle_gate: &mut LifecycleGate,
    lifecycle_status_output: &DataId,
) -> Result<()> {
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } if id.as_str() == "lifecycle_command" => {
                match lifecycle_transition(data.as_ref(), lifecycle_gate) {
                    Ok(Some(LifecycleTransition::Resume)) => {
                        send_lifecycle_status(
                            &node,
                            lifecycle_status_output,
                            lifecycle_gate,
                            LifecycleComponentState::Resuming,
                            None,
                        )?;
                        let profile = startup_profile();
                        let loading = build_status(SttState::Loading, profile, None);
                        emit_status(&node, &loading)?;
                        let initialized = wait_for_initialization(
                            &node,
                            events,
                            &loading,
                            lifecycle_gate,
                            lifecycle_status_output,
                        )?;
                        let Some(initialized) = initialized else {
                            return Ok(());
                        };
                        match initialized {
                            Ok((config, models)) => {
                                if lifecycle_gate.desired_state()
                                    == Some(robo_rover_lib::LifecycleDesiredState::Quiesced)
                                {
                                    drop(models.vad);
                                    drop(models.recognizer);
                                    lifecycle_gate.complete(LifecycleTransition::Quiesce);
                                    send_lifecycle_status(
                                        &node,
                                        lifecycle_status_output,
                                        lifecycle_gate,
                                        LifecycleComponentState::Quiesced,
                                        None,
                                    )?;
                                    continue;
                                }
                                drop(models.vad);
                                let ready =
                                    build_status(SttState::Ready, config.models.profile, None);
                                emit_status(&node, &ready)?;
                                lifecycle_gate.complete(LifecycleTransition::Resume);
                                send_lifecycle_status(
                                    &node,
                                    lifecycle_status_output,
                                    lifecycle_gate,
                                    LifecycleComponentState::Running,
                                    None,
                                )?;
                                return run_ready(
                                    node,
                                    events,
                                    config,
                                    models.recognizer,
                                    ready,
                                    lifecycle_gate,
                                    lifecycle_status_output,
                                );
                            }
                            Err(error) => {
                                let message = sanitize_startup_error(&error);
                                emit_status(
                                    &node,
                                    &build_status(SttState::Error, profile, Some(message)),
                                )?;
                                send_lifecycle_status(
                                    &node,
                                    lifecycle_status_output,
                                    lifecycle_gate,
                                    LifecycleComponentState::Failed,
                                    Some(LifecycleReasonCode::Internal),
                                )?;
                            }
                        }
                    }
                    Ok(Some(LifecycleTransition::Quiesce)) => {
                        // The resources were already released by the previous
                        // quiesce, but this is a newer authority revision and
                        // still needs an explicit acknowledgement.
                        lifecycle_gate.complete(LifecycleTransition::Quiesce);
                        send_lifecycle_status(
                            &node,
                            lifecycle_status_output,
                            lifecycle_gate,
                            LifecycleComponentState::Quiesced,
                            None,
                        )?;
                    }
                    Ok(None) => {}
                    Err(error) => tracing::warn!(%error, "rejected central STT lifecycle command"),
                }
            }
            Event::Input { id, .. } if id.as_str() == "stt_status_request" => {
                emit_status(
                    &node,
                    &build_status(SttState::Loading, startup_profile(), None),
                )?;
            }
            Event::Input { id, .. } => {
                tracing::debug!(input = %id, "discarded STT input while lifecycle-quiesced");
            }
            Event::Stop(_) => return Ok(()),
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn lifecycle_transition(
    data: &dyn Array,
    gate: &mut LifecycleGate,
) -> Result<Option<LifecycleTransition>, String> {
    let array = data
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| "lifecycle command must be binary".to_owned())?;
    if array.len() != 1 {
        return Err("lifecycle command must contain exactly one item".into());
    }
    let command: LifecycleCommand = serde_json::from_slice(array.value(0))
        .map_err(|_| "invalid lifecycle command".to_owned())?;
    gate.begin(&command)
}

pub(crate) fn send_lifecycle_status(
    node: &SharedNode,
    output: &DataId,
    gate: &LifecycleGate,
    state: LifecycleComponentState,
    reason: Option<LifecycleReasonCode>,
) -> Result<()> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let status = gate.status(state, reason, timestamp);
    let json = serde_json::to_vec(&status)?;
    let array = BinaryArray::from_vec(vec![json.as_slice()]);
    node.lock()
        .map_err(|_| eyre!("Dora node lock poisoned"))?
        .send_output(output.clone(), Default::default(), array)
        .map_err(Into::into)
}

fn closes_browser_sessions(input_id: &str) -> bool {
    matches!(input_id, "audio_browser" | "browser_control")
}

fn is_timeout_error(error: &str) -> bool {
    error.starts_with("Timeout event stream error:")
}

fn run_unavailable(
    node: SharedNode,
    events: &mut dora_node_api::EventStream,
    status: SttStatus,
) -> Result<()> {
    let mut rejected_audio = 0u64;
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } if id.as_str() == "stt_status_request" => {
                emit_status(&node, &status)?
            }
            Event::Input { id, .. } => {
                rejected_audio += 1;
                tracing::warn!(input = %id, rejected_audio, "speech input rejected while STT unavailable");
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}

fn record_outcome(
    outcome: crate::session::FrameOutcome,
    submitter: &DecodeSubmitter,
    metrics: &mut RuntimeMetrics,
) {
    metrics.frames += 1;
    metrics.sequence_resets += u64::from(outcome.sequence_reset);
    submit_jobs(outcome.jobs, submitter, metrics);
}

fn record_non_frame_outcome(
    outcome: crate::session::FrameOutcome,
    submitter: &DecodeSubmitter,
    metrics: &mut RuntimeMetrics,
) {
    metrics.sequence_resets += u64::from(outcome.sequence_reset);
    submit_jobs(outcome.jobs, submitter, metrics);
}

fn submit_jobs(jobs: Vec<DecodeJob>, submitter: &DecodeSubmitter, metrics: &mut RuntimeMetrics) {
    for job in jobs {
        metrics.speech_segments += 1;
        match submitter.try_submit(job) {
            SubmitResult::Submitted => {}
            SubmitResult::Full | SubmitResult::Disconnected => {
                metrics.queue_drops += 1;
                tracing::warn!(
                    "dropping newest speech segment because decode queue is unavailable"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lifecycle_command(target: LifecycleTarget) -> LifecycleCommand {
        LifecycleCommand {
            protocol_version: 1,
            request_id: "550e8400-e29b-41d4-a716-446655440000".into(),
            manager_epoch: 9,
            target,
            desired_state: robo_rover_lib::LifecycleDesiredState::Quiesced,
            expected_revision: 0,
            issued_at_ms: 1,
            expires_at_ms: 10,
            origin: Default::default(),
            transition_id: None,
        }
    }

    #[test]
    fn browser_input_closure_triggers_session_flush() {
        assert!(closes_browser_sessions("audio_browser"));
        assert!(closes_browser_sessions("browser_control"));
        assert!(!closes_browser_sessions("audio_rover"));
    }

    #[test]
    fn only_dora_timeout_errors_are_suppressed() {
        assert!(is_timeout_error(
            "Timeout event stream error: Receiver timed out"
        ));
        assert!(!is_timeout_error("fatal event stream error: disconnected"));
    }

    #[test]
    fn lifecycle_command_closes_stt_admission_for_its_exact_target() {
        let mut gate = LifecycleGate::new(lifecycle_target());
        let command = lifecycle_command(lifecycle_target());
        let json = serde_json::to_vec(&command).unwrap();
        let array = BinaryArray::from_vec(vec![json.as_slice()]);

        assert_eq!(
            lifecycle_transition(&array, &mut gate).unwrap(),
            Some(LifecycleTransition::Quiesce)
        );
        assert!(!gate.admission_open());
    }

    #[test]
    fn lifecycle_command_rejects_a_different_workload_target() {
        let mut gate = LifecycleGate::new(lifecycle_target());
        let mut foreign = lifecycle_target();
        foreign.node_id = "media-recorder".into();
        let json = serde_json::to_vec(&lifecycle_command(foreign)).unwrap();
        let array = BinaryArray::from_vec(vec![json.as_slice()]);

        assert!(lifecycle_transition(&array, &mut gate).is_err());
        assert!(gate.admission_open());
    }
}
