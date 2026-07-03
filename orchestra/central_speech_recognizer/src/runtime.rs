use crate::audio_input::{parse_browser, parse_rover};
use crate::browser_control::handle_control;
use crate::config::SttConfig;
use crate::decoder::{self, DecodeSubmitter, SharedNode, SherpaDecoder, SubmitResult};
use crate::metrics::RuntimeMetrics;
use crate::segmenter::sherpa_factory;
use crate::session::{DecodeJob, SessionManager};
use crate::startup::wait_for_initialization;
use crate::status::{build_status, emit_status, sanitize_startup_error, startup_profile};
use dora_node_api::{DoraNode, Event};
use eyre::{eyre, Result};
use robo_rover_lib::{SttState, SttStatus};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub fn run() -> Result<()> {
    let (node, mut events) = DoraNode::init_from_env()?;
    let node = Arc::new(Mutex::new(node));
    let profile = startup_profile();
    let loading = build_status(SttState::Loading, profile, None);
    emit_status(&node, &loading)?;

    let initialized = wait_for_initialization(&node, &mut events, &loading)?;
    let Some(initialized) = initialized else {
        return Ok(());
    };
    match initialized {
        Ok((config, models)) => {
            drop(models.vad);
            let ready = build_status(SttState::Ready, config.models.profile, None);
            if let Err(error) =
                run_ready(node.clone(), &mut events, config, models.recognizer, ready)
            {
                enter_error_state(node, &mut events, profile, error)
            } else {
                Ok(())
            }
        }
        Err(error) => enter_error_state(node, &mut events, profile, error),
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
                        .map(|jobs| submit_jobs(jobs, &submitter, &mut metrics)),
                    "stt_status_request" => emit_status(&node, &status),
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
}
