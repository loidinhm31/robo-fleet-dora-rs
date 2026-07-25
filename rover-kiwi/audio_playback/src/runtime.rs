use std::sync::Arc;
use std::time::{Duration, Instant};

use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    DoraNode, Event, MetadataParameters, Parameter,
};
use eyre::Result;
use robo_rover_lib::{
    AudioFrameMetadata, LifecycleCommand, LifecycleComponentState, LifecycleGate,
    LifecycleReasonCode, LifecycleRole, LifecycleTarget, LifecycleTransition, PcmSampleFormat,
    TtsResultState,
};
use uuid::Uuid;

use crate::arbiter::SourceArbiter;
use crate::buffers::PlaybackBuffers;
use crate::device::default_output_plan;
use crate::playback_event::ArbiterEvent;
use crate::playback_result::{report_tts_interrupted_by_lifecycle, report_tts_result};
use crate::protocol::{parse_source_frame, AudioSource};
use crate::state::{current_time_ms, PlaybackOutputs, StateReporter};
use crate::tts_result::parse_tts_result;

const DEFAULT_TTS_BUFFER_MILLIS: usize = 1_000;
const DEFAULT_WALKIE_BUFFER_MILLIS: usize = 80;
const DEFAULT_TTS_STALL_MILLIS: u64 = 60;
const MIN_BUFFER_MILLIS: usize = 20;
const MAX_TTS_BUFFER_MILLIS: usize = 5_000;
const MAX_WALKIE_BUFFER_MILLIS: usize = 250;
const MAX_TTS_STALL_MILLIS: u64 = 250;
const METRIC_INTERVAL: Duration = Duration::from_secs(5);

pub fn run() -> Result<()> {
    tracing::info!("starting source-aware audio playback node");
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let outputs = PlaybackOutputs::new();
    let entity_id = std::env::var("ENTITY_ID").unwrap_or_else(|_| "rover-kiwi".to_owned());
    let volume = playback_volume();
    let mut reporter = StateReporter::new(entity_id.clone());
    let mut lifecycle_gate = LifecycleGate::new(LifecycleTarget {
        role: LifecycleRole::Rover,
        entity_id,
        node_id: "audio-playback".into(),
    });

    let plan = default_output_plan();
    let output_rate = plan
        .as_ref()
        .map(|plan| plan.sample_rate())
        .unwrap_or(48_000);
    let config = PlaybackRuntimeConfig::from_env(output_rate);
    let buffers = Arc::new(PlaybackBuffers::with_monitor_capacity(
        config.tts_capacity_samples,
        config.walkie_capacity_samples,
        config.monitor_capacity_samples,
    ));
    let mut arbiter = SourceArbiter::new(output_rate, buffers.clone(), config.tts_stall_timeout);
    let mut playback_monitor = PlaybackMonitor::new(output_rate);
    let mut last_metric = Instant::now();
    let opened_device = plan.and_then(|plan| plan.open(buffers.clone(), volume));
    let mut device = None;
    match opened_device {
        Ok(playback_device) => {
            tracing::info!(
                sample_rate = playback_device.sample_rate,
                channels = playback_device.channels,
                sample_format = ?playback_device.sample_format,
                volume,
                "audio playback ready"
            );
            device = Some(playback_device);
            reporter.report_consumption(
                &mut node,
                &outputs,
                crate::buffers::SOURCE_IDLE,
                0,
                arbiter.command_ids(),
            )?;
        }
        Err(error) => {
            tracing::warn!(%error, "audio output unavailable; entering explicit silent mode");
            reporter.report_unavailable(&mut node, &outputs)?;
        }
    }
    let mut available = device.is_some();
    let mut unavailable_failed_command = None;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata } => match id.as_str() {
                "lifecycle_command" => {
                    handle_lifecycle_command(
                        lifecycle_payload(data.as_ref())?,
                        &mut node,
                        &outputs,
                        &mut reporter,
                        &mut lifecycle_gate,
                        &mut arbiter,
                        &buffers,
                        &mut playback_monitor,
                        &mut device,
                        &mut available,
                        volume,
                    )?;
                }
                "tick" => {
                    if available && lifecycle_gate.admission_open() {
                        let event = buffers.take_interval_consumption();
                        reporter.report_consumption(
                            &mut node,
                            &outputs,
                            event.source,
                            event.token,
                            arbiter.command_ids(),
                        )?;
                        playback_monitor.flush(
                            &buffers,
                            &mut node,
                            &outputs,
                            event.source == crate::buffers::SOURCE_IDLE,
                        )?;
                        if let Some(event) = arbiter.tick(Instant::now()) {
                            handle_arbiter_event(event, &mut reporter, &mut node, &outputs)?;
                        }
                        arbiter.prune_command_ids();
                        if buffers.stream_errors() > 0 {
                            if let Some(event) = arbiter.fail_playback() {
                                if let ArbiterEvent::TtsPlaybackFailed { command_id } = &event {
                                    unavailable_failed_command = Some(command_id.clone());
                                }
                                handle_arbiter_event(event, &mut reporter, &mut node, &outputs)?;
                            }
                            device.take();
                            available = false;
                            reporter.report_unavailable(&mut node, &outputs)?;
                        }
                    }
                    if last_metric.elapsed() >= METRIC_INTERVAL {
                        log_playback_metrics("audio_pipeline", &buffers, &arbiter, output_rate);
                        last_metric = Instant::now();
                    }
                }
                "tts_audio" | "walkie_audio" => {
                    let source = if id.as_str() == "tts_audio" {
                        AudioSource::Tts
                    } else {
                        AudioSource::Walkie
                    };
                    if !lifecycle_gate.admission_open() {
                        continue;
                    }
                    match parse_source_frame(source, &metadata.parameters, data.as_ref()) {
                        Ok(frame) if available => {
                            if frame.normalized_samples > 0 {
                                tracing::warn!(
                                    count = frame.normalized_samples,
                                    "normalized non-finite playback samples to silence"
                                );
                            }
                            if let Some(event) = arbiter.accept(frame, Instant::now())? {
                                handle_arbiter_event(event, &mut reporter, &mut node, &outputs)?;
                            }
                        }
                        Ok(frame) => {
                            if frame.source == AudioSource::Tts {
                                if let Some(command_id) = frame.command_id {
                                    if unavailable_failed_command.as_deref() != Some(&command_id) {
                                        report_tts_result(
                                            &mut node,
                                            &outputs,
                                            reporter.entity_id(),
                                            command_id.clone(),
                                            false,
                                        )?;
                                        unavailable_failed_command = Some(command_id);
                                    }
                                }
                            }
                            tracing::debug!("discarded validated audio in silent mode");
                        }
                        Err(error) => tracing::warn!(input = %id, %error, "rejected audio frame"),
                    }
                }
                "tts_synthesis_state" if !lifecycle_gate.admission_open() => {}
                "tts_synthesis_state" => match parse_tts_result(data.as_ref()) {
                    Ok(result) if result.state == TtsResultState::Completed && !available => {
                        if unavailable_failed_command.as_deref() != Some(&result.command_id) {
                            report_tts_result(
                                &mut node,
                                &outputs,
                                reporter.entity_id(),
                                result.command_id.clone(),
                                false,
                            )?;
                            unavailable_failed_command = Some(result.command_id);
                        }
                    }
                    Ok(result) if result.state == TtsResultState::Completed => {
                        let known_command = arbiter.owns_tts(&result.command_id);
                        if let Some(event) =
                            arbiter.finish_tts(&result.command_id, Instant::now())?
                        {
                            handle_arbiter_event(event, &mut reporter, &mut node, &outputs)?;
                        } else if !known_command {
                            report_tts_result(
                                &mut node,
                                &outputs,
                                reporter.entity_id(),
                                result.command_id,
                                true,
                            )?;
                        }
                    }
                    Ok(result) => arbiter.abort_tts(&result.command_id),
                    Err(error) => tracing::warn!(%error, "rejected TTS terminal result"),
                },
                other => tracing::warn!(input = other, "ignored unexpected playback input"),
            },
            Event::Stop(_) => break,
            _ => {}
        }
    }

    let (dropped_tts, dropped_walkie) = buffers.dropped_counts();
    log_playback_metrics("audio_pipeline_total", &buffers, &arbiter, output_rate);
    tracing::info!(
        dropped_tts,
        dropped_walkie,
        stream_errors = buffers.stream_errors(),
        "audio playback stopped"
    );
    drop(device);
    Ok(())
}

struct PlaybackRuntimeConfig {
    tts_capacity_samples: usize,
    walkie_capacity_samples: usize,
    monitor_capacity_samples: usize,
    tts_stall_timeout: Duration,
}

impl PlaybackRuntimeConfig {
    fn from_env(output_rate: u32) -> Self {
        let tts_buffer_ms = env_millis(
            "PLAYBACK_TTS_BUFFER_MS",
            DEFAULT_TTS_BUFFER_MILLIS,
            MIN_BUFFER_MILLIS,
            MAX_TTS_BUFFER_MILLIS,
        );
        let walkie_buffer_ms = env_millis(
            "PLAYBACK_WALKIE_BUFFER_MS",
            DEFAULT_WALKIE_BUFFER_MILLIS,
            MIN_BUFFER_MILLIS,
            MAX_WALKIE_BUFFER_MILLIS,
        );
        let stall_ms = env_millis_u64(
            "PLAYBACK_TTS_STALL_MS",
            DEFAULT_TTS_STALL_MILLIS,
            MIN_BUFFER_MILLIS as u64,
            MAX_TTS_STALL_MILLIS,
        );
        let monitor_buffer_ms = env_millis(
            "PLAYBACK_MONITOR_BUFFER_MS",
            DEFAULT_TTS_BUFFER_MILLIS,
            MIN_BUFFER_MILLIS,
            MAX_TTS_BUFFER_MILLIS,
        );
        let tts_capacity_samples = millis_to_samples(output_rate, tts_buffer_ms).max(1);
        let walkie_capacity_samples = millis_to_samples(output_rate, walkie_buffer_ms).max(1);
        let monitor_capacity_samples = millis_to_samples(output_rate, monitor_buffer_ms).max(1);
        tracing::info!(
            output_rate,
            tts_buffer_ms,
            walkie_buffer_ms,
            monitor_buffer_ms,
            stall_ms,
            tts_capacity_samples,
            walkie_capacity_samples,
            monitor_capacity_samples,
            "audio playback buffer configuration"
        );
        Self {
            tts_capacity_samples,
            walkie_capacity_samples,
            monitor_capacity_samples,
            tts_stall_timeout: Duration::from_millis(stall_ms),
        }
    }
}

struct PlaybackMonitor {
    stream_id: Uuid,
    next_frame_id: u64,
    frame_samples: usize,
    sample_rate: u32,
    pending: Vec<f32>,
}

impl PlaybackMonitor {
    fn new(sample_rate: u32) -> Self {
        let frame_samples = millis_to_samples(sample_rate, 20).max(1);
        Self {
            stream_id: Uuid::new_v4(),
            next_frame_id: 0,
            frame_samples,
            sample_rate,
            pending: Vec::with_capacity(frame_samples),
        }
    }

    fn flush(
        &mut self,
        buffers: &PlaybackBuffers,
        node: &mut DoraNode,
        outputs: &PlaybackOutputs,
        flush_partial: bool,
    ) -> Result<()> {
        loop {
            let needed = self.frame_samples.saturating_sub(self.pending.len());
            if needed > 0 {
                let drained = buffers.drain_monitor_samples(needed);
                if drained.is_empty() {
                    break;
                }
                self.pending.extend(drained);
            }
            if self.pending.len() < self.frame_samples {
                break;
            }

            let remainder = self.pending.split_off(self.frame_samples);
            let samples = std::mem::replace(&mut self.pending, remainder);
            self.send_frame(samples, node, outputs)?;
        }
        if flush_partial && !self.pending.is_empty() {
            let samples = std::mem::take(&mut self.pending);
            self.send_frame(samples, node, outputs)?;
        }
        Ok(())
    }

    fn send_frame(
        &mut self,
        samples: Vec<f32>,
        node: &mut DoraNode,
        outputs: &PlaybackOutputs,
    ) -> Result<()> {
        let payload = float32_to_s16le(&samples);
        let metadata = AudioFrameMetadata {
            stream_id: self.stream_id,
            frame_id: self.next_frame_id,
            capture_timestamp_ms: current_time_ms(),
            sample_rate: self.sample_rate,
            channels: 1,
            sample_count: samples.len().try_into()?,
            format: PcmSampleFormat::S16Le,
        };
        metadata
            .validate_payload_len(payload.len())
            .map_err(eyre::Report::msg)?;
        self.next_frame_id = self.next_frame_id.saturating_add(1);
        let params = audio_parameters(metadata, payload.len())?;
        node.send_output(
            outputs.playback_audio.clone(),
            params,
            BinaryArray::from_vec(vec![payload.as_slice()]),
        )?;
        Ok(())
    }
}

fn audio_parameters(
    metadata: AudioFrameMetadata,
    payload_len: usize,
) -> Result<MetadataParameters> {
    metadata
        .validate_payload_len(payload_len)
        .map_err(eyre::Report::msg)?;
    Ok(MetadataParameters::from([
        (
            "stream_id".into(),
            Parameter::String(metadata.stream_id.to_string()),
        ),
        (
            "frame_id".into(),
            Parameter::Integer(metadata.frame_id.try_into()?),
        ),
        (
            "capture_timestamp_ms".into(),
            Parameter::Integer(metadata.capture_timestamp_ms.try_into()?),
        ),
        (
            "sample_rate".into(),
            Parameter::Integer(i64::from(metadata.sample_rate)),
        ),
        (
            "channels".into(),
            Parameter::Integer(i64::from(metadata.channels)),
        ),
        (
            "sample_count".into(),
            Parameter::Integer(i64::from(metadata.sample_count)),
        ),
        (
            "format".into(),
            Parameter::String(metadata.format.metadata_name().into()),
        ),
        ("size".into(), Parameter::Integer(payload_len.try_into()?)),
    ]))
}

fn float32_to_s16le(samples: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let converted = if sample <= -1.0 {
            i16::MIN
        } else {
            (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
        };
        output.extend_from_slice(&converted.to_le_bytes());
    }
    output
}

fn env_millis(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn env_millis_u64(name: &str, default: u64, min: u64, max: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn millis_to_samples(output_rate: u32, millis: usize) -> usize {
    (output_rate as usize)
        .saturating_mul(millis)
        .saturating_add(999)
        / 1_000
}

fn log_playback_metrics(
    metric: &'static str,
    buffers: &PlaybackBuffers,
    arbiter: &SourceArbiter,
    output_rate: u32,
) {
    let counts = buffers.playback_counts();
    let tts = arbiter.tts_stats();
    let dropped_walkie_duration_ms =
        samples_to_millis(counts.dropped_walkie, output_rate).unwrap_or(u64::MAX);
    tracing::info!(
        metric,
        stage = "playback",
        tts_enqueued = counts.tts_enqueued,
        tts_retired = counts.tts_retired,
        tts_cleared = counts.tts_cleared,
        walkie_enqueued = counts.walkie_enqueued,
        monitor_enqueued = counts.monitor_enqueued,
        dropped_tts = counts.dropped_tts,
        dropped_walkie = counts.dropped_walkie,
        dropped_monitor = counts.dropped_monitor,
        stream_errors = counts.stream_errors,
        tts_depth = counts.tts_depth,
        walkie_depth = counts.walkie_depth,
        monitor_depth = counts.monitor_depth,
        pending_tts_frames = tts.pending_frames,
        pending_tts_samples = tts.pending_samples,
        pending_overflows = tts.pending_overflows,
        pending_tts_frames_cleared = tts.pending_frames_cleared,
        pending_tts_samples_cleared = tts.pending_samples_cleared,
        stall_failures = tts.stall_failures,
        sequence_failures = tts.sequence_failures,
        dropped_walkie_duration_ms
    );
}

fn samples_to_millis(samples: u64, sample_rate: u32) -> Option<u64> {
    samples
        .checked_mul(1_000)?
        .checked_div(u64::from(sample_rate))
}

fn handle_arbiter_event(
    event: ArbiterEvent,
    reporter: &mut StateReporter,
    node: &mut DoraNode,
    outputs: &PlaybackOutputs,
) -> Result<()> {
    match event {
        ArbiterEvent::WalkieStarted { interrupted } => {
            reporter.report_walkie_active(node, outputs, interrupted)?
        }
        ArbiterEvent::WalkieEnded => reporter.report_walkie_idle(node, outputs)?,
        ArbiterEvent::TtsRejectedWhileWalkie => {
            tracing::debug!("rejected TTS audio while walkie is active")
        }
        ArbiterEvent::TtsPlaybackCompleted { command_id } => {
            report_tts_result(node, outputs, reporter.entity_id(), command_id, true)?
        }
        ArbiterEvent::TtsPlaybackFailed { command_id } => {
            report_tts_result(node, outputs, reporter.entity_id(), command_id, false)?
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_lifecycle_command(
    bytes: &[u8],
    node: &mut DoraNode,
    outputs: &PlaybackOutputs,
    reporter: &mut StateReporter,
    gate: &mut LifecycleGate,
    arbiter: &mut SourceArbiter,
    buffers: &Arc<PlaybackBuffers>,
    playback_monitor: &mut PlaybackMonitor,
    device: &mut Option<crate::device::PlaybackDevice>,
    available: &mut bool,
    volume: f32,
) -> Result<()> {
    let Some(transition) = begin_lifecycle_transition(bytes, gate)? else {
        return Ok(());
    };
    emit_lifecycle_status(
        node,
        outputs,
        gate,
        match transition {
            LifecycleTransition::Quiesce => LifecycleComponentState::Quiescing,
            LifecycleTransition::Resume => LifecycleComponentState::Resuming,
        },
        None,
    )?;

    match transition {
        LifecycleTransition::Quiesce => {
            if let Some(command_id) = arbiter.interrupt_for_lifecycle() {
                report_tts_interrupted_by_lifecycle(
                    node,
                    outputs,
                    reporter.entity_id(),
                    command_id,
                )?;
            }
            buffers.clear_all();
            playback_monitor.pending.clear();
            device.take();
            *available = false;
            reporter.report_consumption(
                node,
                outputs,
                crate::buffers::SOURCE_IDLE,
                0,
                arbiter.command_ids(),
            )?;
            gate.complete(transition);
            emit_lifecycle_status(node, outputs, gate, LifecycleComponentState::Quiesced, None)?;
        }
        LifecycleTransition::Resume => {
            if *available && device.is_some() {
                gate.complete(transition);
                return emit_lifecycle_status(
                    node,
                    outputs,
                    gate,
                    LifecycleComponentState::Running,
                    None,
                );
            }
            buffers.clear_all();
            playback_monitor.pending.clear();
            match default_output_plan().and_then(|plan| plan.open(buffers.clone(), volume)) {
                Ok(playback_device) => {
                    *device = Some(playback_device);
                    *available = true;
                    gate.complete(transition);
                    emit_lifecycle_status(
                        node,
                        outputs,
                        gate,
                        LifecycleComponentState::Running,
                        None,
                    )?;
                }
                Err(error) => {
                    tracing::warn!(%error, "audio playback resume could not open output device");
                    *available = false;
                    emit_lifecycle_status(
                        node,
                        outputs,
                        gate,
                        LifecycleComponentState::Failed,
                        Some(LifecycleReasonCode::Internal),
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn begin_lifecycle_transition(
    bytes: &[u8],
    gate: &mut LifecycleGate,
) -> Result<Option<LifecycleTransition>> {
    let command = serde_json::from_slice::<LifecycleCommand>(bytes)?;
    if !gate.accepts_target(&command.target) {
        tracing::trace!(target = ?command.target, "ignoring lifecycle command for another workload");
        return Ok(None);
    }
    gate.begin(&command).map_err(eyre::Report::msg)
}

fn lifecycle_payload(data: &dyn Array) -> Result<&[u8]> {
    let array = data
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| eyre::eyre!("lifecycle command must be binary"))?;
    if array.len() != 1 {
        return Err(eyre::eyre!("lifecycle command must contain one payload"));
    }
    Ok(array.value(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{LifecycleDesiredState, LIFECYCLE_PROTOCOL_VERSION};

    fn sibling_command() -> LifecycleCommand {
        LifecycleCommand {
            protocol_version: LIFECYCLE_PROTOCOL_VERSION,
            request_id: "550e8400-e29b-41d4-a716-446655440000".into(),
            manager_epoch: 1,
            target: LifecycleTarget {
                role: LifecycleRole::Rover,
                entity_id: "rover-kiwi".into(),
                node_id: "edge-voice".into(),
            },
            desired_state: LifecycleDesiredState::Quiesced,
            expected_revision: 0,
            issued_at_ms: 1,
            expires_at_ms: 2,
            origin: Default::default(),
            transition_id: None,
        }
    }

    #[test]
    fn sibling_lifecycle_command_leaves_audio_playback_running() {
        let mut gate = LifecycleGate::new(LifecycleTarget {
            role: LifecycleRole::Rover,
            entity_id: "rover-kiwi".into(),
            node_id: "audio-playback".into(),
        });
        let payload = serde_json::to_vec(&sibling_command()).unwrap();

        assert_eq!(
            begin_lifecycle_transition(&payload, &mut gate).unwrap(),
            None
        );
        assert!(gate.admission_open());
        assert_eq!(gate.authority(), (0, 0));
        assert_eq!(gate.desired_state(), None);
    }
}

fn emit_lifecycle_status(
    node: &mut DoraNode,
    outputs: &PlaybackOutputs,
    gate: &LifecycleGate,
    state: LifecycleComponentState,
    reason: Option<LifecycleReasonCode>,
) -> Result<()> {
    let status = gate.status(state, reason, current_time_ms());
    status.validate().map_err(eyre::Report::msg)?;
    let bytes = serde_json::to_vec(&status)?;
    node.send_output(
        outputs.lifecycle_component_status.clone(),
        MetadataParameters::default(),
        BinaryArray::from_vec(vec![bytes.as_slice()]),
    )?;
    Ok(())
}

fn playback_volume() -> f32 {
    std::env::var("PLAYBACK_VOLUME")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(1.0)
}
