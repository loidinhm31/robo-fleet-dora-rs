use std::sync::Arc;
use std::time::Instant;

use dora_node_api::{DoraNode, Event};
use eyre::Result;

use crate::arbiter::SourceArbiter;
use crate::buffers::PlaybackBuffers;
use crate::device::default_output_plan;
use crate::playback_event::ArbiterEvent;
use crate::playback_result::report_tts_result;
use crate::protocol::{parse_source_frame, AudioSource};
use crate::state::{PlaybackOutputs, StateReporter};
use crate::tts_result::parse_tts_result;
use robo_rover_lib::TtsResultState;

const TTS_BUFFER_SECONDS: usize = 5;
const WALKIE_BUFFER_MILLIS: usize = 250;

pub fn run() -> Result<()> {
    tracing::info!("starting source-aware audio playback node");
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let outputs = PlaybackOutputs::new();
    let entity_id = std::env::var("ENTITY_ID").unwrap_or_else(|_| "rover-kiwi".to_owned());
    let volume = playback_volume();
    let mut reporter = StateReporter::new(entity_id.clone());

    let plan = default_output_plan();
    let output_rate = plan
        .as_ref()
        .map(|plan| plan.sample_rate())
        .unwrap_or(48_000);
    let buffers = Arc::new(PlaybackBuffers::new(
        output_rate as usize * TTS_BUFFER_SECONDS,
        output_rate as usize * WALKIE_BUFFER_MILLIS / 1_000,
    ));
    let mut arbiter = SourceArbiter::new(output_rate, buffers.clone());
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
                "tick" => {
                    if available {
                        while let Some(event) = buffers.pop_consumption_event() {
                            reporter.report_consumption(
                                &mut node,
                                &outputs,
                                event.source,
                                event.token,
                                arbiter.command_ids(),
                            )?;
                        }
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
                }
                "tts_audio" | "walkie_audio" => {
                    let source = if id.as_str() == "tts_audio" {
                        AudioSource::Tts
                    } else {
                        AudioSource::Walkie
                    };
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
                        if let Some(event) = arbiter.finish_tts(&result.command_id)? {
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
    tracing::info!(
        dropped_tts,
        dropped_walkie,
        stream_errors = buffers.stream_errors(),
        consumption_event_overflows = buffers.consumption_event_overflows(),
        "audio playback stopped"
    );
    drop(device);
    Ok(())
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

fn playback_volume() -> f32 {
    std::env::var("PLAYBACK_VOLUME")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(1.0)
}
