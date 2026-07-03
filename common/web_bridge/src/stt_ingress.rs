use crate::security::validation::{validate_voice_audio_frame, validate_voice_stream_format};
use crate::stt_bridge::{BridgeInner, SttBridgeConfig};
use crate::stt_protocol::{
    BrowserControlOutput, SttDoraMessage, VoiceCommandAudioFrame, VoiceCommandControl,
};
use std::time::Instant;
use uuid::Uuid;

pub(crate) fn handle_control(
    inner: &mut BridgeInner,
    config: SttBridgeConfig,
    owner_socket: &str,
    control: VoiceCommandControl,
    selected_target: Option<&str>,
    target_is_active: bool,
) -> Result<(), String> {
    let now = Instant::now();
    match control {
        VoiceCommandControl::Start {
            stream_id,
            sample_rate,
            channels,
        } => {
            validate_voice_stream_format(sample_rate, channels)?;
            let target = selected_target
                .filter(|target| !target.trim().is_empty())
                .ok_or_else(|| "no rover selected for browser speech".to_string())?;
            if !target_is_active {
                return Err("selected rover is not active".into());
            }
            let future_active = inner.streams.active_count().saturating_add(1);
            let future_total = inner.streams.len().saturating_add(1);
            let max_active = (config.queue_capacity / 4).max(1);
            let queued = inner.queue.len() + usize::from(inner.delivery_in_flight);
            if future_active > max_active
                || future_total > config.queue_capacity
                || queued >= config.queue_capacity.saturating_sub(future_active)
            {
                return Err("browser speech transport capacity reached".into());
            }
            inner.streams.start(
                stream_id,
                owner_socket.to_owned(),
                target.to_owned(),
                sample_rate,
                channels,
                now,
            )?;
            inner
                .queue
                .push_back(SttDoraMessage::Control(BrowserControlOutput::Start {
                    stream_id,
                    sample_rate,
                    channels,
                    target_entity_id: target.to_owned(),
                }));
            Ok(())
        }
        VoiceCommandControl::Stop { stream_id } => {
            if inner
                .streams
                .close(stream_id, owner_socket, now, config.closing_ttl)?
            {
                enqueue_marked_close(inner, stream_id);
            }
            Ok(())
        }
    }
}

pub(crate) fn handle_audio(
    inner: &mut BridgeInner,
    config: SttBridgeConfig,
    owner_socket: &str,
    frame: VoiceCommandAudioFrame,
) -> Result<(), String> {
    validate_voice_audio_frame(
        frame.sample_rate,
        frame.channels,
        frame.sample_count,
        &frame.audio_data,
    )?;
    let now = Instant::now();
    let target = match inner.streams.accept_frame(owner_socket, &frame, now) {
        Ok(target) => target,
        Err(error) => {
            if error.terminates_owner_stream()
                && inner
                    .streams
                    .force_close(frame.stream_id, now, config.closing_ttl)
            {
                enqueue_marked_close(inner, frame.stream_id);
            }
            return Err(error.message);
        }
    };
    let reserved_stops = inner.streams.active_count();
    let queued = inner.queue.len() + usize::from(inner.delivery_in_flight);
    if queued >= config.queue_capacity.saturating_sub(reserved_stops) {
        inner.metrics.queue_drops = inner.metrics.queue_drops.saturating_add(1);
        tracing::warn!(
            metric = "browser_stt_queue_drop",
            queue_drops = inner.metrics.queue_drops,
            stream_id = %frame.stream_id,
            "Dropping newest browser speech frame and terminating stream"
        );
        if inner
            .streams
            .force_close(frame.stream_id, now, config.closing_ttl)
        {
            enqueue_marked_close(inner, frame.stream_id);
        }
        return Err("browser speech queue full; stream terminated".into());
    }
    inner.queue.push_back(SttDoraMessage::Audio {
        frame,
        target_entity_id: target,
    });
    Ok(())
}

pub(crate) fn enqueue_marked_close(inner: &mut BridgeInner, stream_id: Uuid) {
    if inner
        .queue
        .iter()
        .any(|message| message.stream_id() == Some(stream_id) && message.is_start())
    {
        inner
            .queue
            .retain(|message| message.stream_id() != Some(stream_id));
        inner.streams.remove(stream_id);
    } else {
        inner
            .queue
            .push_back(SttDoraMessage::Control(BrowserControlOutput::Stop {
                stream_id,
            }));
    }
    inner.metrics.terminated_streams = inner.metrics.terminated_streams.saturating_add(1);
}
