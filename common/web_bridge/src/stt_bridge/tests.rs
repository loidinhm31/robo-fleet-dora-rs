use super::*;
use crate::stt_protocol::BrowserControlOutput;
use robo_rover_lib::{SpeechTranscription, SttProfile, SttSourceKind, SttState};
use uuid::Uuid;

pub(super) fn bridge(capacity: usize) -> SttBridge {
    SttBridge::new(SttBridgeConfig {
        queue_capacity: capacity,
        stream_idle_ttl: Duration::from_secs(30),
        closing_ttl: Duration::from_secs(15),
    })
}

pub(super) fn start(id: Uuid) -> VoiceCommandControl {
    VoiceCommandControl::Start {
        stream_id: id,
        sample_rate: 48_000,
        channels: 1,
    }
}

pub(super) fn audio(id: Uuid, frame_id: u64) -> VoiceCommandAudioFrame {
    VoiceCommandAudioFrame {
        stream_id: id,
        frame_id,
        sample_rate: 48_000,
        channels: 1,
        sample_count: 2,
        audio_data: vec![0.0, 0.1],
    }
}

pub(super) fn deliver_next(bridge: &SttBridge) -> Option<SttDoraMessage> {
    let message = bridge.pop_message();
    if message.is_some() {
        bridge.complete_delivery();
    }
    message
}

mod ingress;
mod lifecycle;
mod routing;
