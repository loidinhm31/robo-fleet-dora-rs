use crate::stt_bridge::BridgeInner;
use robo_rover_lib::{SpeechTranscription, SttSourceKind};
use uuid::Uuid;

#[derive(Debug, PartialEq, Eq)]
pub enum TranscriptRoute {
    Browser { socket_id: String },
    RoverBroadcast,
    Drop(String),
}

pub(crate) fn route_transcription(
    inner: &mut BridgeInner,
    transcription: &SpeechTranscription,
) -> TranscriptRoute {
    match transcription.source_kind {
        SttSourceKind::Browser => route_browser(inner, transcription),
        SttSourceKind::Rover => {
            let valid = transcription
                .entity_id
                .as_deref()
                .is_some_and(|id| !id.trim().is_empty() && id == transcription.target_entity_id);
            if valid {
                TranscriptRoute::RoverBroadcast
            } else {
                TranscriptRoute::Drop("invalid rover transcription identity".into())
            }
        }
    }
}

fn route_browser(inner: &mut BridgeInner, transcription: &SpeechTranscription) -> TranscriptRoute {
    if transcription.entity_id.is_some() || transcription.target_entity_id.trim().is_empty() {
        return TranscriptRoute::Drop("invalid browser transcription identity".into());
    }
    let Ok(stream_id) = Uuid::parse_str(&transcription.stream_id) else {
        return TranscriptRoute::Drop("invalid browser transcription stream UUID".into());
    };
    match inner
        .streams
        .route_browser_result(stream_id, &transcription.target_entity_id)
    {
        Some(socket_id) => TranscriptRoute::Browser { socket_id },
        None => {
            inner.metrics.late_transcriptions = inner.metrics.late_transcriptions.saturating_add(1);
            TranscriptRoute::Drop("browser transcription has no matching owner".into())
        }
    }
}
