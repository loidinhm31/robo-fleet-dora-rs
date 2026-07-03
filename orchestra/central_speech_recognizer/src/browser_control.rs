use crate::audio_input::SourceIdentity;
use crate::session::{FrameOutcome, SessionManager};
use dora_node_api::arrow::array::{Array, BinaryArray};
use eyre::{eyre, Result};
use robo_rover_lib::SttSourceKind;
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
#[serde(tag = "command", rename_all = "lowercase")]
enum BrowserControl {
    Start {
        stream_id: Uuid,
        sample_rate: u32,
        channels: u16,
        target_entity_id: String,
    },
    Stop {
        stream_id: Uuid,
    },
}

pub(crate) fn handle_control(
    data: &dyn Array,
    sessions: &mut SessionManager,
) -> Result<FrameOutcome> {
    let array = data
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| eyre!("browser control must be a BinaryArray"))?;
    if array.len() != 1 || array.is_null(0) {
        return Err(eyre!("browser control must contain one non-null command"));
    }
    match serde_json::from_slice(array.value(0))? {
        BrowserControl::Start {
            stream_id,
            sample_rate,
            channels,
            target_entity_id,
        } => {
            if channels != 1 {
                return Err(eyre!("browser speech stream must be mono"));
            }
            sessions.start_browser(
                SourceIdentity {
                    stream_id,
                    source_kind: SttSourceKind::Browser,
                    entity_id: None,
                    target_entity_id,
                },
                sample_rate,
            )
        }
        BrowserControl::Stop { stream_id } => Ok(FrameOutcome {
            jobs: sessions.stop_browser(stream_id)?,
            sequence_reset: false,
        }),
    }
}

#[cfg(test)]
mod tests;
