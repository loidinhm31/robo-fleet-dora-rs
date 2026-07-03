use dora_node_api::{
    arrow::array::{BinaryArray, Float32Array},
    dora_core::config::DataId,
    DoraNode, MetadataParameters, Parameter,
};
use eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "command", rename_all = "lowercase", deny_unknown_fields)]
pub enum VoiceCommandControl {
    Start {
        stream_id: Uuid,
        sample_rate: u32,
        channels: u16,
    },
    Stop {
        stream_id: Uuid,
    },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VoiceCommandAudioFrame {
    pub stream_id: Uuid,
    pub frame_id: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_count: u32,
    pub audio_data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "command", rename_all = "lowercase")]
pub enum BrowserControlOutput {
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

#[derive(Debug, Clone, PartialEq)]
pub enum SttDoraMessage {
    Audio {
        frame: VoiceCommandAudioFrame,
        target_entity_id: String,
    },
    Control(BrowserControlOutput),
    StatusRequest,
}

impl SttDoraMessage {
    pub fn stream_id(&self) -> Option<Uuid> {
        match self {
            Self::Audio { frame, .. } => Some(frame.stream_id),
            Self::Control(BrowserControlOutput::Start { stream_id, .. })
            | Self::Control(BrowserControlOutput::Stop { stream_id }) => Some(*stream_id),
            Self::StatusRequest => None,
        }
    }

    pub fn is_start(&self) -> bool {
        matches!(self, Self::Control(BrowserControlOutput::Start { .. }))
    }
}

#[derive(Clone)]
pub struct SttOutputIds {
    pub audio: DataId,
    pub control: DataId,
    pub status_request: DataId,
}

pub fn send_dora_message(
    node: &mut DoraNode,
    outputs: &SttOutputIds,
    message: &SttDoraMessage,
) -> Result<()> {
    match message {
        SttDoraMessage::Audio {
            frame,
            target_entity_id,
        } => {
            let parameters = audio_parameters(frame, target_entity_id.clone())?;
            node.send_output(
                outputs.audio.clone(),
                parameters,
                Float32Array::from(frame.audio_data.clone()),
            )?;
        }
        SttDoraMessage::Control(control) => {
            let bytes = serde_json::to_vec(control)?;
            node.send_output(
                outputs.control.clone(),
                Default::default(),
                BinaryArray::from_vec(vec![bytes.as_slice()]),
            )?;
        }
        SttDoraMessage::StatusRequest => {
            let payload = b"{}";
            node.send_output(
                outputs.status_request.clone(),
                Default::default(),
                BinaryArray::from_vec(vec![payload.as_slice()]),
            )?;
        }
    }
    Ok(())
}

fn audio_parameters(
    frame: &VoiceCommandAudioFrame,
    target_entity_id: String,
) -> Result<MetadataParameters> {
    let integer = |value: u64, field: &str| {
        i64::try_from(value)
            .map(Parameter::Integer)
            .map_err(|_| eyre!("{field} exceeds i64"))
    };
    Ok([
        ("source_kind".into(), Parameter::String("browser".into())),
        (
            "stream_id".into(),
            Parameter::String(frame.stream_id.to_string()),
        ),
        ("frame_id".into(), integer(frame.frame_id, "frame_id")?),
        (
            "capture_timestamp_ms".into(),
            integer(now_ms(), "capture_timestamp_ms")?,
        ),
        (
            "sample_rate".into(),
            Parameter::Integer(i64::from(frame.sample_rate)),
        ),
        (
            "channels".into(),
            Parameter::Integer(i64::from(frame.channels)),
        ),
        (
            "sample_count".into(),
            Parameter::Integer(i64::from(frame.sample_count)),
        ),
        ("format".into(), Parameter::String("f32le".into())),
        (
            "target_entity_id".into(),
            Parameter::String(target_entity_id),
        ),
    ]
    .into_iter()
    .collect())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
