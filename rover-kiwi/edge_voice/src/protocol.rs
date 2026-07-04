use std::time::{SystemTime, UNIX_EPOCH};

use dora_node_api::{arrow::array::BinaryArray, MetadataParameters, Parameter};
use eyre::{eyre, Result};
use robo_rover_lib::{
    sanitize_external_detail, PlaybackSource, PlaybackState, PlaybackStateKind, TtsCommand,
    TtsCommandResult, TtsConfigCommand, TtsPriority, TtsResultState, TtsRuntimeConfig,
    VoiceReasonCode, VoiceState, VoiceStatus,
};
use serde::Deserialize;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct IncomingTtsCommand {
    #[serde(default)]
    command_id: String,
    text: String,
    #[serde(default)]
    timestamp: u64,
    #[serde(default)]
    priority: TtsPriority,
}

pub fn parse_tts_command(bytes: &[u8]) -> Result<TtsCommand> {
    let incoming: IncomingTtsCommand = serde_json::from_slice(bytes)?;
    let command = TtsCommand {
        command_id: if incoming.command_id.trim().is_empty() {
            Uuid::new_v4().to_string()
        } else {
            incoming.command_id
        },
        text: incoming.text,
        timestamp: if incoming.timestamp == 0 {
            current_time_ms()
        } else {
            incoming.timestamp
        },
        priority: incoming.priority,
    };
    command.validate().map_err(eyre::Report::msg)?;
    Ok(command)
}

pub fn parse_config_command(bytes: &[u8]) -> Result<TtsConfigCommand> {
    let command = serde_json::from_slice::<TtsConfigCommand>(bytes)?;
    command.validate().map_err(eyre::Report::msg)?;
    Ok(command)
}

pub fn parse_playback_state(bytes: &[u8]) -> Result<PlaybackState> {
    let state = serde_json::from_slice::<PlaybackState>(bytes)?;
    state.validate().map_err(eyre::Report::msg)?;
    Ok(state)
}

pub fn to_binary<T: serde::Serialize>(value: &T) -> Result<BinaryArray> {
    let serialized = serde_json::to_vec(value)?;
    Ok(BinaryArray::from_vec(vec![serialized.as_slice()]))
}

pub fn voice_status(
    entity_id: &str,
    state: VoiceState,
    applied_revision: u64,
    applied_config: TtsRuntimeConfig,
    active_command_id: Option<String>,
    reason_code: Option<VoiceReasonCode>,
    detail: Option<String>,
) -> VoiceStatus {
    VoiceStatus {
        entity_id: entity_id.to_string(),
        state,
        applied_revision,
        applied_config,
        active_command_id,
        timestamp: current_time_ms(),
        reason_code,
        detail: detail.and_then(|value| sanitize_external_detail(&value)),
    }
}

pub fn command_result(
    entity_id: &str,
    command_id: &str,
    state: TtsResultState,
    reason_code: Option<VoiceReasonCode>,
    detail: Option<String>,
) -> TtsCommandResult {
    TtsCommandResult {
        command_id: command_id.to_string(),
        entity_id: entity_id.to_string(),
        state,
        timestamp: current_time_ms(),
        reason_code,
        detail: detail.and_then(|value| sanitize_external_detail(&value)),
    }
}

pub fn audio_metadata(
    command_id: &str,
    frame_id: u64,
    timestamp_ms: u64,
    sample_count: usize,
    priority: TtsPriority,
) -> Result<MetadataParameters> {
    let mut metadata = MetadataParameters::default();
    metadata.insert(
        "source_kind".to_string(),
        Parameter::String("tts".to_string()),
    );
    metadata.insert(
        "command_id".to_string(),
        Parameter::String(command_id.to_string()),
    );
    metadata.insert(
        "stream_id".to_string(),
        Parameter::String(command_id.to_string()),
    );
    metadata.insert(
        "frame_id".to_string(),
        Parameter::Integer(i64::try_from(frame_id)?),
    );
    metadata.insert(
        "capture_timestamp_ms".to_string(),
        Parameter::Integer(i64::try_from(timestamp_ms)?),
    );
    metadata.insert("sample_rate".to_string(), Parameter::Integer(44_100));
    metadata.insert("channels".to_string(), Parameter::Integer(1));
    metadata.insert("format".to_string(), Parameter::String("f32le".to_string()));
    metadata.insert(
        "sample_count".to_string(),
        Parameter::Integer(i64::try_from(sample_count)?),
    );
    metadata.insert(
        "priority".to_string(),
        Parameter::String(priority_name(priority).to_string()),
    );
    Ok(metadata)
}

pub fn walkie_is_active(state: &PlaybackState) -> bool {
    state.state == PlaybackStateKind::Active && state.source == Some(PlaybackSource::Walkie)
}

pub fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

pub fn sanitized_error(error: impl std::fmt::Display) -> String {
    sanitize_external_detail(&error.to_string()).unwrap_or_else(|| "internal error".to_string())
}

pub fn validate_voice_status(status: &VoiceStatus) -> Result<()> {
    status.validate().map_err(|error| eyre!(error))
}

pub fn validate_result(result: &TtsCommandResult) -> Result<()> {
    result.validate().map_err(|error| eyre!(error))
}

fn priority_name(priority: TtsPriority) -> &'static str {
    match priority {
        TtsPriority::Low => "low",
        TtsPriority::Normal => "normal",
        TtsPriority::High => "high",
        TtsPriority::Emergency => "emergency",
    }
}
