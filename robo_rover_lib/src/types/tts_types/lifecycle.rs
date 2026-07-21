use super::TtsRuntimeConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceReasonCode {
    InvalidCommand,
    InvalidConfig,
    StaleRevision,
    QueueFull,
    VoiceNotReady,
    WalkieActive,
    InterruptedByWalkie,
    InterruptedByLifecycle,
    Cancelled,
    SynthesisFailed,
    PlaybackFailed,
    PlaybackUnavailable,
    InternalError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsAckState {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsCommandAck {
    pub command_id: String,
    pub target_entity_id: String,
    pub state: TtsAckState,
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<VoiceReasonCode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsResultState {
    Completed,
    Rejected,
    Interrupted,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsCommandResult {
    pub command_id: String,
    pub entity_id: String,
    pub state: TtsResultState,
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<VoiceReasonCode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceState {
    Loading,
    Ready,
    Speaking,
    Error,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceStatus {
    pub entity_id: String,
    pub state: VoiceState,
    pub applied_revision: u64,
    pub applied_config: TtsRuntimeConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_command_id: Option<String>,
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<VoiceReasonCode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackSource {
    Tts,
    Walkie,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackStateKind {
    Idle,
    Active,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaybackState {
    pub entity_id: String,
    pub producer_instance_id: String,
    pub sequence_id: u64,
    pub state: PlaybackStateKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<PlaybackSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_id: Option<String>,
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<VoiceReasonCode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}
