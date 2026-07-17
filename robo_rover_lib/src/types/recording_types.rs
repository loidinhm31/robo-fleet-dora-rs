use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const RECORDING_PROTOCOL_VERSION: u8 = 1;
const MAX_ID_LEN: usize = 128;
const MAX_RELATIVE_DIRECTORY_LEN: usize = 240;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingSessionState {
    Starting,
    Recording,
    Stopping,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingReasonCode {
    InvalidRequest,
    Unauthenticated,
    InvalidEntity,
    InvalidDirectory,
    AlreadyRecording,
    StartupTimeout,
    StorageUnavailable,
    EncoderFailed,
    ResourceLimit,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RecordingSessionAction {
    Start {
        entity_id: String,
        relative_directory: String,
    },
    Stop {
        recording_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingSessionCommand {
    pub protocol_version: u8,
    pub request_id: String,
    #[serde(flatten)]
    pub action: RecordingSessionAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingSessionCommandResult {
    pub protocol_version: u8,
    pub request_id: String,
    pub accepted: bool,
    pub recording_id: Option<String>,
    pub reason_code: Option<RecordingReasonCode>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingSessionStatus {
    pub protocol_version: u8,
    pub request_id: String,
    pub recording_id: String,
    pub entity_id: String,
    pub state: RecordingSessionState,
    pub started_at_ms: Option<u64>,
    pub duration_ms: u64,
    pub bytes_written: u64,
    pub reason_code: Option<RecordingReasonCode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingClipQuery {
    pub protocol_version: u8,
    pub request_id: String,
    pub entity_id: Option<String>,
    pub relative_directory: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingClip {
    pub recording_id: String,
    pub entity_id: String,
    pub relative_path: String,
    pub started_at_ms: u64,
    pub duration_ms: u64,
    pub bytes_written: u64,
    pub video_codec: RecordingVideoCodec,
    pub audio_codec: RecordingAudioCodec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingClipListResult {
    pub protocol_version: u8,
    pub request_id: String,
    pub clips: Vec<RecordingClip>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingPlaybackTicketRequest {
    pub protocol_version: u8,
    pub request_id: String,
    pub recording_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingPlaybackTicketResult {
    pub protocol_version: u8,
    pub request_id: String,
    pub recording_id: String,
    pub ticket: String,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingVideoCodec {
    H264,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingAudioCodec {
    Aac,
}

/// Exact rover-targeted change. `None` means this resource is unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetedMediaControl {
    pub protocol_version: u8,
    pub entity_id: String,
    pub camera_enabled: Option<bool>,
    pub jpeg_enabled: Option<bool>,
    pub microphone_enabled: Option<bool>,
}

impl RecordingSessionCommand {
    pub fn validate(&self) -> Result<(), String> {
        validate_version(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        match &self.action {
            RecordingSessionAction::Start {
                entity_id,
                relative_directory,
            } => {
                validate_id("entity_id", entity_id)?;
                validate_relative_directory(relative_directory)
            }
            RecordingSessionAction::Stop { recording_id } => {
                validate_uuid("recording_id", recording_id)
            }
        }
    }
}

impl TargetedMediaControl {
    pub fn validate(&self) -> Result<(), String> {
        validate_version(self.protocol_version)?;
        validate_id("entity_id", &self.entity_id)?;
        if self.camera_enabled.is_none()
            && self.jpeg_enabled.is_none()
            && self.microphone_enabled.is_none()
        {
            return Err("targeted media control has no resource change".into());
        }
        Ok(())
    }
}

impl RecordingSessionCommandResult {
    pub fn validate(&self) -> Result<(), String> {
        validate_version(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        if let Some(recording_id) = &self.recording_id {
            validate_uuid("recording_id", recording_id)?;
        }
        if self.accepted != self.reason_code.is_none() {
            return Err("accepted result has inconsistent reason_code".into());
        }
        if self.accepted != self.recording_id.is_some() {
            return Err("accepted result has inconsistent recording_id".into());
        }
        if self
            .detail
            .as_ref()
            .is_some_and(|detail| detail.len() > 256)
        {
            return Err("recording result detail exceeds 256 characters".into());
        }
        Ok(())
    }
}

impl RecordingSessionStatus {
    pub fn validate(&self) -> Result<(), String> {
        validate_version(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        validate_uuid("recording_id", &self.recording_id)?;
        validate_id("entity_id", &self.entity_id)?;
        if matches!(
            self.state,
            RecordingSessionState::Recording
                | RecordingSessionState::Stopping
                | RecordingSessionState::Completed
        ) && self.started_at_ms.is_none()
        {
            return Err("active or completed status requires started_at_ms".into());
        }
        if self.state == RecordingSessionState::Failed {
            if self.reason_code.is_none() {
                return Err("failed status requires reason_code".into());
            }
        } else if self.reason_code.is_some() {
            return Err("nonfailed status cannot include reason_code".into());
        }
        Ok(())
    }
}

impl RecordingClipQuery {
    pub fn validate(&self) -> Result<(), String> {
        validate_version(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        if let Some(entity_id) = &self.entity_id {
            validate_id("entity_id", entity_id)?;
        }
        if let Some(directory) = &self.relative_directory {
            validate_relative_directory(directory)?;
        }
        Ok(())
    }
}

impl RecordingClip {
    pub fn validate(&self) -> Result<(), String> {
        validate_uuid("recording_id", &self.recording_id)?;
        validate_id("entity_id", &self.entity_id)?;
        validate_relative_directory(&self.relative_path)
    }
}

impl RecordingClipListResult {
    pub fn validate(&self) -> Result<(), String> {
        validate_version(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        self.clips.iter().try_for_each(RecordingClip::validate)
    }
}

impl RecordingPlaybackTicketRequest {
    pub fn validate(&self) -> Result<(), String> {
        validate_version(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        validate_uuid("recording_id", &self.recording_id)
    }
}

impl RecordingPlaybackTicketResult {
    pub fn validate(&self) -> Result<(), String> {
        validate_version(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        validate_uuid("recording_id", &self.recording_id)?;
        validate_ticket(&self.ticket)
    }
}

pub fn validate_version(version: u8) -> Result<(), String> {
    (version == RECORDING_PROTOCOL_VERSION)
        .then_some(())
        .ok_or_else(|| format!("unsupported recording protocol version: {version}"))
}

pub fn validate_id(field: &str, value: &str) -> Result<(), String> {
    (!value.is_empty()
        && value.len() <= MAX_ID_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')))
    .then_some(())
    .ok_or_else(|| format!("invalid {field}"))
}

pub fn validate_uuid(field: &str, value: &str) -> Result<(), String> {
    Uuid::parse_str(value)
        .map(|_| ())
        .map_err(|_| format!("invalid {field}"))
}

pub fn validate_relative_directory(value: &str) -> Result<(), String> {
    (!value.is_empty()
        && value.len() <= MAX_RELATIVE_DIRECTORY_LEN
        && !value.contains('\\')
        && !value.contains('\0')
        && !value.starts_with('/')
        && value.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        }))
    .then_some(())
    .ok_or_else(|| "invalid relative_directory".into())
}

fn validate_ticket(value: &str) -> Result<(), String> {
    (!value.is_empty()
        && value.len() <= 1024
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')))
    .then_some(())
    .ok_or_else(|| "invalid playback ticket".into())
}
