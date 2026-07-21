use uuid::Uuid;

use super::{
    validation::{validate_external_detail, validate_timestamp, validate_wire_integer},
    PlaybackSource, PlaybackState, PlaybackStateKind, TtsAckState, TtsCommandAck, TtsCommandResult,
    TtsResultState, VoiceReasonCode, VoiceState, VoiceStatus,
};

impl TtsCommandAck {
    pub fn validate(&self) -> Result<(), String> {
        validate_common(
            &self.command_id,
            &self.target_entity_id,
            self.timestamp,
            &self.detail,
        )?;
        match (self.state, self.reason_code, self.detail.as_ref()) {
            (TtsAckState::Accepted, None, None) => Ok(()),
            (TtsAckState::Rejected, Some(reason), _) if is_valid_ack_reason(reason) => Ok(()),
            _ => Err("ack fields do not match its admission outcome".into()),
        }
    }
}

impl TtsCommandResult {
    pub fn validate(&self) -> Result<(), String> {
        validate_common(
            &self.command_id,
            &self.entity_id,
            self.timestamp,
            &self.detail,
        )?;
        match (self.state, self.reason_code, self.detail.as_ref()) {
            (TtsResultState::Completed, None, None) => Ok(()),
            (TtsResultState::Rejected, Some(reason), _)
                if is_valid_rejected_result_reason(reason) =>
            {
                Ok(())
            }
            (TtsResultState::Interrupted, Some(reason), _)
                if is_valid_interrupted_result_reason(reason) =>
            {
                Ok(())
            }
            (TtsResultState::Failed, Some(reason), _) if is_valid_failed_result_reason(reason) => {
                Ok(())
            }
            _ => Err("result fields do not match its terminal lifecycle state".into()),
        }
    }
}

impl VoiceStatus {
    pub fn validate(&self) -> Result<(), String> {
        validate_entity(&self.entity_id)?;
        validate_wire_integer(self.applied_revision, "applied TTS revision")?;
        validate_timestamp(self.timestamp)?;
        self.applied_config.validate()?;
        validate_external_detail(&self.detail)?;
        if let Some(id) = &self.active_command_id {
            validate_command_id(id)?;
        }
        match (
            self.state,
            self.active_command_id.as_ref(),
            self.reason_code,
            self.detail.as_ref(),
        ) {
            (VoiceState::Speaking, Some(_), None, None) => Ok(()),
            (VoiceState::Loading | VoiceState::Ready, None, None, None) => Ok(()),
            (VoiceState::Error, None, Some(reason), _) if is_valid_voice_error_reason(reason) => {
                Ok(())
            }
            (VoiceState::Unavailable, None, Some(reason), _)
                if is_valid_voice_unavailable_reason(reason) =>
            {
                Ok(())
            }
            _ => Err("voice status fields do not match its lifecycle state".into()),
        }
    }
}

impl PlaybackState {
    pub fn validate(&self) -> Result<(), String> {
        validate_entity(&self.entity_id)?;
        validate_command_id(&self.producer_instance_id)
            .map_err(|_| "playback producer_instance_id must be a UUID".to_string())?;
        validate_wire_integer(self.sequence_id, "playback sequence")?;
        validate_timestamp(self.timestamp)?;
        validate_external_detail(&self.detail)?;
        match (
            self.state,
            self.source,
            self.command_id.as_deref(),
            self.reason_code,
            self.detail.as_ref(),
        ) {
            (PlaybackStateKind::Idle, None, None, None, None) => Ok(()),
            (PlaybackStateKind::Active, Some(PlaybackSource::Tts), Some(id), None, None) => {
                validate_command_id(id)
            }
            (PlaybackStateKind::Active, Some(PlaybackSource::Walkie), None, None, None) => Ok(()),
            (
                PlaybackStateKind::Active,
                Some(PlaybackSource::Walkie),
                Some(id),
                Some(VoiceReasonCode::InterruptedByWalkie),
                _,
            ) => validate_command_id(id),
            (PlaybackStateKind::Unavailable, None, None, Some(reason), _)
                if is_valid_playback_unavailable_reason(reason) =>
            {
                Ok(())
            }
            _ => Err("playback state fields do not match its lifecycle state".into()),
        }
    }
}

fn is_valid_ack_reason(reason: VoiceReasonCode) -> bool {
    matches!(
        reason,
        VoiceReasonCode::InvalidCommand
            | VoiceReasonCode::VoiceNotReady
            | VoiceReasonCode::WalkieActive
            | VoiceReasonCode::InternalError
    )
}

fn is_valid_rejected_result_reason(reason: VoiceReasonCode) -> bool {
    matches!(
        reason,
        VoiceReasonCode::QueueFull
            | VoiceReasonCode::VoiceNotReady
            | VoiceReasonCode::WalkieActive
            | VoiceReasonCode::PlaybackUnavailable
            | VoiceReasonCode::InvalidCommand
    )
}

fn is_valid_interrupted_result_reason(reason: VoiceReasonCode) -> bool {
    matches!(
        reason,
        VoiceReasonCode::InterruptedByWalkie
            | VoiceReasonCode::InterruptedByLifecycle
            | VoiceReasonCode::Cancelled
    )
}

fn is_valid_failed_result_reason(reason: VoiceReasonCode) -> bool {
    matches!(
        reason,
        VoiceReasonCode::SynthesisFailed
            | VoiceReasonCode::PlaybackFailed
            | VoiceReasonCode::PlaybackUnavailable
            | VoiceReasonCode::InternalError
    )
}

fn is_valid_voice_error_reason(reason: VoiceReasonCode) -> bool {
    matches!(
        reason,
        VoiceReasonCode::SynthesisFailed
            | VoiceReasonCode::PlaybackFailed
            | VoiceReasonCode::InternalError
    )
}

fn is_valid_voice_unavailable_reason(reason: VoiceReasonCode) -> bool {
    matches!(
        reason,
        VoiceReasonCode::VoiceNotReady
            | VoiceReasonCode::PlaybackUnavailable
            | VoiceReasonCode::PlaybackFailed
            | VoiceReasonCode::InternalError
    )
}

fn is_valid_playback_unavailable_reason(reason: VoiceReasonCode) -> bool {
    matches!(
        reason,
        VoiceReasonCode::PlaybackUnavailable
            | VoiceReasonCode::PlaybackFailed
            | VoiceReasonCode::InternalError
    )
}

fn validate_common(
    id: &str,
    entity: &str,
    timestamp: u64,
    detail: &Option<String>,
) -> Result<(), String> {
    validate_command_id(id)?;
    validate_entity(entity)?;
    validate_timestamp(timestamp)?;
    validate_external_detail(detail)
}

fn validate_command_id(id: &str) -> Result<(), String> {
    Uuid::parse_str(id)
        .map(|_| ())
        .map_err(|_| "TTS command ID must be a UUID".into())
}

fn validate_entity(entity: &str) -> Result<(), String> {
    if entity.trim().is_empty() || entity.len() > 128 {
        return Err("voice entity ID must contain 1..=128 bytes".into());
    }
    Ok(())
}
