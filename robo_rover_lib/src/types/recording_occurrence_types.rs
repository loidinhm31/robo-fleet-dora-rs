use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{RecordingScheduleReasonCode, RecordingSessionState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingOccurrenceState {
    Planned,
    Due,
    StartPending,
    Active,
    StopPending,
    Completed,
    Suppressed,
    Missed,
    Failed,
    Cancelled,
}

impl RecordingOccurrenceState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Suppressed | Self::Missed | Self::Failed | Self::Cancelled
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DstResolution {
    Exact,
    GapShifted,
    FoldEarlier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingAttemptState {
    Started,
    Partial,
    Failed,
    Recovered,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingClipAttempt {
    pub recording_id: String,
    pub state: RecordingAttemptState,
    pub started_at_ms: i64,
    pub ended_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingOccurrence {
    pub occurrence_id: String,
    pub schedule_id: String,
    pub schedule_revision: u64,
    pub entity_id: String,
    pub planned_start_ms: i64,
    pub planned_end_ms: i64,
    pub dst_resolution: DstResolution,
    pub state: RecordingOccurrenceState,
    pub retry_count: u32,
    pub next_retry_at_ms: Option<i64>,
    pub group_id: Option<String>,
    pub start_request_id: String,
    pub attempts: Vec<RecordingClipAttempt>,
    pub last_error: Option<RecordingOccurrenceError>,
    pub suppressed_by_manual: bool,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    /// Set once at a terminal transition; retention derives from this audit timestamp.
    pub terminal_at_ms: Option<i64>,
    /// Scheduler persistence sets this only after terminal transition for Mongo TTL.
    pub expires_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingOccurrenceError {
    pub reason_code: RecordingScheduleReasonCode,
    pub detail: String,
}

/// Intent sent only to the web-bridge coordinator, never to a rover or FFmpeg node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledRecordingIntent {
    /// Deterministic per-action identity for stale-feedback rejection.
    pub intent_id: String,
    pub occurrence_id: String,
    pub group_id: String,
    pub generation: u64,
    pub entity_id: String,
    pub start_request_id: String,
    pub planned_start_ms: i64,
    pub planned_end_ms: i64,
    pub relative_directory: String,
    pub action: ScheduledRecordingIntentAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledRecordingIntentAction {
    Acquire,
    Release,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingCoordinatorFeedback {
    /// Echoes the scheduler intent ID; browser payloads never carry this field.
    pub intent_id: String,
    pub occurrence_id: String,
    pub generation: u64,
    pub accepted: bool,
    pub applied: bool,
    pub retryable: bool,
    pub group_id: Option<String>,
    pub recording_id: Option<String>,
    pub recorder_state: Option<RecordingSessionState>,
    pub reason_code: Option<RecordingScheduleReasonCode>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingReconciliationRequest {
    pub request_id: String,
    pub entity_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingReconciliationSnapshot {
    pub request_id: String,
    pub sessions: Vec<RecordingReconciliationSession>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingReconciliationSession {
    pub entity_id: String,
    pub start_request_id: String,
    pub recording_id: String,
    pub state: RecordingSessionState,
}

pub fn occurrence_id(
    schedule_id: &str,
    revision: u64,
    planned_start_ms: i64,
) -> Result<String, String> {
    let namespace = Uuid::parse_str(schedule_id).map_err(|_| "invalid schedule_id")?;
    Ok(Uuid::new_v5(
        &namespace,
        format!("{revision}:{planned_start_ms}").as_bytes(),
    )
    .to_string())
}

pub fn scheduled_start_request_id(occurrence_id: &str) -> Result<String, String> {
    let occurrence = Uuid::parse_str(occurrence_id).map_err(|_| "invalid occurrence_id")?;
    Ok(Uuid::new_v5(&occurrence, b"recording-start").to_string())
}

pub fn scheduled_intent_id(
    occurrence_id: &str,
    generation: u64,
    action: ScheduledRecordingIntentAction,
) -> Result<String, String> {
    let occurrence = Uuid::parse_str(occurrence_id).map_err(|_| "invalid occurrence_id")?;
    let action = match action {
        ScheduledRecordingIntentAction::Acquire => "acquire",
        ScheduledRecordingIntentAction::Release => "release",
    };
    Ok(Uuid::new_v5(&occurrence, format!("{generation}:{action}").as_bytes()).to_string())
}

/// Per-rover overlap groups are stable across scheduler restarts for one union-window start.
pub fn scheduled_group_id(entity_id: &str, group_start_ms: i64) -> Result<String, String> {
    if entity_id.is_empty() || group_start_ms < 0 {
        return Err("invalid group identity".into());
    }
    Ok(Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!("robo-fleet/recording-group/{entity_id}/{group_start_ms}").as_bytes(),
    )
    .to_string())
}
