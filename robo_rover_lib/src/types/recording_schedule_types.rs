use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

pub const RECORDING_SCHEDULE_PROTOCOL_VERSION: u8 = 1;

/// Public schedule settings. Audit and lifecycle fields are scheduler-owned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordingScheduleDefinition {
    pub entity_id: String,
    pub title: String,
    pub enabled: bool,
    pub recurrence: RecordingScheduleRecurrence,
    pub duration_ms: i64,
    pub relative_directory_template: String,
}

/// Preserves wall-clock intent; daily and weekly dates are inclusive recurrence anchors.
/// Occurrences hold the resolved Unix-millisecond times.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordingLocalStart {
    pub date: String,
    pub time: String,
    pub timezone: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecordingScheduleRecurrence {
    OneTime {
        local_start: RecordingLocalStart,
    },
    Daily {
        local_start: RecordingLocalStart,
    },
    Weekly {
        local_start: RecordingLocalStart,
        weekdays: Vec<IsoWeekday>,
    },
}

impl RecordingScheduleRecurrence {
    pub fn local_start(&self) -> &RecordingLocalStart {
        match self {
            Self::OneTime { local_start }
            | Self::Daily { local_start }
            | Self::Weekly { local_start, .. } => local_start,
        }
    }
}

/// ISO-8601 weekdays, where Monday is day one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsoWeekday {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingSchedule {
    pub schedule_id: String,
    pub revision: u64,
    #[serde(flatten)]
    pub definition: RecordingScheduleDefinition,
    pub created_at_ms: i64,
    pub created_by: String,
    pub updated_at_ms: i64,
    pub updated_by: String,
}

/// Browser mutation input. It deliberately has no actor, occurrence, or recording fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecordingScheduleCommand {
    pub protocol_version: u8,
    pub request_id: String,
    #[serde(flatten)]
    pub action: RecordingScheduleAction,
}

#[derive(Deserialize)]
struct RecordingScheduleCommandWire {
    protocol_version: u8,
    request_id: String,
    #[serde(flatten)]
    action: RecordingScheduleAction,
}

impl<'de> Deserialize<'de> for RecordingScheduleCommand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("schedule command must be an object"))?;
        let action = object
            .get("action")
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("schedule command action is required"))?;
        let allowed = match action {
            "create" => ["protocol_version", "request_id", "action", "schedule"].as_slice(),
            "update" => [
                "protocol_version",
                "request_id",
                "action",
                "schedule_id",
                "expected_revision",
                "schedule",
            ]
            .as_slice(),
            "set_enabled" => [
                "protocol_version",
                "request_id",
                "action",
                "schedule_id",
                "expected_revision",
                "enabled",
            ]
            .as_slice(),
            "delete" => [
                "protocol_version",
                "request_id",
                "action",
                "schedule_id",
                "expected_revision",
            ]
            .as_slice(),
            _ => return Err(serde::de::Error::custom("unknown schedule command action")),
        };
        if object.keys().any(|key| !allowed.contains(&key.as_str())) {
            return Err(serde::de::Error::custom(
                "schedule command has unknown fields",
            ));
        }
        let wire: RecordingScheduleCommandWire =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            protocol_version: wire.protocol_version,
            request_id: wire.request_id,
            action: wire.action,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RecordingScheduleAction {
    Create {
        schedule: RecordingScheduleDefinition,
    },
    Update {
        schedule_id: String,
        expected_revision: u64,
        schedule: RecordingScheduleDefinition,
    },
    SetEnabled {
        schedule_id: String,
        expected_revision: u64,
        enabled: bool,
    },
    Delete {
        schedule_id: String,
        expected_revision: u64,
    },
}

/// Dora-only wrapper constructed after the web bridge verifies the login session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthenticatedRecordingScheduleCommand {
    pub command: RecordingScheduleCommand,
    pub audit_actor: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordingScheduleQuery {
    pub protocol_version: u8,
    pub request_id: String,
    pub entity_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingScheduleReasonCode {
    InvalidRequest,
    Unauthenticated,
    InvalidSchedule,
    InvalidRecurrence,
    InvalidTimezone,
    InvalidDirectory,
    Conflict,
    NotFound,
    Unavailable,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingScheduleCommandResult {
    pub protocol_version: u8,
    pub request_id: String,
    pub accepted: bool,
    pub schedule: Option<RecordingSchedule>,
    /// Present for a CAS conflict so clients can refresh from the authoritative revision.
    pub current_schedule: Option<RecordingSchedule>,
    pub reason_code: Option<RecordingScheduleReasonCode>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingScheduleSnapshot {
    pub protocol_version: u8,
    pub request_id: String,
    pub entity_id: String,
    pub schedules: Vec<RecordingSchedule>,
}
