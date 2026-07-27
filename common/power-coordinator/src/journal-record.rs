use robo_rover_lib::{PowerEvent, PowerJournalAcknowledgement, PowerStatus};
use serde::{Deserialize, Serialize};

pub const JOURNAL_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JournalIntent {
    BootAwake,
    Command,
    CommandApplied,
    Transition,
    TransitionApplied,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JournalRecord {
    pub format_version: u8,
    pub sequence: u64,
    pub intent: JournalIntent,
    pub event: PowerEvent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<PowerStatus>,
}

pub type JournalAcknowledgement = PowerJournalAcknowledgement;

impl JournalRecord {
    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != JOURNAL_VERSION || self.sequence == 0 {
            return Err("invalid journal record header".into());
        }
        self.event.validate()?;
        if let Some(status) = &self.status {
            status.validate()?;
            if status.role != self.event.role || status.entity_id != self.event.entity_id {
                return Err("journal status target differs from event".into());
            }
            sanitize_detail(status.detail.as_deref())?;
        }
        sanitize_detail(self.event.detail.as_deref())
    }
}

fn sanitize_detail(detail: Option<&str>) -> Result<(), String> {
    let Some(detail) = detail else {
        return Ok(());
    };
    let lower = detail.to_ascii_lowercase();
    if detail.starts_with('/')
        || detail.contains("\\\\")
        || ["token", "secret", "password", "authorization", "bearer"]
            .iter()
            .any(|item| lower.contains(item))
    {
        return Err("journal detail contains sensitive or path data".into());
    }
    Ok(())
}
