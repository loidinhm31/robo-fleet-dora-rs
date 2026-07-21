use super::{
    validation::validate_uuid, LifecycleDesiredState, LifecycleReasonCode, LifecycleTarget,
};
use serde::{Deserialize, Serialize};

pub const LIFECYCLE_PROTOCOL_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleCommand {
    pub protocol_version: u8,
    pub request_id: String,
    pub manager_epoch: u64,
    pub target: LifecycleTarget,
    pub desired_state: LifecycleDesiredState,
    pub expected_revision: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleCommandResult {
    pub protocol_version: u8,
    pub request_id: String,
    pub accepted: bool,
    pub manager_epoch: u64,
    pub revision: u64,
    pub reason_code: Option<LifecycleReasonCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl LifecycleCommand {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != LIFECYCLE_PROTOCOL_VERSION {
            return Err("unsupported lifecycle protocol version".into());
        }
        validate_uuid("request_id", &self.request_id)?;
        self.target.validate()?;
        if self.manager_epoch == 0
            || self.issued_at_ms == 0
            || self.expires_at_ms <= self.issued_at_ms
        {
            return Err("invalid lifecycle epoch or command timestamps".into());
        }
        if self.expires_at_ms.saturating_sub(self.issued_at_ms) > 60_000 {
            return Err("lifecycle command ttl exceeds 60 seconds".into());
        }
        Ok(())
    }
}

impl LifecycleCommandResult {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != LIFECYCLE_PROTOCOL_VERSION {
            return Err("unsupported lifecycle protocol version".into());
        }
        validate_uuid("request_id", &self.request_id)?;
        if self.manager_epoch == 0 || self.accepted == self.reason_code.is_some() {
            return Err("inconsistent lifecycle command result".into());
        }
        if self.detail.as_ref().is_some_and(|value| value.len() > 256) {
            return Err("lifecycle command detail exceeds 256 characters".into());
        }
        Ok(())
    }
}
