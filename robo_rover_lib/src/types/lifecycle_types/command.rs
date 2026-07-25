use super::{
    validation::validate_uuid, LifecycleDesiredState, LifecycleReasonCode, LifecycleTarget,
};
use serde::{Deserialize, Serialize};

pub const LIFECYCLE_PROTOCOL_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleCommandOrigin {
    #[default]
    User,
    Coordinator,
}

fn is_user_origin(origin: &LifecycleCommandOrigin) -> bool {
    *origin == LifecycleCommandOrigin::User
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleCommand {
    pub protocol_version: u8,
    pub request_id: String,
    pub manager_epoch: u64,
    pub target: LifecycleTarget,
    pub desired_state: LifecycleDesiredState,
    pub expected_revision: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    /// Coordinator commands bind asynchronous status to one power transition.
    #[serde(default, skip_serializing_if = "is_user_origin")]
    pub origin: LifecycleCommandOrigin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
        match (self.origin, self.transition_id.as_deref()) {
            (LifecycleCommandOrigin::Coordinator, Some(transition_id)) => {
                validate_uuid("transition_id", transition_id)?;
            }
            (LifecycleCommandOrigin::Coordinator, None) => {
                return Err("coordinator lifecycle command requires transition_id".into());
            }
            (LifecycleCommandOrigin::User, None) => {}
            (LifecycleCommandOrigin::User, Some(_)) => {
                return Err("user lifecycle command cannot carry transition_id".into());
            }
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
