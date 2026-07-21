use super::{validation::validate_id, LIFECYCLE_PROTOCOL_VERSION};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleRole {
    Orchestra,
    Rover,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleDesiredState {
    Running,
    Quiesced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleEffectiveState {
    Running,
    Cancelling,
    Quiescing,
    Quiesced,
    Resuming,
    Degraded,
    Failed,
    Superseded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleComponentState {
    Running,
    Cancelling,
    Quiescing,
    Quiesced,
    Resuming,
    Degraded,
    Failed,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleReasonCode {
    InvalidRequest,
    InvalidTarget,
    Unsupported,
    Conflict,
    Expired,
    StaleEpoch,
    DuplicateMismatch,
    InterruptedByLifecycle,
    Timeout,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LifecycleTarget {
    pub role: LifecycleRole,
    pub entity_id: String,
    pub node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleCapability {
    pub target: LifecycleTarget,
    pub supported: bool,
    pub always_on: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleComponentStatus {
    pub node_id: String,
    pub state: LifecycleComponentState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<LifecycleReasonCode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleStatus {
    pub protocol_version: u8,
    pub manager_epoch: u64,
    pub target: LifecycleTarget,
    pub revision: u64,
    pub desired_state: LifecycleDesiredState,
    pub effective_state: LifecycleEffectiveState,
    pub components: Vec<LifecycleComponentStatus>,
    pub updated_at_ms: u64,
}

impl LifecycleTarget {
    pub fn validate(&self) -> Result<(), String> {
        validate_id("entity_id", &self.entity_id)?;
        validate_id("node_id", &self.node_id)
    }
}

impl LifecycleStatus {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != LIFECYCLE_PROTOCOL_VERSION
            || self.manager_epoch == 0
            || self.updated_at_ms == 0
        {
            return Err("invalid lifecycle status version, epoch, or timestamp".into());
        }
        self.target.validate()?;
        if self.components.len() > 128 {
            return Err("too many lifecycle components".into());
        }
        for component in &self.components {
            validate_id("component node_id", &component.node_id)?;
        }
        Ok(())
    }
}
