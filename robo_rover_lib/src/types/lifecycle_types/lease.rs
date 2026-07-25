use super::{validation::validate_id, LifecycleTarget, LIFECYCLE_PROTOCOL_VERSION};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleWakeLeaseAction {
    Acquire,
    Release,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleWakeLease {
    pub protocol_version: u8,
    pub lease_id: String,
    pub target: LifecycleTarget,
    pub expires_at_ms: u64,
    /// Monotonic per-lease generation fences delayed acquire/release messages.
    #[serde(
        default = "default_generation",
        skip_serializing_if = "is_default_generation"
    )]
    pub generation: u64,
    pub action: LifecycleWakeLeaseAction,
}

impl LifecycleWakeLease {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != LIFECYCLE_PROTOCOL_VERSION
            || self.expires_at_ms == 0
            || self.generation == 0
        {
            return Err("invalid lifecycle wake lease version or expiry".into());
        }
        validate_id("wake lease id", &self.lease_id)?;
        self.target.validate()
    }
}

fn default_generation() -> u64 {
    1
}

fn is_default_generation(generation: &u64) -> bool {
    *generation == default_generation()
}
