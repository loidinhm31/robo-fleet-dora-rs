use super::{
    validation::*, PowerAuthority, PowerDemand, PowerPolicy, PowerReasonCode, PowerReservation,
    POWER_PROTOCOL_VERSION,
};
use crate::types::lifecycle_types::LifecycleRole;
use serde::{Deserialize, Serialize};

const MAX_COMMAND_TTL_MS: u64 = 60_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum PowerCommandAction {
    SetPolicy { policy: PowerPolicy },
    RegisterDemand { demand: PowerDemand },
    ReleaseDemand { demand_id: String },
    RegisterReservation { reservation: PowerReservation },
    ReleaseReservation { reservation_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerCommand {
    pub protocol_version: u8,
    pub command_id: String,
    pub role: LifecycleRole,
    pub entity_id: String,
    pub authority: PowerAuthority,
    pub action: PowerCommandAction,
    pub issued_at_ms: u64,
    pub not_before_ms: u64,
    pub expires_at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerCommandResult {
    pub protocol_version: u8,
    pub command_id: String,
    pub accepted: bool,
    pub authority: PowerAuthority,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<PowerReasonCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl PowerCommand {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != POWER_PROTOCOL_VERSION {
            return Err("unsupported power protocol version".into());
        }
        validate_uuid("command_id", &self.command_id)?;
        validate_id("entity_id", &self.entity_id)?;
        self.authority.validate()?;
        validate_detail(self.detail.as_ref())?;
        if self.issued_at_ms == 0
            || self.issued_at_ms > self.not_before_ms
            || self.not_before_ms >= self.expires_at_ms
            || self.expires_at_ms - self.issued_at_ms > MAX_COMMAND_TTL_MS
        {
            return Err("invalid power command timestamps".into());
        }
        match &self.action {
            PowerCommandAction::SetPolicy { .. } => Ok(()),
            PowerCommandAction::RegisterDemand { demand } => {
                demand.validates_for(self.role, &self.entity_id)
            }
            PowerCommandAction::ReleaseDemand { demand_id } => {
                validate_uuid("demand_id", demand_id)
            }
            PowerCommandAction::RegisterReservation { reservation } => {
                reservation.validates_for(self.role, &self.entity_id)
            }
            PowerCommandAction::ReleaseReservation { reservation_id } => {
                validate_uuid("reservation_id", reservation_id)
            }
        }
    }

    pub fn validates_for(&self, role: LifecycleRole, entity_id: &str) -> Result<(), String> {
        self.validate()?;
        if self.role != role || self.entity_id != entity_id {
            return Err("power command target does not match coordinator".into());
        }
        Ok(())
    }
}

impl PowerCommandResult {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != POWER_PROTOCOL_VERSION {
            return Err("unsupported power protocol version".into());
        }
        validate_uuid("command_id", &self.command_id)?;
        self.authority.validate()?;
        validate_detail(self.detail.as_ref())?;
        if self.accepted == self.reason_code.is_some() {
            return Err("inconsistent power command result".into());
        }
        Ok(())
    }
}
