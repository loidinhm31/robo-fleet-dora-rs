use super::{validation::*, PowerAuthority, PowerProfile, POWER_PROTOCOL_VERSION};
use crate::types::lifecycle_types::LifecycleRole;
use serde::{Deserialize, Serialize};

const MAX_RESERVATION_TTL_MS: u64 = 604_800_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerDemandAction {
    Acquire,
    Renew,
    Release,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerDemandSource {
    Ui,
    Scheduler,
    Kws,
    Media,
    Safety,
    Maintenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerDemandPriority {
    Low,
    Normal,
    High,
    Emergency,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerDemand {
    pub protocol_version: u8,
    pub demand_id: String,
    pub action: PowerDemandAction,
    pub source: PowerDemandSource,
    pub priority: PowerDemandPriority,
    pub role: LifecycleRole,
    pub entity_id: String,
    pub required_profile: PowerProfile,
    pub authority: PowerAuthority,
    pub issued_at_ms: u64,
    pub not_before_ms: u64,
    pub expires_at_ms: u64,
    pub renew_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerReservation {
    pub protocol_version: u8,
    pub reservation_id: String,
    pub role: LifecycleRole,
    pub entity_id: String,
    pub authority: PowerAuthority,
    pub required_profile: PowerProfile,
    pub issued_at_ms: u64,
    pub not_before_ms: u64,
    pub expires_at_ms: u64,
}

impl PowerDemand {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != POWER_PROTOCOL_VERSION {
            return Err("unsupported power protocol version".into());
        }
        validate_uuid("demand_id", &self.demand_id)?;
        validate_id("entity_id", &self.entity_id)?;
        self.authority.validate()?;
        self.required_profile.validate_for_role(self.role)?;
        if !self.source.allows(self.required_profile) || self.renew_sequence == 0 {
            return Err("invalid power demand source, profile, or renew sequence".into());
        }
        validate_window(
            self.issued_at_ms,
            self.not_before_ms,
            self.expires_at_ms,
            self.source.max_ttl_ms(),
        )
    }

    pub fn validates_for(&self, role: LifecycleRole, entity_id: &str) -> Result<(), String> {
        self.validate()?;
        if self.role != role || self.entity_id != entity_id {
            return Err("power demand target does not match coordinator".into());
        }
        Ok(())
    }

    pub fn same_immutable_payload(&self, other: &Self) -> bool {
        self.demand_id == other.demand_id
            && self.source == other.source
            && self.priority == other.priority
            && self.role == other.role
            && self.entity_id == other.entity_id
            && self.required_profile == other.required_profile
            && self.authority == other.authority
            && self.not_before_ms == other.not_before_ms
    }
}

impl PowerReservation {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != POWER_PROTOCOL_VERSION {
            return Err("unsupported power protocol version".into());
        }
        validate_uuid("reservation_id", &self.reservation_id)?;
        validate_id("entity_id", &self.entity_id)?;
        self.authority.validate()?;
        if self.required_profile != PowerProfile::ScheduledCapture {
            return Err("power reservation must request scheduled_capture".into());
        }
        self.required_profile.validate_for_role(self.role)?;
        validate_window(
            self.issued_at_ms,
            self.not_before_ms,
            self.expires_at_ms,
            MAX_RESERVATION_TTL_MS,
        )
    }

    pub fn validates_for(&self, role: LifecycleRole, entity_id: &str) -> Result<(), String> {
        self.validate()?;
        if self.role != role || self.entity_id != entity_id {
            return Err("power reservation target does not match coordinator".into());
        }
        Ok(())
    }

    /// Retain a replay fence for the longest validity window accepted by V1.
    pub fn tombstone_expires_at_ms(&self) -> u64 {
        self.issued_at_ms.saturating_add(MAX_RESERVATION_TTL_MS)
    }
}

impl PowerDemandSource {
    pub const fn max_ttl_ms(self) -> u64 {
        match self {
            Self::Ui => 120_000,
            Self::Scheduler | Self::Kws => 300_000,
            Self::Media => 900_000,
            Self::Safety | Self::Maintenance => 3_600_000,
        }
    }

    pub const fn capacity(self) -> usize {
        match self {
            Self::Ui | Self::Media => 32,
            Self::Scheduler => 16,
            Self::Kws | Self::Safety | Self::Maintenance => 8,
        }
    }

    fn allows(self, profile: PowerProfile) -> bool {
        matches!(
            (self, profile),
            (
                Self::Ui,
                PowerProfile::NormalRover | PowerProfile::OrchestraSpeech
            ) | (Self::Scheduler, PowerProfile::ScheduledCapture)
                | (Self::Kws, PowerProfile::NormalRover)
                | (
                    Self::Media,
                    PowerProfile::NormalRover | PowerProfile::ScheduledCapture
                )
                | (
                    Self::Safety,
                    PowerProfile::NormalRover | PowerProfile::OrchestraSpeech
                )
                | (
                    Self::Maintenance,
                    PowerProfile::NormalRover | PowerProfile::OrchestraSpeech
                )
        )
    }
}

fn validate_window(issued: u64, not_before: u64, expires: u64, max_ttl: u64) -> Result<(), String> {
    if issued == 0 || issued > not_before || not_before >= expires || expires - issued > max_ttl {
        return Err("invalid power demand or reservation timestamps".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LifecycleRole, PowerAuthority, PowerDemandPriority};

    fn demand(expires_at_ms: u64) -> PowerDemand {
        PowerDemand {
            protocol_version: POWER_PROTOCOL_VERSION,
            demand_id: "f4f3e2d1-c0b9-48a7-9615-141312111000".into(),
            action: PowerDemandAction::Acquire,
            source: PowerDemandSource::Ui,
            priority: PowerDemandPriority::Normal,
            role: LifecycleRole::Rover,
            entity_id: "rover".into(),
            required_profile: PowerProfile::NormalRover,
            authority: PowerAuthority {
                epoch: 1,
                sequence: 1,
            },
            issued_at_ms: 1,
            not_before_ms: 1,
            expires_at_ms,
            renew_sequence: 1,
        }
    }

    #[test]
    fn ui_ttl_is_bounded_to_two_minutes() {
        assert!(demand(120_002).validate().is_err());
        assert!(demand(120_001).validate().is_ok());
    }
}
