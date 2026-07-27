use super::{validation::*, PowerDemandPriority, PowerDemandSource, POWER_PROTOCOL_VERSION};
use crate::types::lifecycle_types::LifecycleRole;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerPolicy {
    Awake,
    Auto,
    Sleep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerProfile {
    Dormant,
    IdleListening,
    ScheduledCapture,
    NormalRover,
    OrchestraIdle,
    OrchestraSpeech,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerState {
    AuthorityUnknown,
    Active,
    IdlePending,
    Quiescing,
    IdleListening,
    Dormant,
    Prewarming,
    Waking,
    Degraded,
    Failed,
}

/// Transport-neutral authority result used before a coordinator emits a
/// remote profile command. A fresh snapshot authorizes exactly one newer
/// authority value; every other condition is deliberately observation-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerAuthorityDecision {
    ObserveOnly,
    CommandAllowed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerReasonCode {
    InvalidRequest,
    InvalidTarget,
    InvalidAuthority,
    Unsupported,
    Conflict,
    Expired,
    StaleAuthority,
    DuplicateMismatch,
    SnapshotMissing,
    SnapshotStale,
    CapacityExceeded,
    Timeout,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerAuthority {
    pub epoch: u64,
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerAuthoritySnapshot {
    pub protocol_version: u8,
    pub snapshot_id: String,
    pub role: LifecycleRole,
    pub entity_id: String,
    pub authority: PowerAuthority,
    pub state: PowerState,
    pub effective_profile: PowerProfile,
    pub captured_at_ms: u64,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerTransition {
    pub protocol_version: u8,
    pub transition_id: String,
    pub role: LifecycleRole,
    pub entity_id: String,
    pub authority: PowerAuthority,
    pub requested_profile: PowerProfile,
    pub effective_profile: PowerProfile,
    pub state: PowerState,
    pub issued_at_ms: u64,
    pub deadline_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerStatus {
    pub protocol_version: u8,
    pub role: LifecycleRole,
    pub entity_id: String,
    pub authority: PowerAuthority,
    pub policy: PowerPolicy,
    pub requested_profile: PowerProfile,
    pub effective_profile: PowerProfile,
    pub state: PowerState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transition_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<PowerReasonCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Reservation-specific evidence. Aggregate profile state never grants
    /// scheduled-recorder admission without this exact reservation ID.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub active_reservations: Vec<PowerReservationReadiness>,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerReservationReadiness {
    pub reservation_id: String,
    pub activation_started_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerEvent {
    pub protocol_version: u8,
    pub event_id: String,
    pub role: LifecycleRole,
    pub entity_id: String,
    pub authority: PowerAuthority,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transition_id: Option<String>,
    pub event_type: PowerEventType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<PowerReasonCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Bounded, operator-safe context required to explain a durable power event.
    #[serde(default, skip_serializing_if = "PowerEventContext::is_empty")]
    pub context: PowerEventContext,
    pub occurred_at_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerEventContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_action: Option<PowerCommandActionKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<PowerPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub demand_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub demand_source: Option<PowerDemandSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_profile: Option<PowerProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub demand_priority: Option<PowerDemandPriority>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub not_before_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reservation_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lifecycle_targets: Vec<PowerLifecycleTargetContext>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerCommandActionKind {
    SetPolicy,
    RegisterDemand,
    ReleaseDemand,
    RegisterReservation,
    ReleaseReservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerLifecycleTargetContext {
    pub node_id: String,
    pub manager_epoch: u64,
    pub expected_revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerEventType {
    PolicyChanged,
    DemandChanged,
    CommandApplied,
    TransitionRequested,
    TransitionApplied,
    TransitionFailed,
    SnapshotObserved,
}

impl PowerAuthority {
    pub fn validate(&self) -> Result<(), String> {
        if self.epoch == 0 || self.sequence == 0 {
            Err("invalid power authority".into())
        } else {
            Ok(())
        }
    }

    /// Normal commands use the receiver's current stamp and advance its
    /// sequence after apply. A reconnect may establish exactly the next epoch
    /// at sequence one. No other authority jump is admissible.
    pub fn accepts_command_authority(&self, proposed: Self) -> bool {
        (proposed == *self && self.next_sequence().is_some())
            || self.next_epoch().is_some_and(|next| proposed == next)
    }

    /// The sole epoch-reconciliation successor. This intentionally differs
    /// from `next_sequence`: Orchestra must use it after observing a Rover
    /// snapshot, because the Rover's sequence is not Orchestra's local one.
    pub fn next_epoch(&self) -> Option<Self> {
        self.epoch
            .checked_add(1)
            .map(|epoch| Self { epoch, sequence: 1 })
    }

    pub fn next_sequence(&self) -> Option<Self> {
        self.sequence.checked_add(1).map(|sequence| Self {
            epoch: self.epoch,
            sequence,
        })
    }
}

impl PowerProfile {
    pub fn validate_for_role(self, role: LifecycleRole) -> Result<(), String> {
        let valid = matches!(
            (role, self),
            (
                LifecycleRole::Rover,
                Self::Dormant | Self::IdleListening | Self::ScheduledCapture | Self::NormalRover
            ) | (
                LifecycleRole::Orchestra,
                Self::OrchestraIdle | Self::OrchestraSpeech
            )
        );
        valid
            .then_some(())
            .ok_or_else(|| "power profile is invalid for role".into())
    }
}

impl PowerAuthoritySnapshot {
    pub fn validate(&self) -> Result<(), String> {
        valid_version(self.protocol_version)?;
        validate_uuid("snapshot_id", &self.snapshot_id)?;
        validate_id("entity_id", &self.entity_id)?;
        self.authority.validate()?;
        self.effective_profile.validate_for_role(self.role)?;
        if self.captured_at_ms == 0 || self.captured_at_ms >= self.expires_at_ms {
            return Err("invalid power snapshot timestamps".into());
        }
        Ok(())
    }

    pub fn validates_for(&self, role: LifecycleRole, entity_id: &str) -> Result<(), String> {
        self.validate()?;
        if self.role != role || self.entity_id != entity_id {
            return Err("power snapshot target does not match coordinator".into());
        }
        Ok(())
    }
}

impl PowerTransition {
    pub fn validate(&self) -> Result<(), String> {
        valid_version(self.protocol_version)?;
        validate_uuid("transition_id", &self.transition_id)?;
        validate_id("entity_id", &self.entity_id)?;
        self.authority.validate()?;
        self.requested_profile.validate_for_role(self.role)?;
        self.effective_profile.validate_for_role(self.role)?;
        if self.issued_at_ms == 0 || self.issued_at_ms >= self.deadline_at_ms {
            return Err("invalid power transition deadline".into());
        }
        Ok(())
    }

    pub fn validates_for(&self, role: LifecycleRole, entity_id: &str) -> Result<(), String> {
        self.validate()?;
        if self.role != role || self.entity_id != entity_id {
            return Err("power transition target does not match coordinator".into());
        }
        Ok(())
    }
}

impl PowerStatus {
    pub fn validate(&self) -> Result<(), String> {
        valid_version(self.protocol_version)?;
        validate_id("entity_id", &self.entity_id)?;
        self.authority.validate()?;
        self.requested_profile.validate_for_role(self.role)?;
        self.effective_profile.validate_for_role(self.role)?;
        validate_optional_uuid("transition_id", self.transition_id.as_deref())?;
        validate_detail(self.detail.as_ref())?;
        if self.active_reservations.len() > 128 {
            return Err("too many active reservation readiness records".into());
        }
        for readiness in &self.active_reservations {
            validate_uuid("reservation_id", &readiness.reservation_id)?;
            if readiness.activation_started_at_ms == 0 {
                return Err("invalid reservation activation timestamp".into());
            }
        }
        if self.updated_at_ms == 0
            || (self.state == PowerState::AuthorityUnknown && self.transition_id.is_some())
        {
            return Err("invalid power status".into());
        }
        Ok(())
    }

    pub fn validates_for(&self, role: LifecycleRole, entity_id: &str) -> Result<(), String> {
        self.validate()?;
        if self.role != role || self.entity_id != entity_id {
            return Err("power status target does not match coordinator".into());
        }
        Ok(())
    }
}

impl PowerEvent {
    pub fn validate(&self) -> Result<(), String> {
        valid_version(self.protocol_version)?;
        validate_uuid("event_id", &self.event_id)?;
        validate_id("entity_id", &self.entity_id)?;
        self.authority.validate()?;
        validate_optional_uuid("transition_id", self.transition_id.as_deref())?;
        validate_detail(self.detail.as_ref())?;
        self.context.validate()?;
        if self.occurred_at_ms == 0 {
            return Err("invalid power event timestamp".into());
        }
        Ok(())
    }
}

impl PowerEventContext {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_optional_uuid("command_id", self.command_id.as_deref())?;
        validate_optional_uuid("demand_id", self.demand_id.as_deref())?;
        validate_optional_uuid("reservation_id", self.reservation_id.as_deref())?;
        if let (Some(not_before), Some(expires)) = (self.not_before_ms, self.expires_at_ms) {
            if not_before >= expires {
                return Err("invalid event context window".into());
            }
        }
        if self.lifecycle_targets.len() > 32 {
            return Err("too many lifecycle target contexts".into());
        }
        for target in &self.lifecycle_targets {
            validate_id("lifecycle target node_id", &target.node_id)?;
            if target.manager_epoch == 0 {
                return Err("invalid lifecycle target epoch".into());
            }
        }
        Ok(())
    }
}

/// Coordinator-side reconnect fence. It makes missing or expired remote state
/// observable and prevents profile authority from being guessed during a split.
#[derive(Debug, Clone)]
pub struct PowerSnapshotGate {
    role: LifecycleRole,
    entity_id: String,
    snapshot: Option<PowerAuthoritySnapshot>,
    last_consumed_authority: Option<PowerAuthority>,
}

impl PowerSnapshotGate {
    pub fn new(role: LifecycleRole, entity_id: String) -> Result<Self, String> {
        validate_id("entity_id", &entity_id)?;
        Ok(Self {
            role,
            entity_id,
            snapshot: None,
            last_consumed_authority: None,
        })
    }

    pub fn observe(&mut self, snapshot: PowerAuthoritySnapshot, now_ms: u64) -> Result<(), String> {
        let observed = (|| {
            snapshot.validates_for(self.role, &self.entity_id)?;
            if snapshot.captured_at_ms > now_ms || snapshot.expires_at_ms <= now_ms {
                return Err("power snapshot is not fresh".into());
            }
            if self
                .last_consumed_authority
                .is_some_and(|authority| snapshot.authority < authority)
                || self
                    .snapshot
                    .as_ref()
                    .is_some_and(|current| current.authority > snapshot.authority)
            {
                return Err("power snapshot authority is stale".into());
            }
            Ok(())
        })();
        if observed.is_err() {
            // A malformed or reordered observation invalidates the current
            // reconciliation grant; later commands require a fresh snapshot.
            self.snapshot = None;
            return observed;
        }
        self.snapshot = Some(snapshot);
        Ok(())
    }

    pub fn state(&self, now_ms: u64) -> PowerState {
        self.snapshot
            .as_ref()
            .filter(|snapshot| snapshot.captured_at_ms <= now_ms && snapshot.expires_at_ms > now_ms)
            .map_or(PowerState::AuthorityUnknown, |snapshot| snapshot.state)
    }

    pub fn consume_profile_authority(
        &mut self,
        authority: PowerAuthority,
        now_ms: u64,
    ) -> PowerAuthorityDecision {
        let Some(snapshot) = self.snapshot.as_ref() else {
            return PowerAuthorityDecision::ObserveOnly;
        };
        if snapshot.captured_at_ms > now_ms
            || snapshot.expires_at_ms <= now_ms
            || snapshot.authority.next_epoch() != Some(authority)
            || self
                .last_consumed_authority
                .is_some_and(|last| snapshot.authority <= last || authority <= last)
        {
            return PowerAuthorityDecision::ObserveOnly;
        }
        self.last_consumed_authority = Some(authority);
        self.snapshot = None;
        PowerAuthorityDecision::CommandAllowed
    }
}

fn valid_version(version: u8) -> Result<(), String> {
    (version == POWER_PROTOCOL_VERSION)
        .then_some(())
        .ok_or_else(|| "unsupported power protocol version".into())
}
fn validate_optional_uuid(label: &str, value: Option<&str>) -> Result<(), String> {
    value.map_or(Ok(()), |value| validate_uuid(label, value))
}
