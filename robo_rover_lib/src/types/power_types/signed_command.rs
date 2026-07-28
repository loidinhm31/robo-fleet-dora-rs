use super::{
    validation::{validate_id, validate_uuid},
    PowerAuthoritySnapshot, PowerCommand, PowerCommandResult, PowerTransition,
    POWER_PROTOCOL_VERSION,
};
use crate::types::lifecycle_types::LifecycleRole;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

/// Short-lived, authenticated bridge payload for a power control-plane message.
///
/// `kind`, role, target, version and lifetime are part of the signed bytes, so a
/// valid command cannot be replayed as a snapshot or acknowledgement.
pub const POWER_COMMAND_ENVELOPE_TTL_MS: u64 = 60_000;
pub const MAX_POWER_COMMAND_CLOCK_SKEW_MS: u64 = 30_000;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignedPowerEnvelopeKind {
    Command,
    CommandResult,
    Snapshot,
    Transition,
    JournalAcknowledgement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedPowerEnvelope<T> {
    pub protocol_version: u8,
    pub kind: SignedPowerEnvelopeKind,
    pub role: LifecycleRole,
    pub target_entity_id: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub payload: T,
    pub signature: Vec<u8>,
}

#[derive(Serialize)]
struct UnsignedPowerEnvelope<'a, T> {
    protocol_version: u8,
    kind: SignedPowerEnvelopeKind,
    role: LifecycleRole,
    target_entity_id: &'a str,
    issued_at_ms: u64,
    expires_at_ms: u64,
    payload: &'a T,
}

impl<T: Serialize> SignedPowerEnvelope<T> {
    pub fn new(
        kind: SignedPowerEnvelopeKind,
        role: LifecycleRole,
        target_entity_id: String,
        issued_at_ms: u64,
        payload: T,
    ) -> Self {
        Self {
            protocol_version: POWER_PROTOCOL_VERSION,
            kind,
            role,
            target_entity_id,
            issued_at_ms,
            expires_at_ms: issued_at_ms.saturating_add(POWER_COMMAND_ENVELOPE_TTL_MS),
            payload,
            signature: vec![],
        }
    }

    pub fn sign(mut self, key: &[u8]) -> Result<Self, String> {
        self.validate_envelope()?;
        let mut mac = HmacSha256::new_from_slice(key).map_err(|_| "invalid power command key")?;
        mac.update(&self.unsigned_payload()?);
        self.signature = mac.finalize().into_bytes().to_vec();
        Ok(self)
    }

    pub fn verify(&self, key: &[u8], now_ms: u64) -> Result<(), String> {
        self.validate_envelope()?;
        if self.issued_at_ms > now_ms.saturating_add(MAX_POWER_COMMAND_CLOCK_SKEW_MS)
            || now_ms >= self.expires_at_ms
            || self.signature.len() != 32
        {
            return Err("expired or malformed power command envelope".into());
        }
        let mut mac = HmacSha256::new_from_slice(key).map_err(|_| "invalid power command key")?;
        mac.update(&self.unsigned_payload()?);
        mac.verify_slice(&self.signature)
            .map_err(|_| "invalid power command signature".into())
    }

    pub fn validates_for(
        &self,
        kind: SignedPowerEnvelopeKind,
        role: LifecycleRole,
        entity_id: &str,
    ) -> Result<(), String> {
        self.validate_envelope()?;
        if self.kind != kind || self.role != role || self.target_entity_id != entity_id {
            return Err("power envelope target does not match bridge".into());
        }
        Ok(())
    }

    fn validate_envelope(&self) -> Result<(), String> {
        if self.protocol_version != POWER_PROTOCOL_VERSION {
            return Err("unsupported power command envelope version".into());
        }
        validate_id("target_entity_id", &self.target_entity_id)?;
        if self.issued_at_ms == 0
            || self.issued_at_ms >= self.expires_at_ms
            || self.expires_at_ms - self.issued_at_ms > POWER_COMMAND_ENVELOPE_TTL_MS
        {
            return Err("invalid power command envelope lifetime".into());
        }
        Ok(())
    }

    fn unsigned_payload(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&UnsignedPowerEnvelope {
            protocol_version: self.protocol_version,
            kind: self.kind,
            role: self.role,
            target_entity_id: &self.target_entity_id,
            issued_at_ms: self.issued_at_ms,
            expires_at_ms: self.expires_at_ms,
            payload: &self.payload,
        })
        .map_err(|_| "failed to encode power command envelope".into())
    }
}

pub type SignedPowerCommand = SignedPowerEnvelope<PowerCommand>;
pub type SignedPowerCommandResult = SignedPowerEnvelope<PowerCommandResult>;
pub type SignedPowerSnapshot = SignedPowerEnvelope<PowerAuthoritySnapshot>;
pub type SignedPowerTransition = SignedPowerEnvelope<PowerTransition>;
pub type SignedPowerJournalAcknowledgement = SignedPowerEnvelope<PowerJournalAcknowledgement>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerJournalAcknowledgement {
    pub protocol_version: u8,
    pub event_id: String,
    pub deployment_id: String,
}

impl PowerJournalAcknowledgement {
    pub fn validates_for(
        &self,
        entity_id: &str,
        deployment_id: Option<&str>,
    ) -> Result<(), String> {
        if self.protocol_version != POWER_PROTOCOL_VERSION {
            return Err("unsupported power journal acknowledgement version".into());
        }
        validate_uuid("event_id", &self.event_id)?;
        validate_id("deployment_id", &self.deployment_id)?;
        if deployment_id.is_some_and(|expected| expected != self.deployment_id) {
            return Err("power journal acknowledgement deployment differs".into());
        }
        validate_id("target_entity_id", entity_id)
    }
}
