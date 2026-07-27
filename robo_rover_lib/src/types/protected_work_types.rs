use super::{validate_id, validate_uuid, RecordingOccurrence, RecordingOccurrenceState};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

pub const PROTECTED_WORK_RELAY_PROTOCOL_VERSION: u8 = 1;
pub const MAX_PROTECTED_WORK_ITEMS: usize = 256;
pub const PROTECTED_WORK_RELAY_TTL_MS: u64 = 120_000;
pub const PROTECTED_WORK_REQUEST_TTL_MS: u64 = 30_000;
/// The relay tolerates modest NTP convergence differences between Orchestra and Rover.
pub const MAX_PROTECTED_WORK_CLOCK_SKEW_MS: u64 = 30_000;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedWorkSnapshotRequest {
    pub protocol_version: u8,
    pub request_id: String,
    pub entity_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedWorkSnapshot {
    pub protocol_version: u8,
    pub snapshot_id: String,
    pub entity_id: String,
    pub generated_at_ms: u64,
    pub occurrences: Vec<RecordingOccurrence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProtectedWorkRelayBody {
    Occurrence {
        occurrence: RecordingOccurrence,
    },
    Snapshot {
        snapshot: ProtectedWorkSnapshot,
    },
    SnapshotRequest {
        request: ProtectedWorkSnapshotRequest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedWorkRelayEnvelope {
    pub protocol_version: u8,
    pub target_entity_id: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub body: ProtectedWorkRelayBody,
    pub signature: Vec<u8>,
}

#[derive(Serialize)]
struct UnsignedEnvelope<'a> {
    protocol_version: u8,
    target_entity_id: &'a str,
    issued_at_ms: u64,
    expires_at_ms: u64,
    body: &'a ProtectedWorkRelayBody,
}

impl ProtectedWorkSnapshotRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != PROTECTED_WORK_RELAY_PROTOCOL_VERSION {
            return Err("unsupported protected-work request version".into());
        }
        validate_uuid("request_id", &self.request_id)?;
        validate_id("entity_id", &self.entity_id)
    }
}

impl ProtectedWorkSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_version != PROTECTED_WORK_RELAY_PROTOCOL_VERSION {
            return Err("unsupported protected-work snapshot version".into());
        }
        validate_uuid("snapshot_id", &self.snapshot_id)?;
        validate_id("entity_id", &self.entity_id)?;
        if self.generated_at_ms == 0 || self.occurrences.len() > MAX_PROTECTED_WORK_ITEMS {
            return Err("invalid protected-work snapshot bounds".into());
        }
        self.occurrences.iter().try_for_each(|occurrence| {
            (occurrence.entity_id == self.entity_id
                && occurrence_requires_protection(occurrence.state)
                && occurrence.updated_at_ms >= 0
                && occurrence.validate().is_ok())
            .then_some(())
            .ok_or_else(|| "invalid protected-work snapshot occurrence".into())
        })
    }
}

impl ProtectedWorkRelayEnvelope {
    pub fn new(
        target_entity_id: String,
        issued_at_ms: u64,
        ttl_ms: u64,
        body: ProtectedWorkRelayBody,
    ) -> Self {
        Self {
            protocol_version: PROTECTED_WORK_RELAY_PROTOCOL_VERSION,
            target_entity_id,
            issued_at_ms,
            expires_at_ms: issued_at_ms.saturating_add(ttl_ms),
            body,
            signature: vec![],
        }
    }

    pub fn sign(mut self, key: &[u8]) -> Result<Self, String> {
        self.validate_unsigned()?;
        let mut mac = HmacSha256::new_from_slice(key).map_err(|_| "invalid protected-work key")?;
        mac.update(&self.unsigned_payload()?);
        self.signature = mac.finalize().into_bytes().to_vec();
        Ok(self)
    }

    pub fn verify(&self, key: &[u8], now_ms: u64) -> Result<(), String> {
        self.validate_unsigned()?;
        if self.issued_at_ms > now_ms.saturating_add(MAX_PROTECTED_WORK_CLOCK_SKEW_MS)
            || now_ms > self.expires_at_ms
            || self.signature.len() != 32
        {
            return Err("expired or malformed protected-work envelope".into());
        }
        let mut mac = HmacSha256::new_from_slice(key).map_err(|_| "invalid protected-work key")?;
        mac.update(&self.unsigned_payload()?);
        mac.verify_slice(&self.signature)
            .map_err(|_| "invalid protected-work signature".into())
    }

    pub fn validate_unsigned(&self) -> Result<(), String> {
        if self.protocol_version != PROTECTED_WORK_RELAY_PROTOCOL_VERSION {
            return Err("unsupported protected-work envelope version".into());
        }
        validate_id("target_entity_id", &self.target_entity_id)?;
        if self.issued_at_ms == 0
            || self.issued_at_ms >= self.expires_at_ms
            || self.expires_at_ms - self.issued_at_ms > PROTECTED_WORK_RELAY_TTL_MS
        {
            return Err("invalid protected-work envelope lifetime".into());
        }
        let entity_id = match &self.body {
            ProtectedWorkRelayBody::Occurrence { occurrence } => {
                occurrence.validate()?;
                &occurrence.entity_id
            }
            ProtectedWorkRelayBody::Snapshot { snapshot } => {
                snapshot.validate()?;
                &snapshot.entity_id
            }
            ProtectedWorkRelayBody::SnapshotRequest { request } => {
                request.validate()?;
                &request.entity_id
            }
        };
        (entity_id == &self.target_entity_id)
            .then_some(())
            .ok_or_else(|| "protected-work envelope target mismatch".into())
    }

    fn unsigned_payload(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&UnsignedEnvelope {
            protocol_version: self.protocol_version,
            target_entity_id: &self.target_entity_id,
            issued_at_ms: self.issued_at_ms,
            expires_at_ms: self.expires_at_ms,
            body: &self.body,
        })
        .map_err(|_| "failed to encode protected-work envelope".into())
    }
}

pub fn occurrence_requires_protection(state: RecordingOccurrenceState) -> bool {
    matches!(
        state,
        RecordingOccurrenceState::StartPending
            | RecordingOccurrenceState::Active
            | RecordingOccurrenceState::StopPending
    )
}
