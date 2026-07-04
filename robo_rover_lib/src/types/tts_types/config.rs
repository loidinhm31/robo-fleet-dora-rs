use serde::{ser::Error as _, Deserialize, Serialize, Serializer};

use super::{validation::validate_wire_integer, VoiceState, VoiceStatus};

pub const MIN_TTS_SPEED: f32 = 0.5;
pub const MAX_TTS_SPEED: f32 = 2.0;
pub const MIN_TTS_STEPS: u8 = 1;
pub const MAX_TTS_STEPS: u8 = 20;
pub const MAX_TTS_SPEAKER_ID: u8 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsLanguage {
    En,
    Vi,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsRuntimeConfig {
    pub language: TtsLanguage,
    pub speaker_id: u8,
    #[serde(serialize_with = "serialize_canonical_float")]
    pub speed: f32,
    pub num_steps: u8,
    #[serde(serialize_with = "serialize_canonical_float")]
    pub volume: f32,
}

fn serialize_canonical_float<S>(value: &f32, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if !value.is_finite() {
        return Err(S::Error::custom("TTS float must be finite"));
    }
    if value.fract() == 0.0 {
        serializer.serialize_i64(*value as i64)
    } else {
        serializer.serialize_f32(*value)
    }
}

impl Default for TtsRuntimeConfig {
    fn default() -> Self {
        Self {
            language: TtsLanguage::En,
            speaker_id: 5,
            speed: 1.0,
            num_steps: 8,
            volume: 0.8,
        }
    }
}

impl TtsRuntimeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.speaker_id > MAX_TTS_SPEAKER_ID {
            return Err(format!("TTS speaker ID must be 0..={MAX_TTS_SPEAKER_ID}"));
        }
        if !self.speed.is_finite() || !(MIN_TTS_SPEED..=MAX_TTS_SPEED).contains(&self.speed) {
            return Err(format!(
                "TTS speed must be finite and within {MIN_TTS_SPEED}..={MAX_TTS_SPEED}"
            ));
        }
        if !(MIN_TTS_STEPS..=MAX_TTS_STEPS).contains(&self.num_steps) {
            return Err(format!(
                "TTS steps must be within {MIN_TTS_STEPS}..={MAX_TTS_STEPS}"
            ));
        }
        if !self.volume.is_finite() || !(0.0..=1.0).contains(&self.volume) {
            return Err("TTS volume must be finite and within 0..=1".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsConfigCommand {
    pub revision: u64,
    pub config: TtsRuntimeConfig,
}

impl TtsConfigCommand {
    pub fn validate(&self) -> Result<(), String> {
        validate_wire_integer(self.revision, "TTS config revision")?;
        self.config.validate()
    }
}

/// Compare-and-set request accepted from an authenticated Socket.IO client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsConfigUpdate {
    pub base_revision: u64,
    pub config: TtsRuntimeConfig,
}

impl TtsConfigUpdate {
    pub fn validate(&self) -> Result<(), String> {
        validate_wire_integer(self.base_revision, "TTS base revision")?;
        self.config.validate()
    }
}

/// Authoritative desired state plus active-rover convergence observed by the server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtsConfigState {
    pub desired_revision: u64,
    pub desired_config: TtsRuntimeConfig,
    pub applied_rovers: u32,
    pub active_rovers: u32,
    pub rovers: Vec<VoiceStatus>,
    pub timestamp: u64,
}

impl TtsConfigState {
    pub fn validate(&self) -> Result<(), String> {
        self.desired_config.validate()?;
        validate_wire_integer(self.desired_revision, "TTS desired revision")?;
        validate_wire_integer(self.timestamp, "voice timestamp")?;
        if self.applied_rovers > self.active_rovers {
            return Err("applied rover count exceeds active rover count".into());
        }
        if self.rovers.len() != self.active_rovers as usize {
            return Err("active rover count does not match rover states".into());
        }
        for status in &self.rovers {
            status.validate()?;
            if status.applied_revision > self.desired_revision {
                return Err("rover applied revision exceeds desired revision".into());
            }
        }
        let applied = self
            .rovers
            .iter()
            .filter(|status| {
                status.applied_revision == self.desired_revision
                    && status.state != VoiceState::Unavailable
                    && status.applied_config == self.desired_config
            })
            .count();
        if applied != self.applied_rovers as usize {
            return Err("applied rover count does not match rover revisions".into());
        }
        Ok(())
    }
}
