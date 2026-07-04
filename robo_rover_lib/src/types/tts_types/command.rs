use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::validation::validate_timestamp;

pub const MAX_TTS_TEXT_CHARS: usize = 1_000;

/// Text-to-speech command after the web bridge has assigned correlation data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsCommand {
    /// Empty only when parsing a legacy internal payload. New producers always set it.
    #[serde(default)]
    pub command_id: String,
    pub text: String,
    pub timestamp: u64,
    pub priority: TtsPriority,
}

impl TtsCommand {
    pub fn validate(&self) -> Result<(), String> {
        Uuid::parse_str(&self.command_id)
            .map_err(|_| "TTS command ID must be a UUID".to_string())?;
        let text = self.text.trim();
        if text.is_empty() {
            return Err("TTS text must not be empty".into());
        }
        if text.chars().count() > MAX_TTS_TEXT_CHARS {
            return Err(format!("TTS text exceeds {MAX_TTS_TEXT_CHARS} characters"));
        }
        validate_timestamp(self.timestamp)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsPriority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Emergency = 3,
}
