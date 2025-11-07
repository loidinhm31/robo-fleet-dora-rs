use serde::{Deserialize, Serialize};

/// Text-to-speech command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsCommand {
    pub text: String,
    pub timestamp: u64,
    pub priority: TtsPriority,
}

/// Priority for TTS messages
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum TtsPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Emergency = 3,
}

impl Default for TtsPriority {
    fn default() -> Self {
        TtsPriority::Normal
    }
}

/// TTS engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    pub rate: f32,          // Speech rate (0.5 to 2.0, default 1.0)
    pub volume: f32,        // Volume (0.0 to 1.0, default 1.0)
    pub pitch: f32,         // Pitch (0.5 to 2.0, default 1.0)
    pub voice: Option<String>, // Voice name (system-dependent)
}

impl Default for TtsConfig {
    fn default() -> Self {
        TtsConfig {
            rate: 1.0,
            volume: 1.0,
            pitch: 1.0,
            voice: None,
        }
    }
}
