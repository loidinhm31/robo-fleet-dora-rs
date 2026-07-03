use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SttProfile {
    EnVadOffline,
    ViVadOffline,
}

impl SttProfile {
    pub fn language_code(self) -> &'static str {
        match self {
            Self::EnVadOffline => "en",
            Self::ViVadOffline => "vi",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SttState {
    Loading,
    Ready,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SttSourceKind {
    Browser,
    Rover,
}

/// Final-only STT output shared between Rust nodes and the web UI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechTranscription {
    /// Transcribed text.
    pub text: String,

    /// Optional confidence. Producers emit `null` when unavailable and accept the
    /// field being omitted during deserialization for limited backward parsing.
    #[serde(default)]
    pub confidence: Option<f32>,

    /// Recognized language code tied to the active STT profile.
    pub language: String,

    /// Duration of the finalized utterance in milliseconds.
    pub duration_ms: u64,

    /// Unix timestamp in milliseconds when transcription was generated.
    pub timestamp: i64,

    /// Stable utterance identifier for cross-node correlation.
    pub utterance_id: String,

    /// Stable capture stream identifier.
    pub stream_id: String,

    /// Whether the utterance originated in the browser or on a rover.
    pub source_kind: SttSourceKind,

    /// Source rover identity for rover-origin speech, otherwise `null`.
    pub entity_id: Option<String>,

    /// Authoritative rover target captured by the server for this utterance.
    pub target_entity_id: String,

    /// Startup-selected STT profile serving every stream.
    pub profile: SttProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SttStatus {
    pub state: SttState,
    pub profile: SttProfile,
    pub language: String,
    pub timestamp: i64,
    pub error: Option<String>,
}

/// Speech recognition statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechStats {
    /// Total number of transcriptions
    pub total_transcriptions: u64,

    /// Average confidence score
    pub avg_confidence: f32,

    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f32,

    /// Number of failed transcriptions
    pub failed_transcriptions: u64,
}

impl SpeechTranscription {
    /// Build a browser-origin final transcription for the current transitional
    /// Whisper path until the dual-source transport lands.
    pub fn new_browser(
        text: String,
        confidence: Option<f32>,
        duration_ms: u64,
        stream_id: String,
        target_entity_id: String,
        profile: SttProfile,
    ) -> Self {
        Self {
            text,
            confidence,
            language: profile.language_code().to_string(),
            duration_ms,
            timestamp: current_timestamp_ms(),
            utterance_id: Uuid::new_v4().to_string(),
            stream_id,
            source_kind: SttSourceKind::Browser,
            entity_id: None,
            target_entity_id,
            profile,
        }
    }

    /// Check if transcription is empty
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Check if confidence is above threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence
            .is_some_and(|confidence| confidence >= threshold)
    }
}

fn current_timestamp_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
