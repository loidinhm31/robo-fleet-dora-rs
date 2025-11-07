use serde::{Deserialize, Serialize};

/// Transcribed speech output from speech recognizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechTranscription {
    /// Transcribed text
    pub text: String,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Language detected (ISO 639-1 code, e.g., "en", "es")
    pub language: String,

    /// Duration of audio segment in milliseconds
    pub duration_ms: u64,

    /// Timestamp when transcription was generated
    pub timestamp: i64,
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
    /// Create a new speech transcription
    pub fn new(text: String, confidence: f32) -> Self {
        Self {
            text,
            confidence,
            language: "en".to_string(),
            duration_ms: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64,
        }
    }

    /// Check if transcription is empty
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Check if confidence is above threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}
