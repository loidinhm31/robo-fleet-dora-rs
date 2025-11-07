use serde::{Deserialize, Serialize};

/// Intent classification for natural language commands
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    // Motion control
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    TurnLeft,
    TurnRight,
    Stop,

    // Arm control
    MoveArmUp,
    MoveArmDown,
    MoveArmLeft,
    MoveArmRight,
    MoveArmForward,
    MoveArmBackward,
    OpenGripper,
    CloseGripper,

    // Vision/Tracking
    TrackObject,
    StopTracking,
    FollowObject,
    StopFollowing,

    // Camera control
    StartCamera,
    StopCamera,

    // Audio control
    StartAudio,
    StopAudio,

    // System
    Unknown,
}

/// Extracted entities from natural language command
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityExtraction {
    /// Distance in meters (e.g., "move forward 2 meters")
    pub distance: Option<f32>,

    /// Angle in degrees (e.g., "turn left 90 degrees")
    pub angle: Option<f32>,

    /// Speed multiplier 0.0-1.0 (e.g., "move slowly" -> 0.3)
    pub speed: Option<f32>,

    /// Object name to track (e.g., "track person", "follow dog")
    pub object_name: Option<String>,

    /// Duration in seconds (e.g., "move forward for 5 seconds")
    pub duration: Option<f32>,
}

/// Result of parsing a natural language command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedCommand {
    /// Classified intent
    pub intent: Intent,

    /// Extracted entities
    pub entities: EntityExtraction,

    /// Confidence score 0.0-1.0
    pub confidence: f32,

    /// Original text input
    pub raw_text: String,

    /// Timestamp when parsed (Unix milliseconds)
    pub timestamp: u64,
}

impl ParsedCommand {
    /// Create a new parsed command
    pub fn new(intent: Intent, raw_text: String) -> Self {
        Self {
            intent,
            entities: EntityExtraction::default(),
            confidence: 1.0,
            raw_text,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add extracted entities
    pub fn with_entities(mut self, entities: EntityExtraction) -> Self {
        self.entities = entities;
        self
    }
}
