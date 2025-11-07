use aho_corasick::AhoCorasick;
use arrow::array::{Array, BinaryArray, StringArray};
use dora_node_api::{dora_core::config::DataId, DoraNode, Event};
use eyre::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use robo_rover_lib::{init_tracing, types::*};
use std::collections::HashMap;

/// Pattern for matching a specific intent
#[derive(Debug)]
#[derive(Clone)]
struct IntentPattern {
    intent: Intent,
    patterns: Vec<Regex>,
}

impl IntentPattern {
    fn new(intent: Intent, patterns: Vec<&str>) -> Self {
        Self {
            intent,
            patterns: patterns.iter().map(|p| Regex::new(p).unwrap()).collect(),
        }
    }

    fn matches(&self, text: &str) -> bool {
        self.patterns.iter().any(|p| p.is_match(text))
    }
}

/// Command parser with hybrid Aho-Corasick + Regex matching
struct CommandParser {
    // Fast keyword matching with Aho-Corasick
    keyword_matcher: AhoCorasick,
    keyword_intents: Vec<Intent>,

    // Complex pattern matching with Regex (for entity extraction)
    regex_patterns: Vec<IntentPattern>,

    confidence_threshold: f32,
}

impl CommandParser {
    fn new() -> Self {
        // Define simple keywords for Aho-Corasick (case-insensitive)
        // Format: (keyword, intent)
        let keyword_mappings = vec![
            // Motion - simple commands
            ("stop", Intent::Stop),
            ("halt", Intent::Stop),
            ("freeze", Intent::Stop),
            ("brake", Intent::Stop),
            ("forward", Intent::MoveForward),
            ("ahead", Intent::MoveForward),
            ("backward", Intent::MoveBackward),
            ("back", Intent::MoveBackward),
            ("reverse", Intent::MoveBackward),
            ("left", Intent::MoveLeft),
            ("right", Intent::MoveRight),
            // Arm - simple commands
            ("open gripper", Intent::OpenGripper),
            ("close gripper", Intent::CloseGripper),
            ("grab", Intent::CloseGripper),
            ("grasp", Intent::CloseGripper),
            ("release", Intent::OpenGripper),
            // Vision
            ("stop tracking", Intent::StopTracking),
            ("stop following", Intent::StopFollowing),
            // Camera
            ("start camera", Intent::StartCamera),
            ("stop camera", Intent::StopCamera),
            // Audio
            ("start audio", Intent::StartAudio),
            ("stop audio", Intent::StopAudio),
            ("start microphone", Intent::StartAudio),
            ("stop microphone", Intent::StopAudio),
        ];

        let keywords: Vec<&str> = keyword_mappings.iter().map(|(k, _)| *k).collect();
        let intents: Vec<Intent> = keyword_mappings.iter().map(|(_, i)| i.clone()).collect();

        // Build Aho-Corasick automaton (case-insensitive)
        let keyword_matcher = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&keywords)
            .expect("Failed to build Aho-Corasick automaton");

        // Define complex regex patterns for entity extraction and compound commands
        let regex_patterns = vec![
            // Motion control - with entities
            IntentPattern::new(
                Intent::MoveForward,
                vec![
                    r"(?i)(move|go|drive|head)\s+(forward|ahead|front|straight)",
                    r"(?i)(advance|proceed)\s*(forward)?",
                ],
            ),
            IntentPattern::new(
                Intent::MoveBackward,
                vec![
                    r"(?i)(move|go|drive|head)\s+(back|backward|reverse)",
                    r"(?i)back\s*up",
                ],
            ),
            IntentPattern::new(
                Intent::MoveLeft,
                vec![
                    r"(?i)(move|go|drive|shift|slide)\s+(left|port)",
                    r"(?i)strafe\s+left",
                ],
            ),
            IntentPattern::new(
                Intent::MoveRight,
                vec![
                    r"(?i)(move|go|drive|shift|slide)\s+(right|starboard)",
                    r"(?i)strafe\s+right",
                ],
            ),
            IntentPattern::new(
                Intent::TurnLeft,
                vec![
                    r"(?i)(turn|rotate|spin)\s+(left|counter\s*clock)",
                    r"(?i)left\s+turn",
                ],
            ),
            IntentPattern::new(
                Intent::TurnRight,
                vec![
                    r"(?i)(turn|rotate|spin)\s+(right|clock\s*wise)",
                    r"(?i)right\s+turn",
                ],
            ),
            // Arm control
            IntentPattern::new(
                Intent::MoveArmUp,
                vec![
                    r"(?i)(move|raise|lift)\s+(the\s+)?arm\s+up",
                    r"(?i)arm\s+up",
                    r"(?i)raise\s+(the\s+)?arm",
                ],
            ),
            IntentPattern::new(
                Intent::MoveArmDown,
                vec![
                    r"(?i)(move|lower)\s+(the\s+)?arm\s+down",
                    r"(?i)arm\s+down",
                    r"(?i)lower\s+(the\s+)?arm",
                ],
            ),
            IntentPattern::new(
                Intent::MoveArmLeft,
                vec![
                    r"(?i)(move|swing)\s+(the\s+)?arm\s+left",
                    r"(?i)arm\s+left",
                ],
            ),
            IntentPattern::new(
                Intent::MoveArmRight,
                vec![
                    r"(?i)(move|swing)\s+(the\s+)?arm\s+right",
                    r"(?i)arm\s+right",
                ],
            ),
            IntentPattern::new(
                Intent::MoveArmForward,
                vec![
                    r"(?i)(extend|reach)\s+(the\s+)?arm\s+(forward|out)",
                    r"(?i)arm\s+(forward|out)",
                ],
            ),
            IntentPattern::new(
                Intent::MoveArmBackward,
                vec![
                    r"(?i)(retract|pull)\s+(the\s+)?arm\s+(back|in)",
                    r"(?i)arm\s+(back|in)",
                ],
            ),
            // Vision control - with object names
            IntentPattern::new(
                Intent::TrackObject,
                vec![
                    r"(?i)track\s+(the\s+)?(\w+)",
                    r"(?i)start\s+tracking",
                ],
            ),
            IntentPattern::new(
                Intent::FollowObject,
                vec![
                    r"(?i)follow\s+(the\s+)?(\w+)",
                    r"(?i)start\s+following",
                ],
            ),
            // Camera control - detailed
            IntentPattern::new(
                Intent::StartCamera,
                vec![
                    r"(?i)turn\s+on\s+(the\s+)?camera",
                    r"(?i)enable\s+(the\s+)?camera",
                ],
            ),
            IntentPattern::new(
                Intent::StopCamera,
                vec![
                    r"(?i)turn\s+off\s+(the\s+)?camera",
                    r"(?i)disable\s+(the\s+)?camera",
                ],
            ),
            // Audio control - detailed
            IntentPattern::new(
                Intent::StartAudio,
                vec![
                    r"(?i)turn\s+on\s+(the\s+)?(audio|microphone|mic)",
                    r"(?i)enable\s+(the\s+)?(audio|microphone|mic)",
                ],
            ),
            IntentPattern::new(
                Intent::StopAudio,
                vec![
                    r"(?i)turn\s+off\s+(the\s+)?(audio|microphone|mic)",
                    r"(?i)disable\s+(the\s+)?(audio|microphone|mic)",
                ],
            ),
        ];

        let confidence_threshold = std::env::var("CONFIDENCE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);

        Self {
            keyword_matcher,
            keyword_intents: intents,
            regex_patterns,
            confidence_threshold,
        }
    }

    /// Parse natural language text into a command
    fn parse(&self, text: &str) -> Result<ParsedCommand> {
        // Clean up speech recognition artifacts
        let cleaned_text = preprocess_text(text);

        tracing::debug!("Original: '{}' -> Cleaned: '{}'", text, cleaned_text);

        // PHASE 1: Fast keyword matching with Aho-Corasick
        if let Some(mat) = self.keyword_matcher.find(&cleaned_text) {
            let matched_intent = &self.keyword_intents[mat.pattern()];
            tracing::debug!("Aho-Corasick matched: {:?} (keyword: '{}')", matched_intent, &cleaned_text[mat.start()..mat.end()]);

            let entities = self.extract_entities(&cleaned_text, matched_intent);
            let parsed = ParsedCommand::new(matched_intent.clone(), text.to_string())
                .with_entities(entities)
                .with_confidence(0.95); // Very high confidence for exact keyword match

            tracing::info!(
                "Parsed via Aho-Corasick: {:?} with confidence {}",
                parsed.intent,
                parsed.confidence
            );

            return Ok(parsed);
        }

        // PHASE 2: Complex regex pattern matching (for entity extraction and compound commands)
        for pattern in &self.regex_patterns {
            if pattern.matches(&cleaned_text) {
                tracing::debug!("Regex matched: {:?}", pattern.intent);
                let entities = self.extract_entities(&cleaned_text, &pattern.intent);
                let parsed = ParsedCommand::new(pattern.intent.clone(), text.to_string())
                    .with_entities(entities)
                    .with_confidence(0.85); // High confidence for regex pattern match

                tracing::info!(
                    "Parsed via Regex: {:?} with confidence {}",
                    parsed.intent,
                    parsed.confidence
                );

                return Ok(parsed);
            }
        }

        // No match found
        tracing::warn!("No pattern matched for: '{}'", cleaned_text);
        Ok(ParsedCommand::new(Intent::Unknown, text.to_string()).with_confidence(0.0))
    }

    /// Extract entities from text based on intent
    fn extract_entities(&self, text: &str, intent: &Intent) -> EntityExtraction {
        let mut entities = EntityExtraction::default();

        // Extract common entities
        entities.distance = extract_distance(text);
        entities.angle = extract_angle(text);
        entities.speed = extract_speed(text);
        entities.duration = extract_duration(text);

        // Intent-specific entity extraction
        match intent {
            Intent::TrackObject | Intent::FollowObject => {
                entities.object_name = extract_object(text);
            }
            _ => {}
        }

        entities
    }
}

// Text preprocessing for speech recognition artifacts

/// Preprocess speech recognition text to remove artifacts and normalize
fn preprocess_text(text: &str) -> String {
    let mut cleaned = text.trim().to_string();

    // Remove common speech recognition artifacts
    cleaned = cleaned.replace("[BLANK_AUDIO]", "");
    cleaned = cleaned.replace("[MUSIC]", "");
    cleaned = cleaned.replace("[NOISE]", "");
    cleaned = cleaned.replace("[SILENCE]", "");

    // Remove extra punctuation at the end
    cleaned = cleaned.trim_end_matches(|c: char| c == '.' || c == ',' || c == '!' || c == '?').to_string();

    // Collapse multiple spaces
    let re = Regex::new(r"\s+").unwrap();
    cleaned = re.replace_all(&cleaned, " ").to_string();

    // Final trim
    cleaned.trim().to_string()
}

// Entity extraction functions

static DISTANCE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(meter|metre|m|feet|foot|ft)").unwrap());
static DISTANCE_WORDS: Lazy<HashMap<&str, f32>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("half", 0.5);
    m.insert("one", 1.0);
    m.insert("two", 2.0);
    m.insert("three", 3.0);
    m.insert("four", 4.0);
    m.insert("five", 5.0);
    m
});

fn extract_distance(text: &str) -> Option<f32> {
    // Try numeric pattern first
    if let Some(cap) = DISTANCE_REGEX.captures(text) {
        if let Ok(value) = cap[1].parse::<f32>() {
            let unit = cap[2].to_lowercase();
            let meters = match unit.as_str() {
                "feet" | "foot" | "ft" => value * 0.3048,
                _ => value,
            };
            return Some(meters);
        }
    }

    // Try word-based distances
    let text_lower = text.to_lowercase();
    for (word, distance) in DISTANCE_WORDS.iter() {
        if text_lower.contains(word) {
            return Some(*distance);
        }
    }

    None
}

static ANGLE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(degree|deg|°)").unwrap());

fn extract_angle(text: &str) -> Option<f32> {
    ANGLE_REGEX.captures(text).and_then(|cap| cap[1].parse::<f32>().ok())
}

fn extract_speed(text: &str) -> Option<f32> {
    let text_lower = text.to_lowercase();

    if text_lower.contains("very slow") || text_lower.contains("super slow") {
        Some(0.2)
    } else if text_lower.contains("slow") {
        Some(0.3)
    } else if text_lower.contains("normal") || text_lower.contains("medium") {
        Some(0.5)
    } else if text_lower.contains("fast") {
        Some(0.8)
    } else if text_lower.contains("very fast") || text_lower.contains("super fast") {
        Some(1.0)
    } else {
        // Try numeric pattern
        let re = Regex::new(r"(?i)speed\s+(\d+(?:\.\d+)?)").unwrap();
        re.captures(text)
            .and_then(|cap| cap[1].parse::<f32>().ok())
    }
}

static DURATION_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)for\s+(\d+(?:\.\d+)?)\s*(second|sec|s|minute|min|m)").unwrap());

fn extract_duration(text: &str) -> Option<f32> {
    DURATION_REGEX.captures(text).and_then(|cap| {
        let value = cap[1].parse::<f32>().ok()?;
        let unit = cap[2].to_lowercase();
        let seconds = match unit.as_str() {
            "minute" | "min" | "m" => value * 60.0,
            _ => value,
        };
        Some(seconds)
    })
}

static YOLO_CLASSES: Lazy<Vec<&str>> = Lazy::new(|| {
    vec![
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ]
});

fn extract_object(text: &str) -> Option<String> {
    let text_lower = text.to_lowercase();
    YOLO_CLASSES
        .iter()
        .find(|&&class| text_lower.contains(class))
        .map(|s| s.to_string())
}

/// Convert ParsedCommand to appropriate output commands
fn convert_to_rover_command(parsed: &ParsedCommand) -> Option<RoverCommandWithMetadata> {
    let speed = parsed.entities.speed.unwrap_or(0.5) as f64;

    let command = match parsed.intent {
        Intent::MoveForward => RoverCommand::new_velocity(0.0, speed, 0.0),
        Intent::MoveBackward => RoverCommand::new_velocity(0.0, -speed, 0.0),
        Intent::MoveLeft => RoverCommand::new_velocity(0.0, 0.0, speed),
        Intent::MoveRight => RoverCommand::new_velocity(0.0, 0.0, -speed),
        Intent::TurnLeft => RoverCommand::new_velocity(speed, 0.0, 0.0),
        Intent::TurnRight => RoverCommand::new_velocity(-speed, 0.0, 0.0),
        Intent::Stop => RoverCommand::new_stop(),
        _ => return None,
    };

    Some(RoverCommandWithMetadata {
        command,
        metadata: CommandMetadata {
            command_id: uuid::Uuid::new_v4().to_string(),
            timestamp: parsed.timestamp,
            source: InputSource::VoiceCommand,
            priority: CommandPriority::Normal,
        },
    })
}

fn convert_to_tracking_command(parsed: &ParsedCommand) -> Option<TrackingCommand> {
    match parsed.intent {
        Intent::TrackObject => Some(TrackingCommand::Enable {
            timestamp: parsed.timestamp,
        }),
        Intent::StopTracking => Some(TrackingCommand::Disable {
            timestamp: parsed.timestamp,
        }),
        Intent::FollowObject => {
            // Enable tracking - visual servo will handle following
            Some(TrackingCommand::Enable {
                timestamp: parsed.timestamp,
            })
        }
        Intent::StopFollowing => Some(TrackingCommand::Disable {
            timestamp: parsed.timestamp,
        }),
        _ => None,
    }
}

fn convert_to_camera_control(parsed: &ParsedCommand) -> Option<CameraControl> {
    let command = match parsed.intent {
        Intent::StartCamera => CameraAction::Start,
        Intent::StopCamera => CameraAction::Stop,
        _ => return None,
    };

    Some(CameraControl {
        command,
        timestamp: parsed.timestamp,
    })
}

/// Generate natural voice feedback for the executed command
fn create_voice_feedback(intent: &Intent, entities: &EntityExtraction) -> String {
    match intent {
        // Motion commands
        Intent::MoveForward => {
            if let Some(speed) = entities.speed {
                format!("Moving forward at speed {:.1}", speed)
            } else {
                "Moving forward".to_string()
            }
        }
        Intent::MoveBackward => "Moving backward".to_string(),
        Intent::MoveLeft => "Moving left".to_string(),
        Intent::MoveRight => "Moving right".to_string(),
        Intent::TurnLeft => "Turning left".to_string(),
        Intent::TurnRight => "Turning right".to_string(),
        Intent::Stop => "Stopping".to_string(),

        // Arm commands
        Intent::MoveArmUp => "Raising arm".to_string(),
        Intent::MoveArmDown => "Lowering arm".to_string(),
        Intent::MoveArmLeft => "Moving arm left".to_string(),
        Intent::MoveArmRight => "Moving arm right".to_string(),
        Intent::MoveArmForward => "Extending arm".to_string(),
        Intent::MoveArmBackward => "Retracting arm".to_string(),
        Intent::OpenGripper => "Opening gripper".to_string(),
        Intent::CloseGripper => "Closing gripper".to_string(),

        // Vision commands
        Intent::TrackObject => {
            if let Some(obj) = &entities.object_name {
                format!("Tracking {}", obj)
            } else {
                "Tracking object".to_string()
            }
        }
        Intent::FollowObject => {
            if let Some(obj) = &entities.object_name {
                format!("Following {}", obj)
            } else {
                "Following object".to_string()
            }
        }
        Intent::StopTracking => "Stopped tracking".to_string(),
        Intent::StopFollowing => "Stopped following".to_string(),

        // Camera control
        Intent::StartCamera => "Camera started".to_string(),
        Intent::StopCamera => "Camera stopped".to_string(),

        // Audio control
        Intent::StartAudio => "Microphone started".to_string(),
        Intent::StopAudio => "Microphone stopped".to_string(),

        // Unknown
        Intent::Unknown => "Command not recognized".to_string(),
    }
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    tracing::info!("Starting command_parser node");

    let parser = CommandParser::new();
    let (mut node, mut events) = DoraNode::init_from_env()?;

    tracing::info!(
        "Command parser initialized: {} Aho-Corasick keywords, {} regex patterns",
        parser.keyword_intents.len(),
        parser.regex_patterns.len()
    );

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } => match id.as_str() {
                "text" => {
                    // Extract SpeechTranscription from Arrow BinaryArray
                    let transcription: SpeechTranscription = if let Some(array) =
                        data.0.as_any().downcast_ref::<arrow::array::BinaryArray>()
                    {
                        if array.len() > 0 {
                            let bytes = array.value(0);
                            match serde_json::from_slice(bytes) {
                                Ok(t) => t,
                                Err(e) => {
                                    tracing::error!("Failed to deserialize SpeechTranscription: {}", e);
                                    continue;
                                }
                            }
                        } else {
                            continue;
                        }
                    } else {
                        tracing::warn!("Unexpected data format for text input");
                        continue;
                    };

                    let text = transcription.text.clone();
                    tracing::info!(
                        "Received transcription: '{}' (confidence: {:.2})",
                        text,
                        transcription.confidence
                    );

                    // Parse the command
                    let parsed = parser.parse(&text)?;

                    if parsed.confidence < parser.confidence_threshold {
                        tracing::warn!(
                            "Low confidence parse ({:.2}): {:?}",
                            parsed.confidence,
                            parsed.intent
                        );
                        continue;
                    }

                    tracing::info!(
                        "Parsed intent: {:?} (confidence: {:.2})",
                        parsed.intent,
                        parsed.confidence
                    );

                    // Convert to appropriate commands and send
                    if let Some(rover_cmd) = convert_to_rover_command(&parsed) {
                        tracing::info!("Sending rover command");
                        let serialized = serde_json::to_vec(&rover_cmd)?;
                        let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                        node.send_output(
                            DataId::from("rover_command".to_owned()),
                            Default::default(),
                            arrow_data,
                        )?;
                    }

                    if let Some(tracking_cmd) = convert_to_tracking_command(&parsed) {
                        tracing::info!("Sending tracking command: {:?}", tracking_cmd);
                        let serialized = serde_json::to_vec(&tracking_cmd)?;
                        let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                        node.send_output(
                            DataId::from("tracking_command".to_owned()),
                            Default::default(),
                            arrow_data,
                        )?;
                    }

                    if let Some(camera_cmd) = convert_to_camera_control(&parsed) {
                        tracing::info!("Sending camera control: {:?}", camera_cmd.command);
                        let serialized = serde_json::to_vec(&camera_cmd)?;
                        let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                        node.send_output(
                            DataId::from("camera_control".to_owned()),
                            Default::default(),
                            arrow_data,
                        )?;
                    }

                    // Send text feedback to web bridge
                    let feedback = format!("Executed: {:?}", parsed.intent);
                    let arrow_data = StringArray::from(vec![feedback.as_str()]);
                    node.send_output(
                        DataId::from("feedback".to_owned()),
                        Default::default(),
                        arrow_data,
                    )?;

                    // Send voice feedback via TTS
                    let tts_text = create_voice_feedback(&parsed.intent, &parsed.entities);
                    let tts_command = TtsCommand {
                        text: tts_text,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                        priority: TtsPriority::Normal,
                    };

                    tracing::info!("Sending TTS feedback: '{}'", tts_command.text);
                    let serialized = serde_json::to_vec(&tts_command)?;
                    let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                    node.send_output(
                        DataId::from("tts_command".to_owned()),
                        Default::default(),
                        arrow_data,
                    )?;
                }
                _ => {
                    tracing::warn!("Unexpected input: {}", id);
                }
            },
            Event::Stop(_) => {
                tracing::info!("Received stop event");
                break;
            }
            _ => {}
        }
    }

    tracing::info!("Command parser node stopped");
    Ok(())
}
