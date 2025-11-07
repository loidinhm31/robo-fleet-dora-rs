use serde::{Deserialize, Serialize};

/// Raw audio frame data from microphone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    pub timestamp: u64,
    pub frame_id: u64,
    pub sample_rate: u32,     // e.g., 48000 Hz
    pub channels: u16,        // 1 = mono, 2 = stereo
    pub bit_depth: u16,       // e.g., 16-bit
    pub format: String,       // "PCM_S16LE", "PCM_F32LE", etc.
    pub data: Vec<u8>,        // Raw PCM audio data
    pub sample_count: usize,  // Number of samples
}

impl AudioFrame {
    pub fn expected_size(&self) -> usize {
        self.sample_count * (self.bit_depth as usize / 8) * self.channels as usize
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = self.expected_size();
        if self.data.len() != expected {
            return Err(format!(
                "Audio data size mismatch: got {} bytes, expected {} bytes",
                self.data.len(),
                expected
            ));
        }
        Ok(())
    }
}

/// Encoded audio frame (Opus, AAC, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedAudioFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,  // Source rover entity ID (for multi-rover support)
    pub timestamp: u64,
    pub frame_id: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub codec: AudioCodec,
    pub data: Vec<u8>,        // Encoded audio data
    pub duration_ms: u32,     // Frame duration in milliseconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioCodec {
    Opus,
    Aac,
    Mp3,
    Pcm,
}

/// Raw camera frame with optional audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,  // Source rover entity ID (for multi-rover support)
    pub timestamp: u64,
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub format: String,  // "RGB8", "BGR8", "GRAY8", "YUV420P"
    pub data: Vec<u8>,   // Raw pixel data
}

/// H.264 encoded video frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H264Frame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,  // Source rover entity ID (for multi-rover support)
    pub timestamp: u64,
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub is_keyframe: bool,
    pub data: Vec<u8>,        // H.264 NAL units
    pub pts: i64,             // Presentation timestamp
    pub dts: i64,             // Decoding timestamp
}

/// Processed video frame with H.264 or JPEG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedVideoFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,  // Source rover entity ID (for multi-rover support)
    pub timestamp: u64,
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub codec: VideoCodec,
    pub is_keyframe: bool,
    pub data: Vec<u8>,        // Compressed video data
    pub overlay_data: Option<OverlayData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoCodec {
    H264,
    Jpeg,
    Vp8,
    Vp9,
}

/// Combined A/V stream packet for synchronized transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AVStreamPacket {
    pub timestamp: u64,
    pub packet_id: u64,
    pub video_frame: Option<ProcessedVideoFrame>,
    pub audio_frame: Option<EncodedAudioFrame>,
}

/// Telemetry overlay information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlayData {
    pub rover_position: Option<(f64, f64)>,
    pub rover_velocity: Option<f64>,
    pub arm_position: Option<[f64; 6]>,
    pub battery_level: Option<f64>,
    pub signal_strength: Option<u8>,
    pub timestamp_text: String,
}

/// Camera control commands for gst-camera node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraControl {
    pub command: CameraAction,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraAction {
    Start,
    Stop,
}

/// Audio control commands for audio-capture node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioControl {
    pub command: AudioAction,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioAction {
    Start,
    Stop,
}

/// Stream control commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamControl {
    pub command: StreamCommand,
    pub video_enabled: bool,
    pub audio_enabled: bool,
    pub quality: Option<StreamQuality>,
    pub target_fps: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamCommand {
    Start,
    Stop,
    Pause,
    Resume,
    Configure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamQuality {
    Low,      // 320x240, H.264 @ 500kbps
    Medium,   // 640x480, H.264 @ 1Mbps
    High,     // 1280x720, H.264 @ 2Mbps
    Ultra,    // 1920x1080, H.264 @ 4Mbps
}

impl StreamQuality {
    pub fn get_resolution(&self) -> (u32, u32) {
        match self {
            StreamQuality::Low => (320, 240),
            StreamQuality::Medium => (640, 480),
            StreamQuality::High => (1280, 720),
            StreamQuality::Ultra => (1920, 1080),
        }
    }

    pub fn get_bitrate_kbps(&self) -> u32 {
        match self {
            StreamQuality::Low => 500,
            StreamQuality::Medium => 1000,
            StreamQuality::High => 2000,
            StreamQuality::Ultra => 4000,
        }
    }
}

/// Streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStats {
    pub timestamp: u64,
    pub video_frames_processed: u64,
    pub video_frames_dropped: u64,
    pub audio_frames_processed: u64,
    pub audio_frames_dropped: u64,
    pub avg_video_size_kb: f64,
    pub avg_audio_size_kb: f64,
    pub current_video_fps: f64,
    pub video_bandwidth_kbps: f64,
    pub audio_bandwidth_kbps: f64,
    pub latency_ms: f64,
}