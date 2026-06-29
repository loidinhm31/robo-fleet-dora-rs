use serde::{Deserialize, Serialize};

pub use super::audio_types::{AudioCodec, AudioFrame, EncodedAudioFrame};

const RAW_FRAME_MAGIC: &[u8; 4] = b"RFRM";
const RAW_FRAME_VERSION: u8 = 1;
const RAW_FRAME_HEADER_LEN: usize = 36;
const MAX_RAW_FRAME_BYTES: usize = 64 * 1024 * 1024;
const JPEG_FRAME_MAGIC: &[u8; 4] = b"JPGF";
const JPEG_FRAME_VERSION: u8 = 1;
const JPEG_FRAME_HEADER_LEN: usize = 36;
const MAX_JPEG_FRAME_BYTES: usize = 8 * 1024 * 1024;
const MAX_JPEG_FRAME_DIMENSION: u32 = 4096;

/// Capture identity that must remain unchanged through the video pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoFrameMetadata {
    pub frame_id: u64,
    pub capture_timestamp_ms: u64,
    pub width: u32,
    pub height: u32,
}

/// Versioned envelope used to preserve capture metadata on the current raw Zenoh topic.
pub struct RawVideoFramePacket<'a> {
    pub metadata: VideoFrameMetadata,
    pub payload: &'a [u8],
}

impl<'a> RawVideoFramePacket<'a> {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        validate_raw_frame(self.metadata, self.payload.len())?;
        let payload_len = u32::try_from(self.payload.len())
            .map_err(|_| "raw frame payload exceeds u32".to_string())?;
        let mut packet = Vec::with_capacity(RAW_FRAME_HEADER_LEN + self.payload.len());
        packet.extend_from_slice(RAW_FRAME_MAGIC);
        packet.push(RAW_FRAME_VERSION);
        packet.extend_from_slice(&[0; 3]);
        packet.extend_from_slice(&self.metadata.frame_id.to_le_bytes());
        packet.extend_from_slice(&self.metadata.capture_timestamp_ms.to_le_bytes());
        packet.extend_from_slice(&self.metadata.width.to_le_bytes());
        packet.extend_from_slice(&self.metadata.height.to_le_bytes());
        packet.extend_from_slice(&payload_len.to_le_bytes());
        packet.extend_from_slice(self.payload);
        Ok(packet)
    }

    pub fn decode(packet: &'a [u8]) -> Result<Self, String> {
        if packet.len() < RAW_FRAME_HEADER_LEN {
            return Err("raw frame packet is truncated".into());
        }
        if &packet[..4] != RAW_FRAME_MAGIC || packet[4] != RAW_FRAME_VERSION {
            return Err("unsupported raw frame packet".into());
        }
        let frame_id = read_u64(packet, 8)?;
        let capture_timestamp_ms = read_u64(packet, 16)?;
        let width = read_u32(packet, 24)?;
        let height = read_u32(packet, 28)?;
        let payload_len = read_u32(packet, 32)? as usize;
        let expected_len = RAW_FRAME_HEADER_LEN
            .checked_add(payload_len)
            .ok_or_else(|| "raw frame packet length overflow".to_string())?;
        if packet.len() != expected_len {
            return Err("raw frame packet payload length mismatch".into());
        }
        let metadata = VideoFrameMetadata {
            frame_id,
            capture_timestamp_ms,
            width,
            height,
        };
        validate_raw_frame(metadata, payload_len)?;
        Ok(Self {
            metadata,
            payload: &packet[RAW_FRAME_HEADER_LEN..],
        })
    }
}

/// Versioned envelope for JPEG video frames transported across Zenoh.
pub struct JpegFramePacket<'a> {
    pub metadata: VideoFrameMetadata,
    pub payload: &'a [u8],
}

impl<'a> JpegFramePacket<'a> {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        validate_jpeg_frame(self.metadata, self.payload)?;
        let payload_len = u32::try_from(self.payload.len())
            .map_err(|_| "jpeg frame payload exceeds u32".to_string())?;
        let mut packet = Vec::with_capacity(JPEG_FRAME_HEADER_LEN + self.payload.len());
        packet.extend_from_slice(JPEG_FRAME_MAGIC);
        packet.push(JPEG_FRAME_VERSION);
        packet.extend_from_slice(&[0; 3]);
        packet.extend_from_slice(&self.metadata.frame_id.to_le_bytes());
        packet.extend_from_slice(&self.metadata.capture_timestamp_ms.to_le_bytes());
        packet.extend_from_slice(&self.metadata.width.to_le_bytes());
        packet.extend_from_slice(&self.metadata.height.to_le_bytes());
        packet.extend_from_slice(&payload_len.to_le_bytes());
        packet.extend_from_slice(self.payload);
        Ok(packet)
    }

    pub fn decode(packet: &'a [u8]) -> Result<Self, String> {
        if packet.len() < JPEG_FRAME_HEADER_LEN {
            return Err("jpeg frame packet is truncated".into());
        }
        if &packet[..4] != JPEG_FRAME_MAGIC {
            return Err("unsupported jpeg frame magic".into());
        }
        if packet[4] != JPEG_FRAME_VERSION {
            return Err("unsupported jpeg frame version".into());
        }
        let frame_id = read_u64(packet, 8)?;
        let capture_timestamp_ms = read_u64(packet, 16)?;
        let width = read_u32(packet, 24)?;
        let height = read_u32(packet, 28)?;
        let payload_len = read_u32(packet, 32)? as usize;
        let expected_len = JPEG_FRAME_HEADER_LEN
            .checked_add(payload_len)
            .ok_or_else(|| "jpeg frame packet length overflow".to_string())?;
        if packet.len() != expected_len {
            return Err("jpeg frame packet payload length mismatch".into());
        }
        let metadata = VideoFrameMetadata {
            frame_id,
            capture_timestamp_ms,
            width,
            height,
        };
        let payload = &packet[JPEG_FRAME_HEADER_LEN..];
        validate_jpeg_frame(metadata, payload)?;
        Ok(Self { metadata, payload })
    }
}

fn validate_raw_frame(metadata: VideoFrameMetadata, payload_len: usize) -> Result<(), String> {
    if metadata.width == 0 || metadata.height == 0 {
        return Err("raw frame dimensions must be non-zero".into());
    }
    let expected = (metadata.width as usize)
        .checked_mul(metadata.height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "raw frame dimensions overflow".to_string())?;
    if expected > MAX_RAW_FRAME_BYTES || payload_len != expected {
        return Err(format!(
            "invalid raw RGB8 payload: expected {expected}, got {payload_len}"
        ));
    }
    Ok(())
}

fn validate_jpeg_frame(metadata: VideoFrameMetadata, payload: &[u8]) -> Result<(), String> {
    if metadata.width == 0 || metadata.height == 0 {
        return Err("jpeg frame dimensions must be non-zero".into());
    }
    if metadata.width > MAX_JPEG_FRAME_DIMENSION || metadata.height > MAX_JPEG_FRAME_DIMENSION {
        return Err("jpeg frame dimensions exceed maximum".into());
    }
    let payload_len = payload.len();
    if payload_len == 0 || payload_len > MAX_JPEG_FRAME_BYTES {
        return Err(format!("invalid jpeg payload size: {payload_len}"));
    }
    if payload_len < 4 || payload[..2] != [0xFF, 0xD8] || payload[payload_len - 2..] != [0xFF, 0xD9]
    {
        return Err("jpeg payload missing SOI/EOI markers".into());
    }
    Ok(())
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    bytes
        .get(offset..offset + 8)
        .and_then(|v| v.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| "truncated u64 field".into())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .and_then(|v| v.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| "truncated u32 field".into())
}

#[cfg(test)]
mod raw_frame_tests {
    use super::*;

    #[test]
    fn raw_frame_packet_round_trips_identity_and_payload() {
        let payload = vec![7; 2 * 3 * 3];
        let metadata = VideoFrameMetadata {
            frame_id: 42,
            capture_timestamp_ms: 1_700_000_000_123,
            width: 2,
            height: 3,
        };
        let encoded = RawVideoFramePacket {
            metadata,
            payload: &payload,
        }
        .encode()
        .unwrap();
        let decoded = RawVideoFramePacket::decode(&encoded).unwrap();
        assert_eq!(decoded.metadata, metadata);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn raw_frame_packet_rejects_bad_lengths_and_headers() {
        assert!(RawVideoFramePacket::decode(b"RFRM").is_err());
        let payload = vec![0; 3];
        let metadata = VideoFrameMetadata {
            frame_id: 1,
            capture_timestamp_ms: 2,
            width: 1,
            height: 1,
        };
        let mut encoded = RawVideoFramePacket {
            metadata,
            payload: &payload,
        }
        .encode()
        .unwrap();
        encoded[4] = 9;
        assert!(RawVideoFramePacket::decode(&encoded).is_err());
        assert!(RawVideoFramePacket {
            metadata,
            payload: &[]
        }
        .encode()
        .is_err());
    }
}

#[cfg(test)]
mod jpeg_frame_tests {
    use super::*;

    fn metadata() -> VideoFrameMetadata {
        VideoFrameMetadata {
            frame_id: 42,
            capture_timestamp_ms: 1_700_000_000_123,
            width: 640,
            height: 480,
        }
    }

    fn jpeg_payload() -> Vec<u8> {
        vec![0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9]
    }

    #[test]
    fn jpeg_frame_packet_round_trips_identity_and_payload() {
        let payload = jpeg_payload();
        let encoded = JpegFramePacket {
            metadata: metadata(),
            payload: &payload,
        }
        .encode()
        .unwrap();
        let decoded = JpegFramePacket::decode(&encoded).unwrap();
        assert_eq!(decoded.metadata, metadata());
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn jpeg_frame_packet_rejects_truncation() {
        assert!(JpegFramePacket::decode(b"JPGF").is_err());
        let payload = jpeg_payload();
        let mut encoded = JpegFramePacket {
            metadata: metadata(),
            payload: &payload,
        }
        .encode()
        .unwrap();
        encoded.pop();
        assert!(JpegFramePacket::decode(&encoded).is_err());
    }

    #[test]
    fn jpeg_frame_packet_rejects_bad_magic_and_version() {
        let payload = jpeg_payload();
        let mut encoded = JpegFramePacket {
            metadata: metadata(),
            payload: &payload,
        }
        .encode()
        .unwrap();
        encoded[0] = b'X';
        assert!(JpegFramePacket::decode(&encoded).is_err());

        let mut encoded = JpegFramePacket {
            metadata: metadata(),
            payload: &payload,
        }
        .encode()
        .unwrap();
        encoded[4] = 9;
        assert!(JpegFramePacket::decode(&encoded).is_err());
    }

    #[test]
    fn jpeg_frame_packet_rejects_invalid_dimensions() {
        let payload = jpeg_payload();
        assert!(JpegFramePacket {
            metadata: VideoFrameMetadata {
                width: 0,
                ..metadata()
            },
            payload: &payload,
        }
        .encode()
        .is_err());
        assert!(JpegFramePacket {
            metadata: VideoFrameMetadata {
                width: MAX_JPEG_FRAME_DIMENSION + 1,
                ..metadata()
            },
            payload: &payload,
        }
        .encode()
        .is_err());
    }

    #[test]
    fn jpeg_frame_packet_rejects_oversized_payload() {
        let oversized = vec![0xFF; MAX_JPEG_FRAME_BYTES + 1];
        assert!(JpegFramePacket {
            metadata: metadata(),
            payload: &oversized,
        }
        .encode()
        .is_err());
    }

    #[test]
    fn jpeg_frame_packet_rejects_empty_or_non_jpeg_payload() {
        assert!(JpegFramePacket {
            metadata: metadata(),
            payload: &[],
        }
        .encode()
        .is_err());
        assert!(JpegFramePacket {
            metadata: metadata(),
            payload: b"not jpeg",
        }
        .encode()
        .is_err());
    }
}

/// Raw camera frame with optional audio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>, // Source rover entity ID (for multi-rover support)
    pub timestamp: u64,
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub format: String, // "RGB8", "BGR8", "GRAY8", "YUV420P"
    pub data: Vec<u8>,  // Raw pixel data
}

/// H.264 encoded video frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H264Frame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>, // Source rover entity ID (for multi-rover support)
    pub timestamp: u64,
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub is_keyframe: bool,
    pub data: Vec<u8>, // H.264 NAL units
    pub pts: i64,      // Presentation timestamp
    pub dts: i64,      // Decoding timestamp
}

/// Processed video frame with H.264 or JPEG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedVideoFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>, // Source rover entity ID (for multi-rover support)
    pub timestamp: u64,
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub codec: VideoCodec,
    pub is_keyframe: bool,
    pub data: Vec<u8>, // Compressed video data
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
    Low,    // 320x240, H.264 @ 500kbps
    Medium, // 640x480, H.264 @ 1Mbps
    High,   // 1280x720, H.264 @ 2Mbps
    Ultra,  // 1920x1080, H.264 @ 4Mbps
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
