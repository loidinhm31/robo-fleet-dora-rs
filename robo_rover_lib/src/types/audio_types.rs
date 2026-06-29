use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 192_000;
const MAX_CHANNELS: u16 = 8;
const MAX_FRAME_DURATION_MS: u64 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PcmSampleFormat {
    F32Le = 1,
    S16Le = 2,
}

impl PcmSampleFormat {
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::F32Le => 4,
            Self::S16Le => 2,
        }
    }

    pub const fn metadata_name(self) -> &'static str {
        match self {
            Self::F32Le => "f32le",
            Self::S16Le => "s16le",
        }
    }

    pub fn from_metadata_name(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "f32le" => Ok(Self::F32Le),
            "s16le" => Ok(Self::S16Le),
            _ => Err(format!("unsupported PCM sample format: {value}")),
        }
    }
}

impl TryFrom<u8> for PcmSampleFormat {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::F32Le),
            2 => Ok(Self::S16Le),
            _ => Err(format!("unsupported PCM sample format id: {value}")),
        }
    }
}

/// Capture identity and dimensions shared by every audio transport stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFrameMetadata {
    pub stream_id: Uuid,
    pub frame_id: u64,
    pub capture_timestamp_ms: u64,
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved scalar samples, including all channels.
    pub sample_count: u32,
    pub format: PcmSampleFormat,
}

impl AudioFrameMetadata {
    pub fn expected_payload_len(self) -> Result<usize, String> {
        self.validate_dimensions()?;
        (self.sample_count as usize)
            .checked_mul(self.format.bytes_per_sample())
            .ok_or_else(|| "PCM payload length overflow".to_string())
    }

    pub fn validate_payload_len(self, payload_len: usize) -> Result<(), String> {
        let expected = self.expected_payload_len()?;
        if payload_len != expected {
            return Err(format!(
                "PCM payload length mismatch: expected {expected}, got {payload_len}"
            ));
        }
        Ok(())
    }

    fn validate_dimensions(self) -> Result<(), String> {
        if self.frame_id > i64::MAX as u64 || self.capture_timestamp_ms > i64::MAX as u64 {
            return Err("PCM identity exceeds Dora metadata range".into());
        }
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&self.sample_rate) {
            return Err(format!("invalid PCM sample rate: {}", self.sample_rate));
        }
        if self.channels == 0 || self.channels > MAX_CHANNELS {
            return Err(format!("invalid PCM channel count: {}", self.channels));
        }
        if self.sample_count == 0 || self.sample_count % u32::from(self.channels) != 0 {
            return Err("PCM sample count must contain complete interleaved frames".into());
        }
        let duration_numerator = u64::from(self.sample_count)
            .checked_mul(1_000)
            .ok_or_else(|| "PCM duration overflow".to_string())?;
        let scalar_rate = u64::from(self.sample_rate)
            .checked_mul(u64::from(self.channels))
            .ok_or_else(|| "PCM sample dimensions overflow".to_string())?;
        if duration_numerator > scalar_rate * MAX_FRAME_DURATION_MS {
            return Err("PCM frame duration exceeds maximum".into());
        }
        Ok(())
    }
}

/// Legacy raw audio frame retained for public API compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    pub timestamp: u64,
    pub frame_id: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u16,
    pub format: String,
    pub data: Vec<u8>,
    pub sample_count: usize,
}

impl AudioFrame {
    pub fn expected_size(&self) -> usize {
        self.sample_count * (self.bit_depth as usize / 8) * self.channels as usize
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = self.expected_size();
        if self.data.len() == expected {
            Ok(())
        } else {
            Err(format!(
                "Audio data size mismatch: got {} bytes, expected {} bytes",
                self.data.len(),
                expected
            ))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedAudioFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    pub timestamp: u64,
    pub frame_id: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub codec: AudioCodec,
    pub data: Vec<u8>,
    pub duration_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioCodec {
    Opus,
    Aac,
    Mp3,
    Pcm,
}
