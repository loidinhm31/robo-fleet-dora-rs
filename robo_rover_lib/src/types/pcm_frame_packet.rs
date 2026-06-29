use super::{AudioFrameMetadata, PcmSampleFormat};
use uuid::Uuid;

const PCM_FRAME_MAGIC: &[u8; 4] = b"PCMF";
const PCM_FRAME_VERSION: u8 = 1;
const PCM_FRAME_HEADER_LEN: usize = 52;

/// Borrowed, versioned PCM envelope for Zenoh transport.
pub struct PcmFramePacket<'a> {
    pub metadata: AudioFrameMetadata,
    pub payload: &'a [u8],
}

impl<'a> PcmFramePacket<'a> {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.metadata.validate_payload_len(self.payload.len())?;
        let payload_len =
            u32::try_from(self.payload.len()).map_err(|_| "PCM payload exceeds u32".to_string())?;
        let mut packet = Vec::with_capacity(PCM_FRAME_HEADER_LEN + self.payload.len());
        packet.extend_from_slice(PCM_FRAME_MAGIC);
        packet.push(PCM_FRAME_VERSION);
        packet.push(self.metadata.format as u8);
        packet.extend_from_slice(&self.metadata.channels.to_le_bytes());
        packet.extend_from_slice(self.metadata.stream_id.as_bytes());
        packet.extend_from_slice(&self.metadata.frame_id.to_le_bytes());
        packet.extend_from_slice(&self.metadata.capture_timestamp_ms.to_le_bytes());
        packet.extend_from_slice(&self.metadata.sample_rate.to_le_bytes());
        packet.extend_from_slice(&self.metadata.sample_count.to_le_bytes());
        packet.extend_from_slice(&payload_len.to_le_bytes());
        packet.extend_from_slice(self.payload);
        Ok(packet)
    }

    pub fn decode(packet: &'a [u8]) -> Result<Self, String> {
        if packet.len() < PCM_FRAME_HEADER_LEN {
            return Err("PCM frame packet is truncated".into());
        }
        if &packet[..4] != PCM_FRAME_MAGIC {
            return Err("unsupported PCM frame magic".into());
        }
        if packet[4] != PCM_FRAME_VERSION {
            return Err(format!("unsupported PCM frame version: {}", packet[4]));
        }

        let metadata = AudioFrameMetadata {
            format: PcmSampleFormat::try_from(packet[5])?,
            channels: read_u16(packet, 6)?,
            stream_id: Uuid::from_slice(
                packet
                    .get(8..24)
                    .ok_or_else(|| "truncated PCM stream ID".to_string())?,
            )
            .map_err(|error| format!("invalid PCM stream ID: {error}"))?,
            frame_id: read_u64(packet, 24)?,
            capture_timestamp_ms: read_u64(packet, 32)?,
            sample_rate: read_u32(packet, 40)?,
            sample_count: read_u32(packet, 44)?,
        };
        let payload_len = read_u32(packet, 48)? as usize;
        let expected_packet_len = PCM_FRAME_HEADER_LEN
            .checked_add(payload_len)
            .ok_or_else(|| "PCM frame packet length overflow".to_string())?;
        if packet.len() != expected_packet_len {
            return Err("PCM frame packet payload length mismatch".into());
        }
        metadata.validate_payload_len(payload_len)?;
        Ok(Self {
            metadata,
            payload: &packet[PCM_FRAME_HEADER_LEN..],
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSequenceObservation {
    pub stream_changed: bool,
    pub missing_frames: u64,
}

#[derive(Default)]
pub struct AudioFrameSequenceTracker {
    stream_id: Option<Uuid>,
    last_frame_id: Option<u64>,
}

impl AudioFrameSequenceTracker {
    pub fn observe(
        &mut self,
        metadata: AudioFrameMetadata,
    ) -> Result<AudioSequenceObservation, String> {
        let stream_changed = self.stream_id != Some(metadata.stream_id);
        if stream_changed {
            self.stream_id = Some(metadata.stream_id);
            self.last_frame_id = Some(metadata.frame_id);
            return Ok(AudioSequenceObservation {
                stream_changed: true,
                missing_frames: 0,
            });
        }

        let missing_frames = match self.last_frame_id {
            None => 0,
            Some(previous) if metadata.frame_id > previous => metadata.frame_id - previous - 1,
            Some(_) => return Err("duplicate or regressed audio frame ID".into()),
        };
        self.last_frame_id = Some(metadata.frame_id);
        Ok(AudioSequenceObservation {
            stream_changed: false,
            missing_frames,
        })
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    bytes
        .get(offset..offset + 2)
        .and_then(|value| value.try_into().ok())
        .map(u16::from_le_bytes)
        .ok_or_else(|| "truncated u16 field".into())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| "truncated u32 field".into())
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    bytes
        .get(offset..offset + 8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| "truncated u64 field".into())
}
