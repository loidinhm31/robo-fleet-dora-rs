use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use dora_node_api::Parameter;
use robo_rover_lib::{
    AudioFrameMetadata, AudioFrameSequenceTracker, PcmFramePacket, PcmSampleFormat,
};
use uuid::Uuid;

const WALKIE_SAMPLE_RATE: u32 = 16_000;
const MAX_WALKIE_BYTES: usize = 64 * 1024;

pub struct DecodedWalkieFrame {
    pub parameters: BTreeMap<String, Parameter>,
    pub samples: Vec<f32>,
}

pub struct WalkieDecoder {
    legacy_stream_id: Uuid,
    next_legacy_frame_id: u64,
    sequence: AudioFrameSequenceTracker,
}

impl WalkieDecoder {
    pub fn new() -> Self {
        Self {
            legacy_stream_id: Uuid::new_v4(),
            next_legacy_frame_id: 0,
            sequence: AudioFrameSequenceTracker::default(),
        }
    }

    pub fn decode(&mut self, payload: &[u8]) -> Result<DecodedWalkieFrame, String> {
        let (metadata, samples) = if payload.starts_with(b"PCMF") {
            let packet = PcmFramePacket::decode(payload)?;
            if packet.metadata.format != PcmSampleFormat::F32Le {
                return Err("walkie packet must contain f32le samples".into());
            }
            (packet.metadata, decode_f32le(packet.payload)?)
        } else {
            let samples = decode_f32le(payload)?;
            let metadata = AudioFrameMetadata {
                stream_id: self.legacy_stream_id,
                frame_id: self.next_legacy_frame_id,
                capture_timestamp_ms: current_time_ms()?,
                sample_rate: WALKIE_SAMPLE_RATE,
                channels: 1,
                sample_count: samples
                    .len()
                    .try_into()
                    .map_err(|_| "walkie sample count exceeds u32")?,
                format: PcmSampleFormat::F32Le,
            };
            self.next_legacy_frame_id = self.next_legacy_frame_id.saturating_add(1);
            metadata.validate_payload_len(payload.len())?;
            (metadata, samples)
        };
        self.sequence.observe(metadata)?;
        Ok(DecodedWalkieFrame {
            parameters: parameters(metadata),
            samples,
        })
    }
}

fn parameters(metadata: AudioFrameMetadata) -> BTreeMap<String, Parameter> {
    BTreeMap::from([
        ("source_kind".into(), Parameter::String("walkie".into())),
        (
            "stream_id".into(),
            Parameter::String(metadata.stream_id.to_string()),
        ),
        (
            "frame_id".into(),
            Parameter::Integer(metadata.frame_id as i64),
        ),
        (
            "capture_timestamp_ms".into(),
            Parameter::Integer(metadata.capture_timestamp_ms as i64),
        ),
        (
            "sample_rate".into(),
            Parameter::Integer(i64::from(metadata.sample_rate)),
        ),
        (
            "channels".into(),
            Parameter::Integer(i64::from(metadata.channels)),
        ),
        (
            "sample_count".into(),
            Parameter::Integer(i64::from(metadata.sample_count)),
        ),
        ("format".into(), Parameter::String("f32le".into())),
        ("priority".into(), Parameter::String("high".into())),
    ])
}

fn decode_f32le(payload: &[u8]) -> Result<Vec<f32>, String> {
    if payload.is_empty() || payload.len() > MAX_WALKIE_BYTES || payload.len() % 4 != 0 {
        return Err(format!("invalid f32le payload length: {}", payload.len()));
    }
    Ok(payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn current_time_ms() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis()
        .try_into()
        .map_err(|_| "current timestamp exceeds u64".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_versioned_walkie_packet_and_restores_dora_metadata() {
        let payload: Vec<u8> = [0.25_f32, -0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let metadata = AudioFrameMetadata {
            stream_id: Uuid::nil(),
            frame_id: 7,
            capture_timestamp_ms: 1,
            sample_rate: WALKIE_SAMPLE_RATE,
            channels: 1,
            sample_count: 2,
            format: PcmSampleFormat::F32Le,
        };
        let packet = PcmFramePacket {
            metadata,
            payload: &payload,
        }
        .encode()
        .unwrap();

        let decoded = WalkieDecoder::new().decode(&packet).unwrap();

        assert_eq!(decoded.samples, vec![0.25, -0.5]);
        assert_eq!(
            decoded.parameters["source_kind"],
            Parameter::String("walkie".into())
        );
        assert_eq!(decoded.parameters["frame_id"], Parameter::Integer(7));
    }

    #[test]
    fn legacy_frames_receive_bounded_monotonic_metadata() {
        let mut decoder = WalkieDecoder::new();
        let payload = 0.25_f32.to_le_bytes();
        let first = decoder.decode(&payload).unwrap();
        let second = decoder.decode(&payload).unwrap();
        assert_eq!(first.parameters["frame_id"], Parameter::Integer(0));
        assert_eq!(second.parameters["frame_id"], Parameter::Integer(1));
        assert!(decoder.decode(&[0, 1, 2]).is_err());
    }

    #[test]
    fn rejects_duplicate_versioned_frame() {
        let payload = 0.25_f32.to_le_bytes();
        let metadata = AudioFrameMetadata {
            stream_id: Uuid::nil(),
            frame_id: 7,
            capture_timestamp_ms: 1,
            sample_rate: WALKIE_SAMPLE_RATE,
            channels: 1,
            sample_count: 1,
            format: PcmSampleFormat::F32Le,
        };
        let packet = PcmFramePacket {
            metadata,
            payload: &payload,
        }
        .encode()
        .unwrap();
        let mut decoder = WalkieDecoder::new();
        decoder.decode(&packet).unwrap();
        assert!(decoder.decode(&packet).is_err());
    }
}
