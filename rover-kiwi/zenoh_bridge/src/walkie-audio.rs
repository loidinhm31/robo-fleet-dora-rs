use std::collections::BTreeMap;

use dora_node_api::Parameter;
use robo_rover_lib::{
    AudioFrameMetadata, AudioFrameSequenceTracker, PcmFramePacket, PcmSampleFormat,
};

const MAX_WALKIE_BYTES: usize = 64 * 1024;

pub struct DecodedWalkieFrame {
    pub parameters: BTreeMap<String, Parameter>,
    pub samples: Vec<f32>,
    pub missing_frames: u64,
    pub missing_samples: u64,
}

pub struct WalkieDecoder {
    sequence: AudioFrameSequenceTracker,
}

impl WalkieDecoder {
    pub fn new() -> Self {
        Self {
            sequence: AudioFrameSequenceTracker::default(),
        }
    }

    pub fn decode(&mut self, payload: &[u8]) -> Result<DecodedWalkieFrame, String> {
        let packet = PcmFramePacket::decode(payload)?;
        if packet.metadata.format != PcmSampleFormat::F32Le {
            return Err("walkie packet must contain f32le samples".into());
        }
        let metadata = packet.metadata;
        let samples = decode_f32le(packet.payload)?;
        let observation = self.sequence.observe(metadata)?;
        Ok(DecodedWalkieFrame {
            parameters: parameters(metadata),
            samples,
            missing_frames: observation.missing_frames,
            missing_samples: observation
                .missing_frames
                .saturating_mul(u64::from(metadata.sample_count)),
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
    payload
        .chunks_exact(4)
        .map(|chunk| {
            let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            sample
                .is_finite()
                .then_some(sample)
                .ok_or_else(|| "walkie payload contains a non-finite sample".to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn decodes_versioned_walkie_packet_and_restores_dora_metadata() {
        for sample_rate in [16_000_u32, 44_100, 48_000] {
            let payload: Vec<u8> = [0.25_f32, -0.5, 0.75]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            let metadata = AudioFrameMetadata {
                stream_id: Uuid::nil(),
                frame_id: 7,
                capture_timestamp_ms: 1,
                sample_rate,
                channels: 1,
                sample_count: 3,
                format: PcmSampleFormat::F32Le,
            };
            let packet = PcmFramePacket {
                metadata,
                payload: &payload,
            }
            .encode()
            .unwrap();

            let decoded = WalkieDecoder::new().decode(&packet).unwrap();

            assert_eq!(decoded.samples, vec![0.25, -0.5, 0.75]);
            assert_eq!(
                decoded.parameters["source_kind"],
                Parameter::String("walkie".into())
            );
            assert_eq!(decoded.parameters["frame_id"], Parameter::Integer(7));
            assert_eq!(
                decoded.parameters["sample_rate"],
                Parameter::Integer(i64::from(sample_rate))
            );
        }
    }

    #[test]
    fn rejects_legacy_raw_frames() {
        assert!(WalkieDecoder::new()
            .decode(&0.25_f32.to_le_bytes())
            .is_err());
    }

    #[test]
    fn rejects_duplicate_versioned_frame() {
        let payload = 0.25_f32.to_le_bytes();
        let metadata = AudioFrameMetadata {
            stream_id: Uuid::nil(),
            frame_id: 7,
            capture_timestamp_ms: 1,
            sample_rate: 16_000,
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

    #[test]
    fn reports_missing_frames_and_samples() {
        let payload = 0.25_f32.to_le_bytes();
        let mut decoder = WalkieDecoder::new();
        for frame_id in [0, 3] {
            let metadata = AudioFrameMetadata {
                stream_id: Uuid::nil(),
                frame_id,
                capture_timestamp_ms: 1,
                sample_rate: 16_000,
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
            let frame = decoder.decode(&packet).unwrap();
            if frame_id == 3 {
                assert_eq!(frame.missing_frames, 2);
                assert_eq!(frame.missing_samples, 2);
            }
        }
    }
}
