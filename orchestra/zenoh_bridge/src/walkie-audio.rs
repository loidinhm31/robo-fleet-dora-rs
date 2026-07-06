use std::collections::BTreeMap;

use dora_node_api::Parameter;
use robo_rover_lib::{AudioFrameMetadata, PcmFramePacket, PcmSampleFormat};
use uuid::Uuid;

const MAX_WALKIE_BYTES: usize = 64 * 1024;

pub fn encode_walkie_packet(
    parameters: &BTreeMap<String, Parameter>,
    samples: &[f32],
) -> Result<Vec<u8>, String> {
    if samples.len().saturating_mul(std::mem::size_of::<f32>()) > MAX_WALKIE_BYTES {
        return Err("walkie frame exceeds transport limit".into());
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("walkie frame contains a non-finite sample".into());
    }
    if string(parameters, "source_kind")? != "walkie" || string(parameters, "priority")? != "high" {
        return Err("invalid walkie source or priority".into());
    }
    let metadata = AudioFrameMetadata {
        stream_id: Uuid::parse_str(string(parameters, "stream_id")?)
            .map_err(|error| error.to_string())?,
        frame_id: integer(parameters, "frame_id")?,
        capture_timestamp_ms: integer(parameters, "capture_timestamp_ms")?,
        sample_rate: integer(parameters, "sample_rate")?
            .try_into()
            .map_err(|_| "walkie sample rate exceeds u32")?,
        channels: integer(parameters, "channels")?
            .try_into()
            .map_err(|_| "walkie channels exceed u16")?,
        sample_count: integer(parameters, "sample_count")?
            .try_into()
            .map_err(|_| "walkie sample count exceeds u32")?,
        format: PcmSampleFormat::from_metadata_name(string(parameters, "format")?)?,
    };
    if metadata.format != PcmSampleFormat::F32Le {
        return Err("walkie transport requires f32le samples".into());
    }
    let payload = encode_f32le(samples);
    PcmFramePacket {
        metadata,
        payload: &payload,
    }
    .encode()
}

fn integer(parameters: &BTreeMap<String, Parameter>, key: &str) -> Result<u64, String> {
    parameters
        .get(key)
        .and_then(|value| match value {
            Parameter::Integer(value) => u64::try_from(*value).ok(),
            _ => None,
        })
        .ok_or_else(|| format!("missing or invalid walkie metadata: {key}"))
}

fn string<'a>(parameters: &'a BTreeMap<String, Parameter>, key: &str) -> Result<&'a str, String> {
    parameters
        .get(key)
        .and_then(|value| match value {
            Parameter::String(value) if !value.is_empty() => Some(value.as_str()),
            _ => None,
        })
        .ok_or_else(|| format!("missing or invalid walkie metadata: {key}"))
}

fn encode_f32le(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(sample_rate: i64, frame_id: i64, sample_count: i64) -> BTreeMap<String, Parameter> {
        BTreeMap::from([
            ("source_kind".into(), Parameter::String("walkie".into())),
            ("priority".into(), Parameter::String("high".into())),
            (
                "stream_id".into(),
                Parameter::String(Uuid::nil().to_string()),
            ),
            ("frame_id".into(), Parameter::Integer(frame_id)),
            ("capture_timestamp_ms".into(), Parameter::Integer(1)),
            ("sample_rate".into(), Parameter::Integer(sample_rate)),
            ("channels".into(), Parameter::Integer(1)),
            ("sample_count".into(), Parameter::Integer(sample_count)),
            ("format".into(), Parameter::String("f32le".into())),
        ])
    }

    #[test]
    fn preserves_walkie_pcm_envelope_for_zenoh() {
        for sample_rate in [16_000_i64, 44_100, 48_000] {
            let packet = encode_walkie_packet(&parameters(sample_rate, 7, 3), &[0.25, -0.5, 0.75])
                .unwrap();
            let decoded = PcmFramePacket::decode(&packet).unwrap();
            assert_eq!(decoded.metadata.sample_rate, sample_rate as u32);
            assert_eq!(decoded.metadata.frame_id, 7);
            assert_eq!(decoded.metadata.sample_count, 3);
            assert_eq!(decoded.payload.len(), 12);
        }
    }

    #[test]
    fn rejects_payload_metadata_mismatch() {
        assert!(encode_walkie_packet(&parameters(16_000, 1, 2), &[0.25]).is_err());
        assert!(encode_walkie_packet(&parameters(16_000, 1, 16_385), &vec![0.0; 16_385]).is_err());
    }
}
