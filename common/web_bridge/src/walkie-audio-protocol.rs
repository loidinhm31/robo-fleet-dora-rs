use std::time::Duration;

use dora_node_api::{MetadataParameters, Parameter};
use robo_rover_lib::{AudioFrameMetadata, PcmSampleFormat};
use serde::Deserialize;
use uuid::Uuid;

const PROTOCOL_VERSION: u8 = 1;
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_WALKIE_BYTES: usize = 64 * 1024;
const FRAME_DURATION_MS: u64 = 20;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WalkieAudioFrameMetadata {
    pub(super) protocol_version: u8,
    pub(super) stream_id: String,
    pub(super) frame_id: u64,
    pub(super) capture_timestamp_ms: u64,
    pub(super) sample_rate: u32,
    pub(super) channels: u16,
    pub(super) sample_count: u32,
    pub(super) format: String,
}

pub(super) fn decode_walkie_frame(
    wire: WalkieAudioFrameMetadata,
    mut attachments: Vec<Vec<u8>>,
) -> Result<(AudioFrameMetadata, Vec<f32>, Duration), String> {
    if attachments.len() != 1 {
        return Err("walkie frame requires exactly one binary attachment".into());
    }
    let payload = attachments.pop().expect("attachment count checked");
    if payload.is_empty() || payload.len() > MAX_WALKIE_BYTES {
        return Err("walkie binary payload length is out of bounds".into());
    }
    let metadata = validate_metadata(wire, payload.len())?;
    let samples = decode_f32le(&payload)?;
    let duration = frame_duration(metadata);
    Ok((metadata, samples, duration))
}

pub(super) fn metadata_parameters(metadata: AudioFrameMetadata) -> MetadataParameters {
    MetadataParameters::from([
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
        (
            "format".into(),
            Parameter::String(metadata.format.metadata_name().into()),
        ),
        ("priority".into(), Parameter::String("high".into())),
    ])
}

fn validate_metadata(
    wire: WalkieAudioFrameMetadata,
    payload_len: usize,
) -> Result<AudioFrameMetadata, String> {
    if wire.protocol_version != PROTOCOL_VERSION {
        return Err("unsupported walkie protocol version".into());
    }
    if wire.frame_id > MAX_SAFE_INTEGER || wire.capture_timestamp_ms > MAX_SAFE_INTEGER {
        return Err("walkie identity exceeds JavaScript safe integer range".into());
    }
    if wire.channels != 1 || wire.format != "f32le" {
        return Err("walkie frames must be mono f32le".into());
    }
    let expected_samples = (u64::from(wire.sample_rate) * FRAME_DURATION_MS + 500) / 1_000;
    if u64::from(wire.sample_count) != expected_samples {
        return Err("walkie frame must contain exactly 20 ms of samples".into());
    }
    let metadata = AudioFrameMetadata {
        stream_id: Uuid::parse_str(&wire.stream_id)
            .map_err(|_| "walkie stream_id must be a UUID".to_string())?,
        frame_id: wire.frame_id,
        capture_timestamp_ms: wire.capture_timestamp_ms,
        sample_rate: wire.sample_rate,
        channels: wire.channels,
        sample_count: wire.sample_count,
        format: PcmSampleFormat::F32Le,
    };
    metadata.validate_payload_len(payload_len)?;
    Ok(metadata)
}

fn decode_f32le(payload: &[u8]) -> Result<Vec<f32>, String> {
    payload
        .chunks_exact(4)
        .map(|chunk| {
            let sample = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
            sample
                .is_finite()
                .then_some(sample)
                .ok_or_else(|| "walkie payload contains a non-finite sample".to_string())
        })
        .collect()
}

pub(super) fn frame_duration(metadata: AudioFrameMetadata) -> Duration {
    let scalar_rate = u64::from(metadata.sample_rate) * u64::from(metadata.channels);
    let nanos = (u64::from(metadata.sample_count) * 1_000_000_000 + scalar_rate / 2) / scalar_rate;
    Duration::from_nanos(nanos)
}
