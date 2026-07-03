use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array},
    MetadataParameters, Parameter,
};
use eyre::{eyre, Result};
use robo_rover_lib::{AudioFrameMetadata, PcmSampleFormat, SttSourceKind};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    pub stream_id: Uuid,
    pub source_kind: SttSourceKind,
    pub entity_id: Option<String>,
    pub target_entity_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioInput {
    pub identity: SourceIdentity,
    pub frame_id: u64,
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

pub fn parse_rover(parameters: &MetadataParameters, data: &dyn Array) -> Result<AudioInput> {
    let array = data
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| eyre!("rover audio must be a BinaryArray"))?;
    if array.len() != 1 || array.is_null(0) {
        return Err(eyre!("rover audio must contain one non-null payload"));
    }
    let metadata = parse_metadata(parameters)?;
    if metadata.format != PcmSampleFormat::S16Le
        || metadata.sample_rate != 16_000
        || metadata.channels != 1
    {
        return Err(eyre!("rover audio must be mono 16 kHz s16le"));
    }
    let payload = array.value(0);
    metadata
        .validate_payload_len(payload.len())
        .map_err(eyre::Report::msg)?;
    if payload.len() % 2 != 0 {
        return Err(eyre!("rover s16le payload has an odd byte count"));
    }
    let entity_id = nonempty_string(parameters, "entity_id")?.to_owned();
    let samples = payload
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
        .collect();
    Ok(AudioInput {
        identity: SourceIdentity {
            stream_id: metadata.stream_id,
            source_kind: SttSourceKind::Rover,
            entity_id: Some(entity_id.clone()),
            target_entity_id: entity_id,
        },
        frame_id: metadata.frame_id,
        sample_rate: metadata.sample_rate,
        samples,
    })
}

pub fn parse_browser(parameters: &MetadataParameters, data: &dyn Array) -> Result<AudioInput> {
    let array = data
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| eyre!("browser audio must be a Float32Array"))?;
    let metadata = parse_metadata(parameters)?;
    if metadata.format != PcmSampleFormat::F32Le || metadata.channels != 1 {
        return Err(eyre!("browser audio must be mono f32le"));
    }
    metadata
        .validate_payload_len(array.len() * std::mem::size_of::<f32>())
        .map_err(eyre::Report::msg)?;
    if array.null_count() != 0 || array.values().iter().any(|sample| !sample.is_finite()) {
        return Err(eyre!("browser audio contains null or non-finite samples"));
    }
    Ok(AudioInput {
        identity: SourceIdentity {
            stream_id: metadata.stream_id,
            source_kind: SttSourceKind::Browser,
            entity_id: None,
            target_entity_id: nonempty_string(parameters, "target_entity_id")?.to_owned(),
        },
        frame_id: metadata.frame_id,
        sample_rate: metadata.sample_rate,
        samples: array.values().to_vec(),
    })
}

fn parse_metadata(parameters: &MetadataParameters) -> Result<AudioFrameMetadata> {
    Ok(AudioFrameMetadata {
        stream_id: Uuid::parse_str(string(parameters, "stream_id")?)?,
        frame_id: integer(parameters, "frame_id")?,
        capture_timestamp_ms: integer(parameters, "capture_timestamp_ms")?,
        sample_rate: integer(parameters, "sample_rate")?.try_into()?,
        channels: integer(parameters, "channels")?.try_into()?,
        sample_count: integer(parameters, "sample_count")?.try_into()?,
        format: PcmSampleFormat::from_metadata_name(string(parameters, "format")?)
            .map_err(eyre::Report::msg)?,
    })
}

fn integer(parameters: &MetadataParameters, key: &str) -> Result<u64> {
    match parameters.get(key) {
        Some(Parameter::Integer(value)) => Ok((*value).try_into()?),
        _ => Err(eyre!("missing or invalid audio metadata: {key}")),
    }
}

fn string<'a>(parameters: &'a MetadataParameters, key: &str) -> Result<&'a str> {
    match parameters.get(key) {
        Some(Parameter::String(value)) => Ok(value),
        _ => Err(eyre!("missing or invalid audio metadata: {key}")),
    }
}

fn nonempty_string<'a>(parameters: &'a MetadataParameters, key: &str) -> Result<&'a str> {
    let value = string(parameters, key)?;
    if value.trim().is_empty() {
        Err(eyre!("audio metadata must not be empty: {key}"))
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
