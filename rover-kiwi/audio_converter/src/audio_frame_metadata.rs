use dora_node_api::Parameter;
use eyre::Result;
use robo_rover_lib::{AudioFrameMetadata, PcmSampleFormat};
use std::collections::BTreeMap;
use uuid::Uuid;

pub fn parse(parameters: &BTreeMap<String, Parameter>) -> Result<AudioFrameMetadata> {
    let integer = |key: &str| -> Result<u64> {
        match parameters.get(key) {
            Some(Parameter::Integer(value)) => Ok(u64::try_from(*value)?),
            _ => Err(eyre::eyre!("missing or invalid audio metadata: {key}")),
        }
    };
    let string = |key: &str| -> Result<&str> {
        match parameters.get(key) {
            Some(Parameter::String(value)) => Ok(value),
            _ => Err(eyre::eyre!("missing or invalid audio metadata: {key}")),
        }
    };
    Ok(AudioFrameMetadata {
        stream_id: Uuid::parse_str(string("stream_id")?)?,
        frame_id: integer("frame_id")?,
        capture_timestamp_ms: integer("capture_timestamp_ms")?,
        sample_rate: integer("sample_rate")?.try_into()?,
        channels: integer("channels")?.try_into()?,
        sample_count: integer("sample_count")?.try_into()?,
        format: PcmSampleFormat::from_metadata_name(string("format")?)
            .map_err(eyre::Report::msg)?,
    })
}

pub fn to_parameters(
    metadata: AudioFrameMetadata,
    payload_len: usize,
) -> Result<BTreeMap<String, Parameter>> {
    metadata
        .validate_payload_len(payload_len)
        .map_err(eyre::Report::msg)?;
    Ok(BTreeMap::from([
        (
            "stream_id".into(),
            Parameter::String(metadata.stream_id.to_string()),
        ),
        (
            "frame_id".into(),
            Parameter::Integer(metadata.frame_id.try_into()?),
        ),
        (
            "capture_timestamp_ms".into(),
            Parameter::Integer(metadata.capture_timestamp_ms.try_into()?),
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
        ("size".into(), Parameter::Integer(payload_len.try_into()?)),
    ]))
}

pub fn env_number<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

pub fn validate_output_format() -> Result<()> {
    let format = std::env::var("OUTPUT_FORMAT").unwrap_or_else(|_| "s16le".into());
    if matches!(format.to_ascii_lowercase().as_str(), "int16" | "s16le") {
        Ok(())
    } else {
        Err(eyre::eyre!("OUTPUT_FORMAT must be int16 or s16le"))
    }
}
