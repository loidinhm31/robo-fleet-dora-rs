use crate::config::EncoderConfig;
use dora_node_api::{Metadata, Parameter};

pub(crate) fn dimensions(metadata: &Metadata, defaults: EncoderConfig) -> (u32, u32) {
    let width = integer(metadata, "width")
        .map(|value| value as u32)
        .unwrap_or(defaults.width);
    let height = integer(metadata, "height")
        .map(|value| value as u32)
        .unwrap_or(defaults.height);
    (width, height)
}

pub(crate) fn capture_identity(metadata: &Metadata) -> Option<(u64, u64)> {
    let frame_id = u64::try_from(integer(metadata, "frame_id")?).ok()?;
    let capture_timestamp_ms = u64::try_from(integer(metadata, "capture_timestamp_ms")?).ok()?;
    Some((frame_id, capture_timestamp_ms))
}

fn integer(metadata: &Metadata, key: &str) -> Option<i64> {
    match metadata.parameters.get(key) {
        Some(Parameter::Integer(value)) => Some(*value),
        _ => None,
    }
}
