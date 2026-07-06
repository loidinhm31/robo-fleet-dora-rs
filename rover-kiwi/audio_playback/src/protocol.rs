use std::collections::BTreeMap;

use dora_node_api::{
    arrow::array::{Array, Float32Array},
    MetadataParameters, Parameter,
};
use eyre::{eyre, Result};
use robo_rover_lib::{AudioFrameMetadata, PcmSampleFormat};
use uuid::Uuid;

const MAX_FRAME_SAMPLES: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioSource {
    Tts,
    Walkie,
}

impl AudioSource {
    fn metadata_name(self) -> &'static str {
        match self {
            Self::Tts => "tts",
            Self::Walkie => "walkie",
        }
    }
}

#[derive(Debug)]
pub struct SourceFrame {
    pub source: AudioSource,
    pub command_id: Option<String>,
    pub frame_id: u64,
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    pub normalized_samples: usize,
}

pub fn parse_source_frame(
    source: AudioSource,
    parameters: &MetadataParameters,
    data: &dyn Array,
) -> Result<SourceFrame> {
    let array = data
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| eyre!("audio payload must be Float32Array"))?;
    if array.null_count() != 0 || array.is_empty() || array.len() > MAX_FRAME_SAMPLES {
        return Err(eyre!("audio payload length or null count is invalid"));
    }

    let metadata = parse_metadata(parameters)?;
    metadata
        .validate_payload_len(array.len() * std::mem::size_of::<f32>())
        .map_err(eyre::Report::msg)?;
    if metadata.format != PcmSampleFormat::F32Le {
        return Err(eyre!("audio format must be f32le"));
    }
    if string(parameters, "source_kind")? != source.metadata_name() {
        return Err(eyre!("audio source metadata does not match input"));
    }
    validate_priority(source, parameters)?;

    let command_id = match source {
        AudioSource::Tts => {
            let value = string(parameters, "command_id")?;
            Uuid::parse_str(value).map_err(|_| eyre!("TTS command_id must be a UUID"))?;
            Some(value.to_owned())
        }
        AudioSource::Walkie => None,
    };
    let mut normalized_samples = 0;
    let samples = downmix_and_normalize(array.values(), metadata.channels, &mut normalized_samples);

    Ok(SourceFrame {
        source,
        command_id,
        frame_id: metadata.frame_id,
        sample_rate: metadata.sample_rate,
        samples,
        normalized_samples,
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

fn validate_priority(source: AudioSource, parameters: &MetadataParameters) -> Result<()> {
    let priority = string(parameters, "priority")?;
    let valid = match source {
        AudioSource::Tts => matches!(priority, "low" | "normal" | "high" | "emergency"),
        AudioSource::Walkie => priority == "high",
    };
    valid
        .then_some(())
        .ok_or_else(|| eyre!("invalid audio priority"))
}

fn downmix_and_normalize(input: &[f32], channels: u16, normalized: &mut usize) -> Vec<f32> {
    input
        .chunks_exact(channels as usize)
        .map(|frame| {
            let sum = frame
                .iter()
                .map(|sample| {
                    if sample.is_finite() {
                        sample.clamp(-1.0, 1.0)
                    } else {
                        *normalized += 1;
                        0.0
                    }
                })
                .sum::<f32>();
            (sum / channels as f32).clamp(-1.0, 1.0)
        })
        .collect()
}

fn integer(parameters: &BTreeMap<String, Parameter>, key: &str) -> Result<u64> {
    match parameters.get(key) {
        Some(Parameter::Integer(value)) => Ok(u64::try_from(*value)?),
        _ => Err(eyre!("missing or invalid audio metadata: {key}")),
    }
}

fn string<'a>(parameters: &'a BTreeMap<String, Parameter>, key: &str) -> Result<&'a str> {
    match parameters.get(key) {
        Some(Parameter::String(value)) if !value.is_empty() => Ok(value),
        _ => Err(eyre!("missing or invalid audio metadata: {key}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(source: &str, channels: i64, sample_count: i64) -> MetadataParameters {
        BTreeMap::from([
            ("source_kind".into(), Parameter::String(source.into())),
            (
                "command_id".into(),
                Parameter::String(Uuid::nil().to_string()),
            ),
            (
                "stream_id".into(),
                Parameter::String(Uuid::nil().to_string()),
            ),
            ("frame_id".into(), Parameter::Integer(1)),
            ("capture_timestamp_ms".into(), Parameter::Integer(1)),
            ("sample_rate".into(), Parameter::Integer(44_100)),
            ("channels".into(), Parameter::Integer(channels)),
            ("sample_count".into(), Parameter::Integer(sample_count)),
            ("format".into(), Parameter::String("f32le".into())),
            ("priority".into(), Parameter::String("normal".into())),
        ])
    }

    #[test]
    fn parses_tts_and_normalizes_non_finite_samples() {
        let mut params = metadata("tts", 1, 2);
        let frame = parse_source_frame(
            AudioSource::Tts,
            &params,
            &Float32Array::from(vec![f32::NAN, 0.5]),
        )
        .unwrap();

        assert_eq!(frame.samples, vec![0.0, 0.5]);
        assert_eq!(frame.normalized_samples, 1);
        params.insert("sample_count".into(), Parameter::Integer(3));
        assert!(parse_source_frame(
            AudioSource::Tts,
            &params,
            &Float32Array::from(vec![0.0, 0.5])
        )
        .is_err());
    }

    #[test]
    fn rejects_source_mismatch_and_invalid_walkie_priority() {
        let params = metadata("tts", 1, 1);
        assert!(
            parse_source_frame(AudioSource::Walkie, &params, &Float32Array::from(vec![0.1]))
                .is_err()
        );
    }
}
