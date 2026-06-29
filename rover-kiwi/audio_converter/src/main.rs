use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use robo_rover_lib::{
    capture_age_ms, init_tracing, AudioFrameMetadata, AudioFrameSequenceTracker, MetricWindow,
    PcmSampleFormat,
};
use std::time::{Duration, Instant};

mod audio_frame_metadata;
use audio_frame_metadata::{env_number, parse, to_parameters, validate_output_format};

fn main() -> Result<()> {
    let _guard = init_tracing();
    let expected_sample_rate = env_number("SAMPLE_RATE", 16_000u32);
    let expected_channels = env_number("CHANNELS", 1u16);
    validate_output_format()?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let output_id = DataId::from("audio_output".to_owned());
    let mut metrics = MetricWindow::new(Duration::from_secs(5));
    let mut frames_converted = 0u64;
    let mut samples_converted = 0u64;
    let mut conversion_errors = 0u64;
    let mut sequence_drops = 0u64;
    let mut sequence = AudioFrameSequenceTracker::default();

    tracing::info!(
        expected_sample_rate,
        expected_channels,
        "rover audio converter ready"
    );

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } if id.as_str() == "audio_input" => {
                let started = Instant::now();
                let result = (|| -> Result<(AudioFrameMetadata, Vec<u8>)> {
                    let input = data
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| eyre::eyre!("audio input must be Float32Array"))?;
                    let input_metadata = parse(&metadata.parameters)?;
                    if input_metadata.format != PcmSampleFormat::F32Le {
                        return Err(eyre::eyre!("audio converter requires f32le input"));
                    }
                    if input_metadata.sample_rate != expected_sample_rate
                        || input_metadata.channels != expected_channels
                    {
                        return Err(eyre::eyre!(
                            "audio dimensions differ from configured {} Hz/{} channels",
                            expected_sample_rate,
                            expected_channels
                        ));
                    }
                    input_metadata
                        .validate_payload_len(
                            input.len() * PcmSampleFormat::F32Le.bytes_per_sample(),
                        )
                        .map_err(eyre::Report::msg)?;
                    let missing = observe_input_sequence(&mut sequence, input_metadata)?;
                    sequence_drops = sequence_drops.saturating_add(missing);
                    metrics.record_drops(missing);
                    let output = float32_to_s16le(input.values().as_ref());
                    Ok((
                        AudioFrameMetadata {
                            format: PcmSampleFormat::S16Le,
                            ..input_metadata
                        },
                        output,
                    ))
                })();

                match result {
                    Ok((output_metadata, bytes)) => {
                        let params = to_parameters(output_metadata, bytes.len())?;
                        let byte_count = bytes.len();
                        let binary = BinaryArray::from_vec(vec![bytes.as_slice()]);
                        if let Err(error) = node.send_output(output_id.clone(), params, binary) {
                            conversion_errors = conversion_errors.saturating_add(1);
                            metrics.record_error();
                            tracing::error!(
                                %error,
                                stream_id = %output_metadata.stream_id,
                                frame_id = output_metadata.frame_id,
                                "failed to send converted audio frame"
                            );
                        } else {
                            frames_converted = frames_converted.saturating_add(1);
                            samples_converted = samples_converted
                                .saturating_add(u64::from(output_metadata.sample_count));
                            metrics.record(started.elapsed(), byte_count);
                        }

                        if let Some(snapshot) = metrics.snapshot_if_due() {
                            let age_ms = capture_age_ms(output_metadata.capture_timestamp_ms);
                            tracing::info!(
                                metric = "audio_pipeline",
                                stage = "rover_convert",
                                stream_id = %output_metadata.stream_id,
                                frame_id = output_metadata.frame_id,
                                frame_age_ms = age_ms,
                                count = snapshot.count,
                                bytes = snapshot.bytes,
                                drops = snapshot.drops,
                                errors = snapshot.errors,
                                p50_us = snapshot.p50_us,
                                p95_us = snapshot.p95_us,
                                p99_us = snapshot.p99_us,
                                max_us = snapshot.max_us
                            );
                        }
                    }
                    Err(error) => {
                        conversion_errors = conversion_errors.saturating_add(1);
                        metrics.record_error();
                        tracing::warn!(%error, "rejected invalid audio frame");
                    }
                }
            }
            Event::Input { id, .. } => tracing::warn!(input = %id, "ignoring unexpected input"),
            Event::Stop(_) => break,
            _ => {}
        }
    }

    tracing::info!(
        metric = "audio_pipeline_total",
        stage = "rover_convert",
        frames_converted,
        samples_converted,
        sequence_drops,
        conversion_errors,
        "audio converter stopped"
    );
    Ok(())
}

fn observe_input_sequence(
    sequence: &mut AudioFrameSequenceTracker,
    metadata: AudioFrameMetadata,
) -> Result<u64> {
    sequence
        .observe(metadata)
        .map(|observation| observation.missing_frames)
        .map_err(eyre::Report::msg)
}

fn float32_to_s16le(samples: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let scaled = if sample <= -1.0 {
            i16::MIN
        } else {
            (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
        };
        output.extend_from_slice(&scaled.to_le_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_float_extremes_to_explicit_s16le() {
        let bytes = float32_to_s16le(&[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]);
        let samples: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        assert_eq!(
            samples,
            vec![i16::MIN, i16::MIN, 0, 16_384, i16::MAX, i16::MAX]
        );
    }

    #[test]
    fn converter_sequence_reports_capture_to_converter_gap() {
        let mut sequence = AudioFrameSequenceTracker::default();
        let mut metadata = AudioFrameMetadata {
            stream_id: uuid::Uuid::from_u128(1),
            frame_id: 10,
            capture_timestamp_ms: 1,
            sample_rate: 16_000,
            channels: 1,
            sample_count: 800,
            format: PcmSampleFormat::F32Le,
        };
        assert_eq!(observe_input_sequence(&mut sequence, metadata).unwrap(), 0);
        metadata.frame_id = 12;
        assert_eq!(observe_input_sequence(&mut sequence, metadata).unwrap(), 1);
    }
}
