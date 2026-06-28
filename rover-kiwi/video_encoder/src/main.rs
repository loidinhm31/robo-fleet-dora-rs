mod config;
mod encoder;
mod frame_metadata;

use config::EncoderConfig;
use dora_node_api::{
    arrow::array::{Array, BinaryArray, UInt8Array},
    DoraNode, Event, Parameter,
};
use encoder::JpegEncoder;
use eyre::Result;
use robo_rover_lib::{capture_age_ms, init_tracing, FrameSequenceTracker, MetricWindow};
use std::time::{Duration, Instant};
use tracing::{debug, error, info};

fn main() -> Result<()> {
    let _guard = init_tracing();
    info!("Starting video_encoder node");

    let config = EncoderConfig::from_env();
    info!(
        "Encoder config: JPEG quality={}, default resolution={}x{}",
        config.jpeg_quality, config.width, config.height
    );
    let mut encoder = JpegEncoder::new(config.jpeg_quality)?;
    let (mut node, mut events) = DoraNode::init_from_env()?;

    let mut frames_encoded = 0u64;
    let mut encoding_errors = 0u64;
    let mut total_encoding_time_ms = 0u64;
    let mut encode_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut encode_age_metrics = MetricWindow::new(Duration::from_secs(5));
    let mut frame_sequence = FrameSequenceTracker::default();

    info!("video_encoder node ready, waiting for video frames...");

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } if id.as_str() == "video_frame" => {
                let start_time = Instant::now();
                let (width, height) = frame_metadata::dimensions(&metadata, config);
                let Some((frame_id, capture_timestamp_ms)) =
                    frame_metadata::capture_identity(&metadata)
                else {
                    encode_metrics.record_error();
                    error!("Video frame missing capture identity");
                    continue;
                };

                match frame_sequence.observe(frame_id) {
                    Ok(missing) => encode_metrics.record_drops(missing),
                    Err(()) => encode_metrics.record_error(),
                }
                let frame_age_ms = capture_age_ms(capture_timestamp_ms).unwrap_or_else(|| {
                    encode_metrics.record_error();
                    0
                });
                encode_age_metrics.record(Duration::from_millis(frame_age_ms), 0);

                let Some(rgb_array) = data.as_any().downcast_ref::<UInt8Array>() else {
                    error!("Invalid video frame data type (expected UInt8Array)");
                    encoding_errors += 1;
                    encode_metrics.record_error();
                    continue;
                };
                let rgb_bytes = rgb_array.values().as_ref();

                match encoder.encode(rgb_bytes, width, height) {
                    Ok(jpeg_data) => {
                        let encoding_time = start_time.elapsed();
                        total_encoding_time_ms += encoding_time.as_millis() as u64;
                        frames_encoded += 1;
                        encode_metrics.record(encoding_time, jpeg_data.len());

                        debug!(
                            "Frame {} encoded: {}x{} RGB ({} bytes) → JPEG ({} bytes, {:.1}x compression, {:.1}ms)",
                            frames_encoded,
                            width,
                            height,
                            rgb_bytes.len(),
                            jpeg_data.len(),
                            rgb_bytes.len() as f32 / jpeg_data.len() as f32,
                            encoding_time.as_secs_f32() * 1000.0
                        );

                        if let Some(snapshot) = encode_metrics.snapshot_if_due() {
                            info!(
                                metric = "video_pipeline",
                                stage = "jpeg_encode",
                                frame_id,
                                frame_age_ms,
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
                        if let Some(snapshot) = encode_age_metrics.snapshot_if_due() {
                            info!(
                                metric = "video_pipeline",
                                stage = "jpeg_encode_age",
                                count = snapshot.count,
                                p50_us = snapshot.p50_us,
                                p95_us = snapshot.p95_us,
                                p99_us = snapshot.p99_us,
                                max_us = snapshot.max_us
                            );
                        }

                        let mut output_metadata = metadata.clone();
                        output_metadata
                            .parameters
                            .insert("codec".to_string(), Parameter::String("jpeg".to_string()));
                        output_metadata.parameters.insert(
                            "quality".to_string(),
                            Parameter::Integer(i64::from(config.jpeg_quality)),
                        );
                        output_metadata.parameters.insert(
                            "compressed_size".to_string(),
                            Parameter::Integer(jpeg_data.len() as i64),
                        );

                        let binary_data = BinaryArray::from_vec(vec![jpeg_data.as_slice()]);
                        node.send_output(
                            "encoded_frame".to_owned().into(),
                            output_metadata.parameters,
                            binary_data,
                        )?;
                    }
                    Err(error) => {
                        encoding_errors += 1;
                        encode_metrics.record_error();
                        error!("Encoding error (frame {}): {}", frames_encoded + 1, error);
                    }
                }
            }
            Event::Input { id, .. } => debug!("Ignoring unexpected input: {}", id.as_str()),
            Event::Stop(_) => {
                info!("Received stop signal");
                break;
            }
            other => debug!("Ignoring event: {:?}", other),
        }
    }

    if frames_encoded > 0 {
        let avg_encoding_time = total_encoding_time_ms as f32 / frames_encoded as f32;
        info!(
            "video_encoder shutting down: {} frames encoded, avg {:.1}ms/frame, {} errors",
            frames_encoded, avg_encoding_time, encoding_errors
        );
    }
    Ok(())
}
