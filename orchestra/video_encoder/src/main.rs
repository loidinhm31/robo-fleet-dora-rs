use dora_node_api::{
    arrow::array::{Array, UInt8Array},
    DoraNode, Event,
};
use eyre::{Result, eyre};
use image::{ImageBuffer, Rgb, codecs::jpeg::JpegEncoder};
use std::io::Cursor;
use std::env;
use tracing::{info, error, debug};
use robo_rover_lib::init_tracing;

#[derive(Debug, Clone, Copy)]
struct EncoderConfig {
    jpeg_quality: u8,
    width: u32,
    height: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            jpeg_quality: 80,
            width: 640,
            height: 480,
        }
    }
}

impl EncoderConfig {
    fn from_env() -> Self {
        Self {
            jpeg_quality: env::var("JPEG_QUALITY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(80)
                .clamp(1, 100),
            width: env::var("IMAGE_WIDTH")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(640),
            height: env::var("IMAGE_HEIGHT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(480),
        }
    }
}

fn encode_jpeg(
    rgb_data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>> {
    // Verify data size
    let expected_size = (width * height * 3) as usize;
    if rgb_data.len() != expected_size {
        return Err(eyre!(
            "Invalid RGB data size: expected {} bytes ({}x{}x3), got {} bytes",
            expected_size, width, height, rgb_data.len()
        ));
    }

    // Create image buffer from raw RGB data
    let img_buf = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb_data)
        .ok_or_else(|| eyre!("Failed to create image buffer from RGB data"))?;

    // Encode to JPEG
    let mut jpeg_data = Vec::new();
    {
        let mut cursor = Cursor::new(&mut jpeg_data);
        let mut encoder = JpegEncoder::new_with_quality(&mut cursor, quality);
        encoder.encode(
            &img_buf,
            width,
            height,
            image::ExtendedColorType::Rgb8,
        ).map_err(|e| eyre!("JPEG encoding failed: {}", e))?;
    }

    Ok(jpeg_data)
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    info!("Starting video_encoder node");

    // Load configuration from environment
    let config = EncoderConfig::from_env();
    info!(
        "Encoder config: JPEG quality={}, default resolution={}x{}",
        config.jpeg_quality, config.width, config.height
    );

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;

    // Statistics
    let mut frames_encoded = 0u64;
    let mut encoding_errors = 0u64;
    let mut total_encoding_time_ms = 0u64;

    info!("video_encoder node ready, waiting for video frames...");

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => {
                match id.as_str() {
                    "video_frame" => {
                        let start_time = std::time::Instant::now();

                        // Extract frame metadata
                        let width = metadata.parameters.get("width")
                            .and_then(|v| match v {
                                dora_node_api::Parameter::Integer(i) => Some(*i as u32),
                                _ => None,
                            })
                            .unwrap_or(config.width);

                        let height = metadata.parameters.get("height")
                            .and_then(|v| match v {
                                dora_node_api::Parameter::Integer(i) => Some(*i as u32),
                                _ => None,
                            })
                            .unwrap_or(config.height);

                        // Extract RGB8 data
                        if let Some(rgb_array) = data.as_any().downcast_ref::<UInt8Array>() {
                            let rgb_bytes = rgb_array.values().as_ref();

                            // Encode to JPEG
                            match encode_jpeg(rgb_bytes, width, height, config.jpeg_quality) {
                                Ok(jpeg_data) => {
                                    let encoding_time = start_time.elapsed();
                                    total_encoding_time_ms += encoding_time.as_millis() as u64;
                                    frames_encoded += 1;

                                    // Calculate compression ratio
                                    let compression_ratio = rgb_bytes.len() as f32 / jpeg_data.len() as f32;

                                    debug!(
                                        "Frame {} encoded: {}x{} RGB ({} bytes) → JPEG ({} bytes, {:.1}x compression, {:.1}ms)",
                                        frames_encoded,
                                        width,
                                        height,
                                        rgb_bytes.len(),
                                        jpeg_data.len(),
                                        compression_ratio,
                                        encoding_time.as_secs_f32() * 1000.0
                                    );

                                    // Log performance stats every 100 frames
                                    if frames_encoded % 100 == 0 {
                                        let avg_encoding_time = total_encoding_time_ms as f32 / frames_encoded as f32;
                                        debug!(
                                            "Performance: {} frames encoded, avg {:.1}ms/frame, {} errors",
                                            frames_encoded, avg_encoding_time, encoding_errors
                                        );
                                    }

                                    // Create output metadata with encoding info
                                    let mut output_metadata = metadata.clone();
                                    output_metadata.parameters.insert(
                                        "codec".to_string(),
                                        dora_node_api::Parameter::String("jpeg".to_string())
                                    );
                                    output_metadata.parameters.insert(
                                        "quality".to_string(),
                                        dora_node_api::Parameter::Integer(config.jpeg_quality as i64)
                                    );
                                    output_metadata.parameters.insert(
                                        "compressed_size".to_string(),
                                        dora_node_api::Parameter::Integer(jpeg_data.len() as i64)
                                    );

                                    // Send encoded frame
                                    let binary_data = dora_node_api::arrow::array::BinaryArray::from_vec(vec![jpeg_data.as_slice()]);
                                    node.send_output(
                                        "encoded_frame".to_owned().into(),
                                        output_metadata.parameters,
                                        binary_data,
                                    )?;
                                }
                                Err(e) => {
                                    encoding_errors += 1;
                                    error!("Encoding error (frame {}): {}", frames_encoded + 1, e);
                                }
                            }
                        } else {
                            error!("Invalid video frame data type (expected UInt8Array)");
                            encoding_errors += 1;
                        }
                    }
                    other => {
                        debug!("Ignoring unexpected input: {}", other);
                    }
                }
            }
            Event::Stop(_) => {
                info!("Received stop signal");
                break;
            }
            other => {
                debug!("Ignoring event: {:?}", other);
            }
        }
    }

    // Final statistics
    if frames_encoded > 0 {
        let avg_encoding_time = total_encoding_time_ms as f32 / frames_encoded as f32;
        info!(
            "video_encoder shutting down: {} frames encoded, avg {:.1}ms/frame, {} errors",
            frames_encoded, avg_encoding_time, encoding_errors
        );
    }

    Ok(())
}
