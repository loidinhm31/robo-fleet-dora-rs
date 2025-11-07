use dora_node_api::{
    arrow::array::{Array, Float32Array},
    DoraNode, Event,
};
use eyre::{Result, eyre};
use std::env;
use tracing::{info, error, debug, warn};
use byteorder::{ByteOrder, LittleEndian};
use robo_rover_lib::init_tracing;

#[derive(Debug, Clone, Copy)]
struct AudioConfig {
    sample_rate: u32,
    channels: u16,
    output_format: OutputFormat,
}

#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Int16LE,  // S16LE (16-bit signed PCM, little-endian)
    Float32,  // F32LE (32-bit float PCM)
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            output_format: OutputFormat::Int16LE,
        }
    }
}

impl AudioConfig {
    fn from_env() -> Self {
        let format_str = env::var("OUTPUT_FORMAT").unwrap_or_else(|_| "int16".to_string());
        let output_format = match format_str.to_lowercase().as_str() {
            "int16" | "s16le" => OutputFormat::Int16LE,
            "float32" | "f32le" => OutputFormat::Float32,
            _ => {
                warn!("Unknown OUTPUT_FORMAT '{}', defaulting to int16", format_str);
                OutputFormat::Int16LE
            }
        };

        Self {
            sample_rate: env::var("SAMPLE_RATE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(16000),
            channels: env::var("CHANNELS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
            output_format,
        }
    }
}

/// Convert Float32 samples [-1.0, 1.0] to Int16 samples [-32768, 32767]
fn float32_to_int16(samples: &[f32]) -> Vec<u8> {
    let mut output = vec![0u8; samples.len() * 2];

    for (i, &sample) in samples.iter().enumerate() {
        // Clamp to valid range
        let clamped = sample.clamp(-1.0, 1.0);

        // Convert to i16 range
        let scaled = (clamped * 32767.0) as i16;

        // Write as little-endian bytes
        LittleEndian::write_i16(&mut output[i * 2..(i + 1) * 2], scaled);
    }

    output
}

/// Convert Int16 samples to Float32 samples
fn int16_to_float32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        return Err(eyre!("Invalid int16 data: byte length must be even"));
    }

    let mut samples = Vec::with_capacity(bytes.len() / 2);

    for chunk in bytes.chunks_exact(2) {
        let int_sample = LittleEndian::read_i16(chunk);
        let float_sample = int_sample as f32 / 32767.0;
        samples.push(float_sample);
    }

    Ok(samples)
}

/// Convert Float32 samples to little-endian bytes
fn float32_to_bytes(samples: &[f32]) -> Vec<u8> {
    let mut output = vec![0u8; samples.len() * 4];

    for (i, &sample) in samples.iter().enumerate() {
        LittleEndian::write_f32(&mut output[i * 4..(i + 1) * 4], sample);
    }

    output
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    info!("Starting audio_converter node");

    // Load configuration from environment
    let config = AudioConfig::from_env();
    info!(
        "Audio converter config: sample_rate={} Hz, channels={}, output_format={:?}",
        config.sample_rate, config.channels, config.output_format
    );

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;

    // Statistics
    let mut chunks_converted = 0u64;
    let mut total_samples_processed = 0u64;
    let mut conversion_errors = 0u64;

    info!("audio_converter node ready, waiting for audio data...");

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, metadata, data } => {
                match id.as_str() {
                    "audio_input" => {
                        // Extract Float32 audio data from audio_capture
                        if let Some(float_array) = data.as_any().downcast_ref::<Float32Array>() {
                            let samples = float_array.values().as_ref();
                            total_samples_processed += samples.len() as u64;

                            // Convert based on output format
                            let converted_data = match config.output_format {
                                OutputFormat::Int16LE => {
                                    debug!(
                                        "Converting {} Float32 samples → Int16LE ({} bytes)",
                                        samples.len(),
                                        samples.len() * 2
                                    );
                                    float32_to_int16(samples)
                                }
                                OutputFormat::Float32 => {
                                    debug!(
                                        "Converting {} Float32 samples → Float32LE bytes ({} bytes)",
                                        samples.len(),
                                        samples.len() * 4
                                    );
                                    float32_to_bytes(samples)
                                }
                            };

                            chunks_converted += 1;

                            // Log stats every 100 chunks
                            if chunks_converted % 100 == 0 {
                                debug!(
                                    "Converted {} chunks, {} total samples, {} errors",
                                    chunks_converted, total_samples_processed, conversion_errors
                                );
                            }

                            // Create output metadata
                            let mut output_metadata = metadata.clone();
                            output_metadata.parameters.insert(
                                "format".to_string(),
                                dora_node_api::Parameter::String(
                                    match config.output_format {
                                        OutputFormat::Int16LE => "S16LE",
                                        OutputFormat::Float32 => "F32LE",
                                    }.to_string()
                                )
                            );
                            output_metadata.parameters.insert(
                                "sample_rate".to_string(),
                                dora_node_api::Parameter::Integer(config.sample_rate as i64)
                            );
                            output_metadata.parameters.insert(
                                "channels".to_string(),
                                dora_node_api::Parameter::Integer(config.channels as i64)
                            );
                            output_metadata.parameters.insert(
                                "size".to_string(),
                                dora_node_api::Parameter::Integer(converted_data.len() as i64)
                            );

                            // Send converted audio
                            let binary_data = dora_node_api::arrow::array::BinaryArray::from_vec(vec![converted_data.as_slice()]);
                            node.send_output(
                                "audio_output".to_owned().into(),
                                output_metadata.parameters,
                                binary_data,
                            )?;
                        } else {
                            error!("Invalid audio data type (expected Float32Array)");
                            conversion_errors += 1;
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
    if chunks_converted > 0 {
        info!(
            "audio_converter shutting down: {} chunks converted, {} samples processed, {} errors",
            chunks_converted, total_samples_processed, conversion_errors
        );
    }

    Ok(())
}
