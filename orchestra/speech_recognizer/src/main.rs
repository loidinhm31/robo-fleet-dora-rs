use dora_node_api::arrow::array::{Array, BinaryArray, Float32Array};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event};
use eyre::Result;
use robo_rover_lib::{SpeechTranscription, init_tracing};
use std::env;
use std::path::PathBuf;
use std::time::Instant;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

const DEFAULT_SAMPLE_RATE: u32 = 16000;
const DEFAULT_BUFFER_DURATION_MS: u32 = 5000; // 5 seconds
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;
const DEFAULT_ENERGY_THRESHOLD: f32 = 0.02; // VAD threshold

fn main() -> Result<()> {
    let _guard = init_tracing();
    tracing::info!("Starting speech recognizer node...");

    // Read configuration from environment variables
    let model_path = env::var("WHISPER_MODEL_PATH")
        .unwrap_or_else(|_| "models/ggml-tiny.bin".to_string());

    let sample_rate: u32 = env::var("SAMPLE_RATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SAMPLE_RATE);

    let buffer_duration_ms: u32 = env::var("BUFFER_DURATION_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BUFFER_DURATION_MS);

    let confidence_threshold: f32 = env::var("CONFIDENCE_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CONFIDENCE_THRESHOLD);

    let energy_threshold: f32 = env::var("ENERGY_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ENERGY_THRESHOLD);

    tracing::info!("Configuration:");
    tracing::info!("Model path: {}", model_path);
    tracing::info!("Sample rate: {} Hz", sample_rate);
    tracing::info!("Buffer duration: {} ms", buffer_duration_ms);
    tracing::info!("Confidence threshold: {}", confidence_threshold);
    tracing::info!("Energy threshold: {}", energy_threshold);

    // Load Whisper model
    tracing::info!("Loading Whisper model from: {}", model_path);
    let model_path = PathBuf::from(model_path);

    if !model_path.exists() {
        tracing::error!("Whisper model not found at: {:?}", model_path);
        tracing::error!("Please download a Whisper model:");
        tracing::error!("Download from: https://huggingface.co/ggerganov/whisper.cpp/tree/main");
        tracing::error!("For Raspberry Pi 5, use: ggml-tiny.bin or ggml-base.bin");
        tracing::error!("Place in models/ directory");
        return Err(eyre::eyre!("Whisper model not found"));
    }

    let ctx = WhisperContext::new_with_params(
        &model_path.to_string_lossy(),
        whisper_rs::WhisperContextParameters::default(),
    )
    .map_err(|e| eyre::eyre!("Failed to load Whisper model: {}", e))?;

    tracing::info!("Whisper model loaded successfully!");

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let transcription_output = DataId::from("transcription".to_owned());

    // Audio buffer
    let max_buffer_samples = (sample_rate * buffer_duration_ms / 1000) as usize;
    let mut audio_buffer: Vec<f32> = Vec::with_capacity(max_buffer_samples);

    // Statistics
    let mut total_transcriptions = 0u64;
    let mut total_confidence = 0.0f32;
    let mut total_processing_time_ms = 0.0f32;

    tracing::info!("Speech recognizer ready! Waiting for audio...");

    loop {
        match events.recv() {
            Some(Event::Input { id, data, .. }) => match id.as_str() {
                "audio_rover" | "audio_web" => {
                    // Receive audio from either rover microphone or web UI
                    let audio_data = handle_audio_input(&*data)?;

                    // Append to buffer
                    audio_buffer.extend_from_slice(&audio_data);

                    // Check if buffer is full or VAD detects end of speech
                    let should_process = audio_buffer.len() >= max_buffer_samples;
                    let has_energy = calculate_energy(&audio_buffer) > energy_threshold;

                    if should_process && has_energy {
                        tracing::debug!("Processing {} samples ({:.2}s of audio)...",
                            audio_buffer.len(),
                            audio_buffer.len() as f32 / sample_rate as f32);

                        let start_time = Instant::now();

                        // Run Whisper inference
                        match transcribe_audio(&ctx, &audio_buffer, sample_rate) {
                            Ok((text, confidence)) => {
                                let processing_time = start_time.elapsed().as_millis() as f32;

                                tracing::info!("Transcription: \"{}\" (confidence: {:.2}, time: {:.0}ms)",
                                    text, confidence, processing_time);

                                // Update statistics
                                total_transcriptions += 1;
                                total_confidence += confidence;
                                total_processing_time_ms += processing_time;

                                // Only output if confidence is above threshold
                                if confidence >= confidence_threshold && !text.trim().is_empty() {
                                    let transcription = SpeechTranscription {
                                        text: text.clone(),
                                        confidence,
                                        language: "en".to_string(),
                                        duration_ms: (audio_buffer.len() as u64 * 1000) / sample_rate as u64,
                                        timestamp: std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis() as i64,
                                    };

                                    // Send transcription via Dora
                                    let json_bytes = serde_json::to_vec(&transcription)?;
                                    let array = BinaryArray::from_vec(vec![json_bytes.as_slice()]);
                                    node.send_output(
                                        transcription_output.clone(),
                                        Default::default(),
                                        array,
                                    )?;

                                    tracing::debug!("Sent transcription (avg conf: {:.2}, avg time: {:.0}ms)",
                                        total_confidence / total_transcriptions as f32,
                                        total_processing_time_ms / total_transcriptions as f32);
                                } else {
                                    tracing::debug!("Skipped: low confidence or empty text");
                                }
                            }
                            Err(e) => {
                                tracing::error!("Transcription error: {}", e);
                            }
                        }

                        // Clear buffer for next segment
                        audio_buffer.clear();
                    } else if !has_energy && audio_buffer.len() > sample_rate as usize {
                        // Discard silent audio after 1 second
                        audio_buffer.clear();
                    }
                }
                _ => {}
            },
            Some(Event::Stop(_)) => {
                tracing::info!("Stopping speech recognizer node...");
                tracing::info!("Statistics:");
                tracing::info!("Total transcriptions: {}", total_transcriptions);
                if total_transcriptions > 0 {
                    tracing::info!("Average confidence: {:.2}", total_confidence / total_transcriptions as f32);
                    tracing::info!("Average processing time: {:.0}ms", total_processing_time_ms / total_transcriptions as f32);
                }
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Handle audio input from Dora (Float32Array from audio_capture)
fn handle_audio_input(data: &dyn Array) -> Result<Vec<f32>> {
    if let Some(float_array) = data.as_any().downcast_ref::<Float32Array>() {
        Ok(float_array.values().to_vec())
    } else {
        Err(eyre::eyre!("Expected Float32Array from audio_capture"))
    }
}

/// Calculate RMS energy of audio buffer (for simple VAD)
fn calculate_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Transcribe audio using Whisper
fn transcribe_audio(
    ctx: &WhisperContext,
    audio: &[f32],
    sample_rate: u32,
) -> Result<(String, f32)> {
    // Resample if needed (Whisper expects 16kHz)
    let audio_16k = if sample_rate != 16000 {
        resample_audio(audio, sample_rate, 16000)
    } else {
        audio.to_vec()
    };

    // Create Whisper parameters
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    // Configure for real-time performance on Raspberry Pi
    params.set_n_threads(4); // Use 4 cores on RPi 5
    params.set_language(Some("en")); // Set language for better performance
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Create a mutable state for Whisper
    let mut state = ctx.create_state()
        .map_err(|e| eyre::eyre!("Failed to create Whisper state: {}", e))?;

    // Run inference
    state.full(params, &audio_16k)
        .map_err(|e| eyre::eyre!("Whisper inference failed: {}", e))?;

    // Get transcription
    let num_segments = state.full_n_segments()
        .map_err(|e| eyre::eyre!("Failed to get segment count: {}", e))?;

    let mut full_text = String::new();
    let mut total_confidence = 0.0f32;
    let mut segment_count = 0;

    for i in 0..num_segments {
        if let Ok(segment_text) = state.full_get_segment_text(i) {
            full_text.push_str(&segment_text);
            full_text.push(' ');

            // Whisper doesn't provide per-segment confidence, so we estimate
            // based on the presence of special tokens or length
            let segment_confidence = if segment_text.trim().is_empty() {
                0.0
            } else {
                0.9 // Default high confidence for non-empty segments
            };

            total_confidence += segment_confidence;
            segment_count += 1;
        }
    }

    let avg_confidence = if segment_count > 0 {
        total_confidence / segment_count as f32
    } else {
        0.0
    };

    Ok((full_text.trim().to_string(), avg_confidence))
}

/// Simple resampling (linear interpolation) - for production, use a proper resampler
fn resample_audio(audio: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return audio.to_vec();
    }

    let ratio = from_rate as f32 / to_rate as f32;
    let output_len = (audio.len() as f32 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f32 * ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let src_idx_ceil = (src_idx_floor + 1).min(audio.len() - 1);
        let frac = src_idx - src_idx_floor as f32;

        // Linear interpolation
        let sample = audio[src_idx_floor] * (1.0 - frac) + audio[src_idx_ceil] * frac;
        output.push(sample);
    }

    output
}
