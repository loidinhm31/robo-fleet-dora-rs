use dora_node_api::arrow::array::{Array, BinaryArray};
use dora_node_api::{DoraNode, Event};
use eyre::Result;
use robo_rover_lib::{init_tracing, TtsCommand};
use sherpa_rs::tts::{CommonTtsConfig, VitsTts, VitsTtsConfig};
use sherpa_rs::OnnxConfig;
use std::env;
use rodio::{OutputStream, Sink};

fn main() -> Result<()> {
    let _guard = init_tracing();

    tracing::info!("Starting Sherpa-ONNX TTS node...");
    tracing::info!("Using VITS model for lightweight speech synthesis on edge device");

    // Get configuration from environment variables
    let model_dir = env::var("TTS_MODEL_DIR")
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap()
                .join(".cache/sherpa-onnx/vits-piper-en_US-lessac-medium")
                .to_string_lossy()
                .to_string()
        });

    let volume = env::var("TTS_VOLUME")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.8);

    let speed = env::var("TTS_SPEED")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);

    tracing::info!("TTS configuration: model_dir={}, volume={}, speed={}", model_dir, volume, speed);

    // Initialize VITS TTS configuration
    tracing::info!("Initializing Sherpa-ONNX VITS TTS engine...");

    let onnx_config = OnnxConfig {
        num_threads: env::var("TTS_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(2),
        provider: env::var("TTS_PROVIDER").unwrap_or_else(|_| "cpu".to_string()),
        debug: false,
    };

    let common_config = CommonTtsConfig {
        rule_fars: String::new(),
        rule_fsts: String::new(),
        max_num_sentences: 1,
        silence_scale: 1.0,
    };

    let vits_config = VitsTtsConfig {
        model: format!("{}/model.onnx", model_dir),
        lexicon: String::new(),  // VITS-Piper doesn't use lexicon
        tokens: format!("{}/tokens.txt", model_dir),
        data_dir: format!("{}/espeak-ng-data", model_dir),  // espeak-ng-data for phoneme processing
        dict_dir: String::new(),
        length_scale: 1.0,
        noise_scale: 0.667,
        noise_scale_w: 0.8,
        silence_scale: 1.0,
        onnx_config,
        tts_config: common_config,
    };

    tracing::info!("Creating VITS TTS engine...");
    let mut tts = VitsTts::new(vits_config);
    tracing::info!("Sherpa-ONNX VITS TTS engine initialized successfully");

    // Initialize audio output
    let (_stream, stream_handle) = OutputStream::try_default()?;

    // Initialize Dora node
    let (_node, mut events) = DoraNode::init_from_env()?;

    tracing::info!("TTS node ready to process commands");

    // Main event loop
    loop {
        match events.recv() {
            Some(Event::Input { id, data, .. }) => match id.as_str() {
                "tts_command" | "tts_command_web" => {
                    if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                        if binary_array.len() > 0 {
                            let command_bytes = binary_array.value(0);
                            if let Ok(tts_command) = serde_json::from_slice::<TtsCommand>(command_bytes) {
                                tracing::info!("TTS command received from {}: '{}'", id, tts_command.text);

                                // Synthesize the text
                                match tts.create(&tts_command.text, 0, speed) {
                                    Ok(audio) => {
                                        tracing::debug!(
                                            "Audio synthesized: {} samples at {} Hz, duration: {}s",
                                            audio.samples.len(),
                                            audio.sample_rate,
                                            audio.duration
                                        );

                                        // Play the audio
                                        if let Err(e) = play_audio(&stream_handle, &audio.samples, audio.sample_rate, volume) {
                                            tracing::error!("Failed to play audio: {}", e);
                                        } else {
                                            tracing::info!("TTS playback completed");
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!("TTS synthesis error: {}", e);
                                    }
                                }
                            } else {
                                tracing::error!("Failed to parse TTS command");
                            }
                        }
                    }
                }
                other => {
                    tracing::warn!("Ignoring unexpected input: {}", other);
                }
            },
            Some(Event::Stop(_)) => {
                tracing::info!("Stop event received");
                break;
            }
            Some(_) => {}
            None => {
                break;
            }
        }
    }

    tracing::info!("Sherpa-ONNX TTS node stopped");
    Ok(())
}

fn play_audio(
    stream_handle: &rodio::OutputStreamHandle,
    samples: &[f32],
    sample_rate: u32,
    volume: f32,
) -> Result<()> {
    let sink = Sink::try_new(stream_handle)?;
    sink.set_volume(volume);

    // Create a source from the samples
    let source = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples.to_vec());

    sink.append(source);
    sink.sleep_until_end();

    Ok(())
}
