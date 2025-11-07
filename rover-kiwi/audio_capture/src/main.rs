use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, Stream, StreamConfig};
use dora_node_api::arrow::array::{Array, BinaryArray, Float32Array};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::Result;
use ringbuf::{traits::*, HeapRb};
use robo_rover_lib::{init_tracing, AudioAction, AudioControl};
use std::env;
use std::sync::{Arc, Mutex};

fn main() -> Result<()> {
    let _guard = init_tracing();

    tracing::info!("Starting audio capture node");

    // Read configuration from environment variables with defaults
    let sample_rate: u32 = env::var("SAMPLE_RATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16000);

    let channels: u16 = env::var("CHANNELS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let chunk_size: usize = env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(800);

    tracing::info!(
        "Audio configuration: {}Hz, {} channels, {} samples per chunk",
        sample_rate, channels, chunk_size
    );

    let host = cpal::default_host();

    // Get default input device
    let device = host
        .default_input_device()
        .ok_or_else(|| eyre::eyre!("No default input device available"))?;

    tracing::info!("Using audio device: {}", device.name()?);

    // Configure audio stream
    let config = StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Fixed(chunk_size as u32),
    };

    // Initialize Dora node
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let output_id = DataId::from("audio".to_owned());

    // Create ring buffer for audio samples (larger buffer to prevent underruns)
    // Use Arc<Mutex<>> to share between stream callback and main loop
    let ring = HeapRb::<f32>::new(chunk_size * 10);
    let (producer, consumer) = ring.split();
    let producer = Arc::new(Mutex::new(producer));
    let consumer = Arc::new(Mutex::new(consumer));

    // Build audio input stream (using f32 samples)
    let err_fn = |err| tracing::error!("Audio stream error: {}", err);
    let producer_clone = producer.clone();

    let mut stream_opt: Option<Stream> = Some(device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            if let Ok(mut prod) = producer_clone.lock() {
                write_audio_data(data, &mut prod);
            }
        },
        err_fn,
        None,
    )?);

    stream_opt.as_ref().unwrap().play()?;
    tracing::info!("Audio stream started successfully");

    let mut frame_count = 0u64;
    let mut audio_buffer = Vec::with_capacity(chunk_size);

    loop {
        match events.recv() {
            Some(Event::Input { id, data, .. }) => match id.as_str() {
                "tick" => {
                    // Only process audio if stream is active
                    if stream_opt.is_some() {
                        // Read available samples from ring buffer
                        if let Ok(mut cons) = consumer.lock() {
                            while cons.occupied_len() > 0 && audio_buffer.len() < chunk_size {
                                if let Some(sample) = cons.try_pop() {
                                    audio_buffer.push(sample);
                                } else {
                                    break;
                                }
                            }
                        }

                        // Send chunk when we have enough samples
                        if audio_buffer.len() >= chunk_size {
                            let chunk: Vec<f32> = audio_buffer.drain(..chunk_size).collect();

                            // Create Float32Array
                            let audio_array = Float32Array::from(chunk.clone());

                            // Create metadata
                            let mut metadata = MetadataParameters::default();
                            metadata.insert("sample_rate".to_string(), Parameter::Integer(sample_rate as i64));
                            metadata.insert("channels".to_string(), Parameter::Integer(channels as i64));
                            metadata.insert("format".to_string(), Parameter::String("f32le".to_string()));

                            // Send to Dora
                            node.send_output(output_id.clone(), metadata, audio_array)?;

                            frame_count += 1;
                            if frame_count <= 5 {
                                let min = chunk.iter().cloned().fold(f32::INFINITY, f32::min);
                                let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                tracing::debug!(
                                    "Sent audio frame {}: {} samples, range [{:.3}, {:.3}]",
                                    frame_count,
                                    chunk_size,
                                    min,
                                    max
                                );
                            }
                        }
                    }
                }
                "audio_control" => {
                    if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
                        if binary_array.len() > 0 {
                            let control_bytes = binary_array.value(0);
                            if let Ok(audio_control) =
                                serde_json::from_slice::<AudioControl>(control_bytes)
                            {
                                tracing::info!("Audio control received: {:?}", audio_control.command);
                                match audio_control.command {
                                    AudioAction::Start => {
                                        if stream_opt.is_none() {
                                            tracing::info!("Starting audio stream");
                                            // Clear existing buffers and recreate stream
                                            audio_buffer.clear();
                                            if let Ok(mut cons) = consumer.lock() {
                                                // Drain any remaining samples
                                                while cons.try_pop().is_some() {}
                                            }

                                            let producer_clone = producer.clone();
                                            let new_stream = device.build_input_stream(
                                                &config,
                                                move |data: &[f32], _: &_| {
                                                    if let Ok(mut prod) = producer_clone.lock() {
                                                        write_audio_data(data, &mut prod);
                                                    }
                                                },
                                                err_fn,
                                                None,
                                            )?;
                                            new_stream.play()?;
                                            stream_opt = Some(new_stream);
                                            tracing::info!("Audio stream started");
                                        }
                                    }
                                    AudioAction::Stop => {
                                        if let Some(_stream) = stream_opt.take() {
                                            tracing::info!("Stopping audio stream");
                                            // Stream is dropped here, stopping capture
                                            // Clear audio buffer
                                            audio_buffer.clear();
                                            tracing::info!("Audio stream stopped");
                                        }
                                    }
                                }
                            } else {
                                tracing::error!("Failed to parse audio control command");
                            }
                        }
                    }
                }
                other => tracing::warn!("Ignoring unexpected input: {}", other),
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

    drop(stream_opt);
    tracing::info!("Audio capture stopped");
    Ok(())
}

fn write_audio_data<T>(input: &[T], producer: &mut ringbuf::HeapProd<f32>)
where
    T: Sample,
    f32: FromSample<T>,
{
    for &sample in input {
        let sample_f32 = f32::from_sample(sample);
        // Try to push, but don't block if buffer is full (drop oldest samples)
        let _ = producer.try_push(sample_f32);
    }
}
