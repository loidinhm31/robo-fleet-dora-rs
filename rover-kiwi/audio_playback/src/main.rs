use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream, StreamConfig};
use dora_node_api::arrow::array::{Array, BinaryArray, Float32Array, Int16Array};
use dora_node_api::{DoraNode, Event};
use eyre::Result;
use robo_rover_lib::init_tracing;
use std::sync::{Arc, Mutex};

/// Simple ring buffer for audio playback
struct AudioBuffer {
    buffer: Vec<f32>,
    read_pos: usize,
    write_pos: usize,
    capacity: usize,
}

impl AudioBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity],
            read_pos: 0,
            write_pos: 0,
            capacity,
        }
    }

    fn write(&mut self, data: &[f32]) -> usize {
        let mut written = 0;
        for &sample in data {
            let available = (self.read_pos + self.capacity - self.write_pos - 1) % self.capacity;
            if available == 0 {
                break; // Buffer full
            }
            self.buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.capacity;
            written += 1;
        }
        written
    }

    fn read(&mut self, data: &mut [f32]) -> usize {
        let mut read = 0;
        for sample in data.iter_mut() {
            if self.read_pos == self.write_pos {
                *sample = 0.0; // Silence if buffer empty
            } else {
                *sample = self.buffer[self.read_pos];
                self.read_pos = (self.read_pos + 1) % self.capacity;
                read += 1;
            }
        }
        read
    }

    fn available(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    tracing::info!("Starting audio_playback node...");

    // Get audio configuration from environment
    let sample_rate: u32 = std::env::var("SAMPLE_RATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16000);

    let channels: u16 = std::env::var("CHANNELS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    tracing::info!(
        "Audio configuration: sample_rate={}Hz, channels={}",
        sample_rate,
        channels
    );

    // Initialize audio output device
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| eyre::eyre!("No output device available"))?;

    tracing::info!("Using audio output device: {}", device.name()?);

    // Configure audio stream
    let config = StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    tracing::info!("Stream config: {:?}", config);

    // Create shared audio buffer (5 seconds of audio)
    let buffer_capacity = (sample_rate * channels as u32 * 5) as usize;
    let audio_buffer = Arc::new(Mutex::new(AudioBuffer::new(buffer_capacity)));
    let audio_buffer_clone = audio_buffer.clone();

    // Build audio output stream
    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut buffer = audio_buffer_clone.lock().unwrap();
            buffer.read(data);
        },
        |err| {
            tracing::error!("Audio stream error: {}", err);
        },
        None,
    )?;

    stream.play()?;
    tracing::info!("Audio playback stream started");

    // Initialize Dora node
    let (_node, mut events) = DoraNode::init_from_env()?;

    tracing::info!("Audio playback node ready");

    let mut total_samples = 0u64;
    let mut buffer_overruns = 0u64;

    // Main event loop
    loop {
        match events.recv() {
            Some(Event::Input { id, data, .. }) => match id.as_str() {
                "audio" => {
                    // Handle incoming audio from web_bridge
                    let samples = parse_audio_data(&*data)?;

                    // Write to playback buffer
                    let mut buffer = audio_buffer.lock().unwrap();
                    let written = buffer.write(&samples);

                    total_samples += written as u64;

                    if written < samples.len() {
                        buffer_overruns += 1;
                        tracing::warn!(
                            "Buffer overrun: wrote {}/{} samples (total overruns: {})",
                            written,
                            samples.len(),
                            buffer_overruns
                        );
                    }

                    let available = buffer.available();
                    tracing::debug!(
                        "Received {} samples, buffer: {}/{} ({:.1}%)",
                        samples.len(),
                        available,
                        buffer_capacity,
                        (available as f32 / buffer_capacity as f32) * 100.0
                    );
                }
                other => {
                    tracing::warn!("Unexpected input: {}", other);
                }
            },
            Some(Event::Stop(_)) => {
                tracing::info!("Stop event received");
                tracing::info!(
                    "Statistics: total_samples={}, buffer_overruns={}",
                    total_samples,
                    buffer_overruns
                );
                break;
            }
            Some(_) => {}
            None => {
                break;
            }
        }
    }

    // Stop stream
    drop(stream);
    tracing::info!("Audio playback node stopped");

    Ok(())
}

/// Parse audio data from various Arrow array formats
fn parse_audio_data(data: &dyn Array) -> Result<Vec<f32>> {
    // Try Float32Array (preferred format from web)
    if let Some(float_array) = data.as_any().downcast_ref::<Float32Array>() {
        return Ok(float_array.values().to_vec());
    }

    // Try Int16Array (16-bit PCM)
    if let Some(int16_array) = data.as_any().downcast_ref::<Int16Array>() {
        let samples: Vec<f32> = int16_array
            .values()
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();
        return Ok(samples);
    }

    // Try BinaryArray (for serialized audio data)
    if let Some(binary_array) = data.as_any().downcast_ref::<BinaryArray>() {
        if binary_array.len() > 0 {
            let bytes = binary_array.value(0);

            // Try to parse as Float32 samples
            if bytes.len() % 4 == 0 {
                let samples: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                    })
                    .collect();
                return Ok(samples);
            }

            // Try to parse as Int16 samples
            if bytes.len() % 2 == 0 {
                let samples: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let value = i16::from_le_bytes([chunk[0], chunk[1]]);
                        value as f32 / 32768.0
                    })
                    .collect();
                return Ok(samples);
            }
        }
    }

    Err(eyre::eyre!("Unsupported audio format"))
}
