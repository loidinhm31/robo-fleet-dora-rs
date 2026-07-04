mod audio_dump;
mod capture_gate;
mod signal_metrics;

use capture_gate::CaptureGate;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream};
use dora_node_api::arrow::array::{Array, BinaryArray, Float32Array};
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use eyre::Result;
use ringbuf::{traits::*, HeapRb};
use robo_rover_lib::PlaybackState;
use robo_rover_lib::{
    describe_device_preference, init_tracing, matches_device_override, select_input_capture_plan,
    select_preferred_device_name, AudioAction, AudioControl, InputCapturePlan, MetricWindow,
    SupportedInputConfigDescriptor,
};
use signal_metrics::{analyze_signal, PreflightSignalProbe, SignalMetricWindow};
use std::env;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const SILENCE_THRESHOLD: f32 = 300.0 / 32768.0;
const PRE_FLIGHT_DURATION_MS: u32 = 1000;
const SIGNAL_METRIC_WINDOW: Duration = Duration::from_secs(5);

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
        sample_rate,
        channels,
        chunk_size
    );

    // Optional debug dump of exactly what this node sends downstream (post
    // downmix/resample, e.g. 16 kHz mono f32) to a playable WAV. Off unless
    // AUDIO_DUMP_FILE is set; AUDIO_DUMP_MAX_SECS caps the file size.
    let mut audio_dumper = audio_dump::AudioDumper::from_env(sample_rate, channels);

    // Initialize Dora node first — so a failed audio init never cascades to other nodes
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let output_id = DataId::from("audio".to_owned());

    // Create ring buffer for audio samples (larger buffer to prevent underruns)
    let ring = HeapRb::<f32>::new(chunk_size * 10);
    let (producer, consumer) = ring.split();
    let producer = Arc::new(Mutex::new(producer));
    let consumer = Arc::new(Mutex::new(consumer));
    let rejected_samples = Arc::new(AtomicU64::new(0));

    // Try to open the microphone — degrade gracefully if ALSA/hardware is unavailable
    let mut stream_opt: Option<Stream> = match try_open_input_stream(
        sample_rate,
        channels,
        producer.clone(),
        rejected_samples.clone(),
    ) {
        Ok(stream) => {
            tracing::info!("Audio stream started successfully");
            Some(stream)
        }
        Err(e) => {
            tracing::warn!(
                "Audio input unavailable ({}), running in silent mode. \
                     No audio will be captured or sent.",
                e
            );
            None
        }
    };

    let stream_id = Uuid::new_v4();
    let mut next_frame_id = 0u64;
    let mut frames_sent = 0u64;
    let mut send_errors = 0u64;
    let mut samples_rejected = 0u64;
    let mut capture_metrics = MetricWindow::new(SIGNAL_METRIC_WINDOW);
    let mut signal_metrics = SignalMetricWindow::new(SIGNAL_METRIC_WINDOW);
    let mut preflight_probe =
        PreflightSignalProbe::new(sample_rate, channels, PRE_FLIGHT_DURATION_MS);
    let mut audio_buffer = Vec::with_capacity(chunk_size);
    let mut capture_gate = CaptureGate::new(true);
    tracing::info!(%stream_id, "audio capture stream identity created");

    loop {
        match events.recv() {
            Some(Event::Input { id, data, .. }) => match id.as_str() {
                "tick" => {
                    let rejected = rejected_samples.swap(0, Ordering::Relaxed);
                    if rejected > 0 {
                        samples_rejected = samples_rejected.saturating_add(rejected);
                        capture_metrics.record_drops(rejected);
                    }
                    // Keep draining captured samples while playback suppression is active.
                    // This prevents speaker audio queued during the 400 ms tail from leaking
                    // into the first frame published after capture resumes.
                    if !capture_gate.can_publish(Instant::now()) {
                        clear_capture_buffers(&consumer, &mut audio_buffer);
                    } else if stream_opt.is_some() {
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
                            audio_dumper.write_chunk(&chunk);
                            let started = Instant::now();
                            let frame_id = next_frame_id;
                            next_frame_id = next_frame_id.saturating_add(1);
                            let capture_timestamp_ms = current_time_ms()?;
                            let sample_count = u32::try_from(chunk.len())?;
                            let payload_bytes = chunk.len() * std::mem::size_of::<f32>();
                            let signal_summary = analyze_signal(&chunk, SILENCE_THRESHOLD);
                            let debug_range = (frames_sent < 5)
                                .then_some((signal_summary.min_sample, signal_summary.max_sample));
                            preflight_probe.observe_summary(&signal_summary);
                            preflight_probe.log_if_ready();
                            signal_metrics.record_summary(&signal_summary);

                            // Create metadata
                            let mut metadata = MetadataParameters::default();
                            metadata.insert(
                                "sample_rate".to_string(),
                                Parameter::Integer(sample_rate as i64),
                            );
                            metadata.insert(
                                "channels".to_string(),
                                Parameter::Integer(channels as i64),
                            );
                            metadata.insert(
                                "format".to_string(),
                                Parameter::String("f32le".to_string()),
                            );
                            metadata.insert(
                                "stream_id".to_string(),
                                Parameter::String(stream_id.to_string()),
                            );
                            metadata.insert(
                                "frame_id".to_string(),
                                Parameter::Integer(i64::try_from(frame_id)?),
                            );
                            metadata.insert(
                                "capture_timestamp_ms".to_string(),
                                Parameter::Integer(i64::try_from(capture_timestamp_ms)?),
                            );
                            metadata.insert(
                                "sample_count".to_string(),
                                Parameter::Integer(i64::from(sample_count)),
                            );

                            // Send to Dora
                            let audio_array = Float32Array::from(chunk);
                            if let Err(error) =
                                node.send_output(output_id.clone(), metadata, audio_array)
                            {
                                send_errors = send_errors.saturating_add(1);
                                capture_metrics.record_error();
                                tracing::error!(%error, %stream_id, frame_id, "failed to send audio frame to Dora");
                                continue;
                            }

                            frames_sent = frames_sent.saturating_add(1);
                            capture_metrics.record(started.elapsed(), payload_bytes);
                            if let Some((min, max)) = debug_range {
                                tracing::debug!(
                                    %stream_id,
                                    frame_id,
                                    "Sent audio frame: {} samples, range [{:.3}, {:.3}]",
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
                                tracing::info!(
                                    "Audio control received: {:?}",
                                    audio_control.command
                                );
                                match audio_control.command {
                                    AudioAction::Start => {
                                        capture_gate.set_user_enabled(true);
                                        if stream_opt.is_none() {
                                            tracing::info!("Starting audio stream");
                                            // Clear existing buffers and recreate stream
                                            audio_buffer.clear();
                                            if let Ok(mut cons) = consumer.lock() {
                                                // Drain any remaining samples
                                                while cons.try_pop().is_some() {}
                                            }

                                            match try_open_input_stream(
                                                sample_rate,
                                                channels,
                                                producer.clone(),
                                                rejected_samples.clone(),
                                            ) {
                                                Ok(new_stream) => {
                                                    stream_opt = Some(new_stream);
                                                    preflight_probe = PreflightSignalProbe::new(
                                                        sample_rate,
                                                        channels,
                                                        PRE_FLIGHT_DURATION_MS,
                                                    );
                                                    tracing::info!("Audio stream started");
                                                }
                                                Err(e) => {
                                                    tracing::warn!(
                                                        "Failed to start audio stream: {}",
                                                        e
                                                    );
                                                }
                                            }
                                        }
                                    }
                                    AudioAction::Stop => {
                                        capture_gate.set_user_enabled(false);
                                        if let Some(_stream) = stream_opt.take() {
                                            preflight_probe
                                                .log_if_pending("capture_preflight_partial");
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
                "playback_state" => match parse_playback_state(&*data) {
                    Ok(state) => {
                        capture_gate.observe_playback(&state, Instant::now());
                        clear_capture_buffers(&consumer, &mut audio_buffer);
                    }
                    Err(error) => tracing::warn!(%error, "rejected playback suppression state"),
                },
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

        if let Some(snapshot) = capture_metrics.snapshot_if_due() {
            let signal_snapshot = signal_metrics.snapshot();
            tracing::info!(
                metric = "audio_pipeline",
                stage = "capture",
                %stream_id,
                count = snapshot.count,
                bytes = snapshot.bytes,
                drops = snapshot.drops,
                errors = snapshot.errors,
                p50_us = snapshot.p50_us,
                p95_us = snapshot.p95_us,
                p99_us = snapshot.p99_us,
                max_us = snapshot.max_us,
                rms_dbfs = signal_snapshot.rms_dbfs,
                peak_dbfs = signal_snapshot.peak_dbfs,
                silence_pct = signal_snapshot.silence_pct
            );
        }
    }

    preflight_probe.log_if_pending("capture_preflight_partial");
    drop(stream_opt);
    audio_dumper.close();
    tracing::info!(
        metric = "audio_pipeline_total",
        stage = "capture",
        %stream_id,
        frames_sent,
        send_errors,
        samples_rejected,
        "Audio capture stopped"
    );
    Ok(())
}

fn parse_playback_state(data: &dyn Array) -> Result<PlaybackState> {
    let binary = data
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| eyre::eyre!("playback state must be BinaryArray"))?;
    if binary.len() != 1 {
        return Err(eyre::eyre!("playback state must contain one payload"));
    }
    let state = serde_json::from_slice::<PlaybackState>(binary.value(0))?;
    state.validate().map_err(eyre::Report::msg)?;
    Ok(state)
}

fn clear_capture_buffers(
    consumer: &Arc<Mutex<ringbuf::HeapCons<f32>>>,
    audio_buffer: &mut Vec<f32>,
) {
    audio_buffer.clear();
    if let Ok(mut consumer) = consumer.lock() {
        while consumer.try_pop().is_some() {}
    }
}

/// Try to open a CPAL input stream. Returns Err if no microphone / ALSA device is available.
/// Respects AUDIO_DEVICE env var: substring-matches against device names reported by cpal.
/// Example: AUDIO_DEVICE="USB Audio" targets the USB camera mic on systems where card 0/1
/// are playback-only and ALSA's 'default' resolves to the wrong card.
fn try_open_input_stream(
    sample_rate: u32,
    channels: u16,
    producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    rejected_samples: Arc<AtomicU64>,
) -> Result<Stream> {
    let host = cpal::default_host();
    let default_device_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let mut devices: Vec<(String, cpal::Device)> = host
        .input_devices()?
        .filter_map(|device| device.name().ok().map(|name| (name, device)))
        .collect();
    let available_names: Vec<String> = devices.iter().map(|(name, _)| name.clone()).collect();

    if available_names.is_empty() {
        return Err(eyre::eyre!("No audio input devices available"));
    }

    let candidate_summary: Vec<String> = available_names
        .iter()
        .map(|name| {
            format!(
                "{} [score={}, reason={}]",
                name,
                robo_rover_lib::device_preference_score(name),
                describe_device_preference(name, default_device_name.as_deref())
            )
        })
        .collect();

    let selected_index = match env::var("AUDIO_DEVICE")
        .ok()
        .filter(|target| !target.trim().is_empty())
    {
        Some(target) => {
            tracing::info!("AUDIO_DEVICE='{}', scanning input devices", target);
            if let Some(index) = devices
                .iter()
                .position(|(name, _)| matches_device_override(name, &target))
            {
                index
            } else {
                let fallback_index = auto_select_device_index(
                    &devices,
                    &available_names,
                    default_device_name.as_deref(),
                )?;
                tracing::warn!(
                    override_value = %target,
                    candidates = ?candidate_summary,
                    fallback_device = %devices[fallback_index].0,
                    fallback_reason = describe_device_preference(
                        &devices[fallback_index].0,
                        default_device_name.as_deref()
                    ),
                    "AUDIO_DEVICE override did not match any input device; falling back to auto-detect"
                );
                fallback_index
            }
        }
        None => {
            tracing::info!(
                candidates = ?candidate_summary,
                default_device = ?default_device_name,
                "Discovered audio input devices for auto-detect"
            );
            auto_select_device_index(&devices, &available_names, default_device_name.as_deref())?
        }
    };

    let (device_name, device) = devices.swap_remove(selected_index);
    let supported_configs: Vec<SupportedInputConfigDescriptor> = device
        .supported_input_configs()?
        .map(SupportedInputConfigDescriptor::from)
        .collect();
    let capture_plan = select_input_capture_plan(&supported_configs, sample_rate, channels)
        .ok_or_else(|| {
            eyre::eyre!(
                "No compatible capture config for {}Hz/{}ch on '{}'. Supported: {:?}",
                sample_rate,
                channels,
                device_name,
                supported_configs
            )
        })?;

    tracing::info!(
        device = %device_name,
        capture_channels = capture_plan.capture_channels,
        output_channels = capture_plan.output_channels,
        capture_sample_rate = capture_plan.capture_sample_rate,
        output_sample_rate = capture_plan.output_sample_rate,
        sample_format = ?capture_plan.sample_format,
        downmix_to_mono = capture_plan.downmix_to_mono,
        resample_to_output_rate = capture_plan.resample_to_output_rate,
        selection_reason = capture_plan.selection_reason,
        device_reason = describe_device_preference(&device_name, default_device_name.as_deref()),
        selected_device_score = robo_rover_lib::device_preference_score(&device_name),
        "Using audio input device"
    );
    if capture_plan.resample_to_output_rate {
        tracing::warn!(
            device = %device_name,
            capture_sample_rate = capture_plan.capture_sample_rate,
            output_sample_rate = capture_plan.output_sample_rate,
            selection_reason = capture_plan.selection_reason,
            resample_mode = if capture_plan.capture_sample_rate % capture_plan.output_sample_rate == 0 {
                "integer-average-downsample"
            } else {
                "linear-fallback-resample"
            },
            "Audio capture is using native-rate fallback with runtime resampling"
        );
    }

    let stream = match capture_plan.sample_format {
        SampleFormat::F32 => build_input_stream_for_format::<f32>(
            &device,
            &capture_plan,
            producer,
            rejected_samples,
        )?,
        SampleFormat::I16 => build_input_stream_for_format::<i16>(
            &device,
            &capture_plan,
            producer,
            rejected_samples,
        )?,
        SampleFormat::U16 => build_input_stream_for_format::<u16>(
            &device,
            &capture_plan,
            producer,
            rejected_samples,
        )?,
        sample_format => {
            return Err(eyre::eyre!(
                "Unsupported input sample format {:?} for device '{}'",
                sample_format,
                device_name
            ));
        }
    };
    stream.play()?;
    Ok(stream)
}

fn auto_select_device_index(
    devices: &[(String, cpal::Device)],
    available_names: &[String],
    default_device_name: Option<&str>,
) -> Result<usize> {
    let selected_name = select_preferred_device_name(available_names, default_device_name)
        .ok_or_else(|| eyre::eyre!("No preferred audio input device found"))?;
    Ok(devices
        .iter()
        .position(|(name, _)| *name == selected_name)
        .unwrap_or(0))
}

fn build_input_stream_for_format<T>(
    device: &cpal::Device,
    capture_plan: &InputCapturePlan,
    producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    rejected_samples: Arc<AtomicU64>,
) -> Result<Stream>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    let mut transform = StreamSampleTransform::new(capture_plan);
    Ok(device.build_input_stream(
        &capture_plan.stream_config(),
        move |data: &[T], _: &_| {
            let output = transform.process(data);
            if let Ok(mut prod) = producer.lock() {
                let rejected = write_samples(output, &mut prod);
                rejected_samples.fetch_add(rejected as u64, Ordering::Relaxed);
            } else {
                rejected_samples.fetch_add(transform.last_output_len as u64, Ordering::Relaxed);
            }
        },
        |err| tracing::error!("Audio stream error: {}", err),
        None,
    )?)
}

fn convert_input_samples<T>(input: &[T], input_channels: u16, output_channels: u16) -> Vec<f32>
where
    T: Sample,
    f32: FromSample<T>,
{
    if input_channels == output_channels {
        return input
            .iter()
            .map(|sample| f32::from_sample(*sample))
            .collect();
    }

    if output_channels == 1 && input_channels > 1 {
        return input
            .chunks_exact(input_channels as usize)
            .map(|frame| {
                frame
                    .iter()
                    .map(|sample| f32::from_sample(*sample))
                    .sum::<f32>()
                    / frame.len() as f32
            })
            .collect();
    }

    Vec::new()
}

fn write_samples(
    samples: impl IntoIterator<Item = f32>,
    producer: &mut ringbuf::HeapProd<f32>,
) -> usize {
    let mut rejected = 0;
    for sample in samples {
        if producer.try_push(sample).is_err() {
            rejected += 1;
        }
    }
    rejected
}

fn current_time_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

struct StreamSampleTransform {
    input_channels: u16,
    output_channels: u16,
    resampler: Option<FrameResampler>,
    last_output_len: usize,
}

impl StreamSampleTransform {
    fn new(plan: &InputCapturePlan) -> Self {
        Self {
            input_channels: plan.capture_channels,
            output_channels: plan.output_channels,
            resampler: plan.resample_to_output_rate.then(|| {
                FrameResampler::new(
                    plan.output_channels as usize,
                    plan.capture_sample_rate,
                    plan.output_sample_rate,
                )
            }),
            last_output_len: 0,
        }
    }

    fn process<T>(&mut self, input: &[T]) -> Vec<f32>
    where
        T: Sample,
        f32: FromSample<T>,
    {
        let converted = convert_input_samples(input, self.input_channels, self.output_channels);
        let output = if let Some(resampler) = self.resampler.as_mut() {
            resampler.process(&converted)
        } else {
            converted
        };
        self.last_output_len = output.len();
        output
    }
}

struct FrameResampler {
    channels: usize,
    input_rate: u32,
    output_rate: u32,
    integer_downsample_factor: Option<usize>,
    frame_position: f64,
    buffer: Vec<f32>,
}

impl FrameResampler {
    fn new(channels: usize, input_rate: u32, output_rate: u32) -> Self {
        let integer_downsample_factor = (input_rate > output_rate && input_rate % output_rate == 0)
            .then_some((input_rate / output_rate) as usize);
        Self {
            channels,
            input_rate,
            output_rate,
            integer_downsample_factor,
            frame_position: 0.0,
            buffer: Vec::new(),
        }
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.input_rate == self.output_rate || self.channels == 0 {
            return input.to_vec();
        }

        if let Some(factor) = self.integer_downsample_factor {
            return self.process_integer_downsample(input, factor);
        }

        self.buffer.extend_from_slice(input);
        let available_frames = self.buffer.len() / self.channels;
        let mut output = Vec::new();
        let frame_step = self.input_rate as f64 / self.output_rate as f64;

        while self.frame_position + 1.0 < available_frames as f64 {
            let frame_index = self.frame_position.floor() as usize;
            let next_frame_index = frame_index + 1;
            let interpolation = (self.frame_position - frame_index as f64) as f32;
            for channel in 0..self.channels {
                let first = self.buffer[frame_index * self.channels + channel];
                let second = self.buffer[next_frame_index * self.channels + channel];
                output.push(first + (second - first) * interpolation);
            }
            self.frame_position += frame_step;
        }

        let consumed_frames = self.frame_position.floor() as usize;
        if consumed_frames > 0 {
            let consumed_samples = consumed_frames * self.channels;
            self.buffer.drain(..consumed_samples);
            self.frame_position -= consumed_frames as f64;
        }

        output
    }

    fn process_integer_downsample(&mut self, input: &[f32], factor: usize) -> Vec<f32> {
        self.buffer.extend_from_slice(input);
        let frames_per_output = factor * self.channels;
        let available_outputs = self.buffer.len() / frames_per_output;
        let mut output = Vec::with_capacity(available_outputs * self.channels);

        for output_index in 0..available_outputs {
            let start = output_index * frames_per_output;
            for channel in 0..self.channels {
                let mut sum = 0.0;
                for frame_offset in 0..factor {
                    sum += self.buffer[start + frame_offset * self.channels + channel];
                }
                output.push(sum / factor as f32);
            }
        }

        let consumed_samples = available_outputs * frames_per_output;
        if consumed_samples > 0 {
            self.buffer.drain(..consumed_samples);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_samples_reports_rejected_samples() {
        let ring = HeapRb::<f32>::new(2);
        let (mut producer, _consumer) = ring.split();

        assert_eq!(write_samples([0.1, 0.2, 0.3], &mut producer), 1);
    }

    #[test]
    fn convert_input_samples_downmixes_stereo_input_to_mono() {
        assert_eq!(
            convert_input_samples(&[0.4, 0.2, -0.2, -0.6], 2, 1),
            vec![0.3, -0.4]
        );
    }

    #[test]
    fn frame_resampler_downsamples_native_rate_to_requested_rate() {
        let mut resampler = FrameResampler::new(1, 48_000, 16_000);
        let output = resampler.process(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(output, vec![1.0, 4.0]);
    }

    #[test]
    fn preflight_probe_classifies_silent_audio() {
        let mut probe = PreflightSignalProbe::new(4, 1, 1000);
        let summary = analyze_signal(&[0.0, 0.0, 0.0, 0.0], SILENCE_THRESHOLD);
        probe.observe_summary(&summary);
        probe.log_if_ready();

        assert!(probe.is_emitted());
    }

    #[test]
    fn suppression_flushes_queued_and_partial_microphone_samples() {
        let ring = HeapRb::<f32>::new(4);
        let (mut producer, consumer) = ring.split();
        producer.try_push(0.1).unwrap();
        producer.try_push(0.2).unwrap();
        let consumer = Arc::new(Mutex::new(consumer));
        let mut partial = vec![0.3];

        clear_capture_buffers(&consumer, &mut partial);

        assert!(partial.is_empty());
        assert_eq!(consumer.lock().unwrap().occupied_len(), 0);
    }
}
