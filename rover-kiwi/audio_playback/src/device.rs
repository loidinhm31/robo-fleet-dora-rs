use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, SupportedStreamConfig};
use eyre::{eyre, Result};

use crate::buffers::{PlaybackBuffers, SOURCE_IDLE, SOURCE_TTS};

pub struct PlaybackDevice {
    _stream: Stream,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: SampleFormat,
}

pub struct OutputPlan {
    device: cpal::Device,
    config: SupportedStreamConfig,
}

impl OutputPlan {
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate().0
    }

    pub fn open(self, buffers: Arc<PlaybackBuffers>, volume: f32) -> Result<PlaybackDevice> {
        open_output(self.device, self.config, buffers, volume)
    }
}

pub fn default_output_plan() -> Result<OutputPlan> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| eyre!("no output device available"))?;
    let config = device.default_output_config()?;
    Ok(OutputPlan { device, config })
}

fn open_output(
    device: cpal::Device,
    config: SupportedStreamConfig,
    buffers: Arc<PlaybackBuffers>,
    volume: f32,
) -> Result<PlaybackDevice> {
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();
    let sample_format = config.sample_format();
    tracing::info!(
        device = %device.name()?,
        sample_rate,
        channels,
        ?sample_format,
        "selected native audio output"
    );

    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(&device, &config, buffers, volume)?,
        SampleFormat::I16 => build_stream::<i16>(&device, &config, buffers, volume)?,
        SampleFormat::U16 => build_stream::<u16>(&device, &config, buffers, volume)?,
        format => return Err(eyre!("unsupported output sample format: {format:?}")),
    };
    stream.play()?;
    Ok(PlaybackDevice {
        _stream: stream,
        sample_rate,
        channels,
        sample_format,
    })
}

fn build_stream<T>(
    device: &cpal::Device,
    supported: &SupportedStreamConfig,
    buffers: Arc<PlaybackBuffers>,
    volume: f32,
) -> Result<Stream>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    let channels = supported.channels() as usize;
    let error_buffers = buffers.clone();
    Ok(device.build_output_stream(
        &supported.config(),
        move |output: &mut [T], _| write_output(output, channels, volume, &buffers),
        move |_| error_buffers.record_stream_error(),
        None,
    )?)
}

fn write_output<T>(output: &mut [T], channels: usize, volume: f32, buffers: &PlaybackBuffers)
where
    T: Sample + FromSample<f32>,
{
    let mut active_source = SOURCE_IDLE;
    let mut active_token = 0;
    for frame in output.chunks_mut(channels) {
        let (source, sample, token) = buffers
            .pop_for_output()
            .map(|(source, sample)| {
                let value = (sample.value * volume).clamp(-1.0, 1.0);
                (source, value, sample.token)
            })
            .unwrap_or((SOURCE_IDLE, 0.0, 0));
        if source != SOURCE_IDLE && sample.abs() > f32::EPSILON {
            active_source = source;
            active_token = if source == SOURCE_TTS { token } else { 0 };
        }
        if source != SOURCE_IDLE {
            buffers.record_monitor_sample(sample);
        }
        let converted = T::from_sample(sample);
        frame.fill(converted);
    }
    buffers.publish_consumption(active_source, active_token);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffers::{SOURCE_TTS, SOURCE_WALKIE};

    #[test]
    fn callback_duplicates_mono_and_reports_actual_consumption() {
        let buffers = PlaybackBuffers::new(8, 8);
        assert!(buffers.try_enqueue_tts_frame(&[0.25, -0.5], 9));
        let mut output = [0.0_f32; 4];

        write_output(&mut output, 2, 1.0, &buffers);

        assert_eq!(output, [0.25, 0.25, -0.5, -0.5]);
        assert_eq!(buffers.active_consumption(), (SOURCE_TTS, 9));
    }

    #[test]
    fn callback_never_mixes_tts_while_walkie_is_active() {
        let buffers = PlaybackBuffers::new(8, 8);
        assert!(buffers.try_enqueue_tts_frame(&[0.25], 1));
        buffers.enqueue_walkie(&[0.75]);
        buffers.preempt_tts();
        let mut output = [0.0_f32; 1];

        write_output(&mut output, 1, 1.0, &buffers);

        assert_eq!(output, [0.75]);
        assert_eq!(buffers.active_consumption(), (SOURCE_WALKIE, 0));
    }

    #[test]
    fn interval_activity_defers_idle_until_later_idle_callback() {
        let buffers = PlaybackBuffers::new(8, 8);
        assert!(buffers.try_enqueue_tts_frame(&[0.25], 9));
        let mut output = [0.0_f32; 1];
        write_output(&mut output, 1, 1.0, &buffers);
        write_output(&mut output, 1, 1.0, &buffers);

        assert_eq!(
            buffers.take_interval_consumption(),
            crate::buffers::ConsumptionEvent {
                source: SOURCE_TTS,
                token: 9
            }
        );
        write_output(&mut output, 1, 1.0, &buffers);
        assert_eq!(
            buffers.take_interval_consumption(),
            crate::buffers::ConsumptionEvent {
                source: SOURCE_IDLE,
                token: 0
            }
        );
    }
}
