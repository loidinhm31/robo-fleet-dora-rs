use eyre::{eyre, Result};
use rubato::{
    audioadapter_buffers::direct::SequentialSlice, calculate_cutoff, Async, FixedAsync, Resampler,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

const MAX_CHUNK_FRAMES: usize = 4_096;
const SINC_LENGTH: usize = 128;

pub struct SourceResampler {
    input_rate: u32,
    output_rate: u32,
    inner: Option<Async<f32>>,
    output: Vec<f32>,
    delay_frames: usize,
    delay_remaining: usize,
    input_frames: usize,
    emitted_frames: usize,
    has_data: bool,
}

impl SourceResampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Result<Self> {
        if input_rate == 0 || output_rate == 0 {
            return Err(eyre!("sample rates must be non-zero"));
        }
        if input_rate == output_rate {
            return Ok(Self {
                input_rate,
                output_rate,
                inner: None,
                output: Vec::new(),
                delay_frames: 0,
                delay_remaining: 0,
                input_frames: 0,
                emitted_frames: 0,
                has_data: false,
            });
        }

        let window = WindowFunction::BlackmanHarris2;
        let parameters = SincInterpolationParameters {
            sinc_len: SINC_LENGTH,
            f_cutoff: calculate_cutoff(SINC_LENGTH, window),
            oversampling_factor: 128,
            interpolation: SincInterpolationType::Cubic,
            window,
        };
        let inner = Async::<f32>::new_sinc(
            output_rate as f64 / input_rate as f64,
            1.0,
            &parameters,
            MAX_CHUNK_FRAMES,
            1,
            FixedAsync::Input,
        )?;
        let delay_frames = inner.output_delay();
        let output = vec![0.0; inner.output_frames_max()];
        Ok(Self {
            input_rate,
            output_rate,
            inner: Some(inner),
            output,
            delay_frames,
            delay_remaining: delay_frames,
            input_frames: 0,
            emitted_frames: 0,
            has_data: false,
        })
    }

    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.inner.is_none() {
            self.has_data |= !input.is_empty();
            self.input_frames += input.len();
            self.emitted_frames += input.len();
            return Ok(input.to_vec());
        }
        let mut result = Vec::with_capacity(
            (input.len() as u64 * u64::from(self.output_rate) / u64::from(self.input_rate))
                as usize
                + 2,
        );
        for chunk in input.chunks(MAX_CHUNK_FRAMES) {
            self.process_chunk(chunk, &mut result)?;
        }
        self.input_frames += input.len();
        self.emitted_frames += result.len();
        self.has_data |= !input.is_empty();
        Ok(result)
    }

    pub fn flush(&mut self) -> Result<Vec<f32>> {
        if !self.has_data || self.inner.is_none() {
            self.reset();
            return Ok(Vec::new());
        }
        let numerator = (self.input_frames as u128) * u128::from(self.output_rate);
        let denominator = u128::from(self.input_rate);
        let expected_frames = usize::try_from(numerator.div_ceil(denominator))?;
        let needed_frames = expected_frames.saturating_sub(self.emitted_frames);
        let mut result = Vec::with_capacity(needed_frames);
        let input_frames = self
            .inner
            .as_ref()
            .map(Resampler::input_frames_next)
            .unwrap_or(1)
            .max(1);
        let output_frames_to_pump = needed_frames + self.delay_remaining;
        let estimated_input = (output_frames_to_pump as u128 * u128::from(self.input_rate))
            .div_ceil(u128::from(self.output_rate))
            + SINC_LENGTH as u128;
        let max_attempts =
            usize::try_from(estimated_input.div_ceil(input_frames as u128))?.saturating_add(2);
        let mut attempts = 0;
        while result.len() < needed_frames && attempts < max_attempts {
            let zeros = vec![0.0; input_frames];
            let mut produced = Vec::new();
            self.process_chunk(&zeros, &mut produced)?;
            let remaining = needed_frames - result.len();
            result.extend(produced.into_iter().take(remaining));
            attempts += 1;
        }
        if result.len() != needed_frames {
            return Err(eyre!(
                "resampler flush did not produce the expected duration"
            ));
        }
        self.reset();
        Ok(result)
    }

    pub fn reset(&mut self) {
        if let Some(inner) = self.inner.as_mut() {
            inner.reset();
        }
        self.delay_remaining = self.delay_frames;
        self.input_frames = 0;
        self.emitted_frames = 0;
        self.has_data = false;
    }

    fn process_chunk(&mut self, input: &[f32], result: &mut Vec<f32>) -> Result<()> {
        let inner = self.inner.as_mut().expect("resampling path requires inner");
        inner.set_chunk_size(input.len())?;
        let input_adapter = SequentialSlice::new(input, 1, input.len())?;
        let output_len = self.output.len();
        let mut output_adapter = SequentialSlice::new_mut(&mut self.output, 1, output_len)?;
        let (_, produced) = inner.process_into_buffer(&input_adapter, &mut output_adapter, None)?;
        let skip = self.delay_remaining.min(produced);
        self.delay_remaining -= skip;
        result.extend_from_slice(&self.output[skip..produced]);
        Ok(())
    }
}

#[cfg(test)]
#[path = "resampler-tests.rs"]
mod tests;
