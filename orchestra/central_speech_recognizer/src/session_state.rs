use crate::audio_input::SourceIdentity;
use crate::config::VAD_WINDOW_SIZE;
use crate::segmenter::Segmenter;
use crate::session::{DecodeJob, FrameOutcome};
use eyre::{eyre, Result};
use sherpa_onnx::LinearResampler;

const TARGET_SAMPLE_RATE: u32 = 16_000;

pub(crate) struct Session {
    pub(crate) identity: SourceIdentity,
    pub(crate) sample_rate: u32,
    last_frame_id: Option<u64>,
    resampler: Option<LinearResampler>,
    tail: Vec<f32>,
    segmenter: Box<dyn Segmenter>,
}

impl Session {
    pub(crate) fn new(
        identity: SourceIdentity,
        sample_rate: u32,
        segmenter: Box<dyn Segmenter>,
    ) -> Result<Self> {
        let resampler = if sample_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            Some(
                LinearResampler::create(sample_rate.try_into()?, TARGET_SAMPLE_RATE as i32)
                    .ok_or_else(|| eyre!("failed to initialize browser resampler"))?,
            )
        };
        Ok(Self {
            identity,
            sample_rate,
            last_frame_id: None,
            resampler,
            tail: Vec::new(),
            segmenter,
        })
    }

    pub(crate) fn accept(&mut self, frame_id: u64, samples: &[f32]) -> FrameOutcome {
        let sequence_reset = self
            .last_frame_id
            .is_some_and(|last| last.checked_add(1) != Some(frame_id));
        if sequence_reset {
            self.reset();
        }
        self.last_frame_id = Some(frame_id);
        let normalized = self
            .resampler
            .as_ref()
            .map(|resampler| resampler.resample(samples, false))
            .unwrap_or_else(|| samples.to_vec());
        FrameOutcome {
            jobs: self.feed(&normalized),
            sequence_reset,
        }
    }

    pub(crate) fn flush(&mut self) -> Vec<DecodeJob> {
        let remainder = self
            .resampler
            .as_ref()
            .map(|resampler| resampler.resample(&[], true))
            .unwrap_or_default();
        let mut jobs = self.feed(&remainder);
        if !self.tail.is_empty() {
            self.tail.resize(VAD_WINDOW_SIZE, 0.0);
            self.segmenter.accept(&self.tail);
            self.tail.clear();
            let segments = self.segmenter.drain();
            jobs.extend(self.jobs(segments));
        }
        let segments = self.segmenter.flush();
        jobs.extend(self.jobs(segments));
        jobs
    }

    fn feed(&mut self, samples: &[f32]) -> Vec<DecodeJob> {
        self.tail.extend_from_slice(samples);
        let complete = self.tail.len() / VAD_WINDOW_SIZE * VAD_WINDOW_SIZE;
        let mut segments = Vec::new();
        for window in self.tail[..complete].chunks_exact(VAD_WINDOW_SIZE) {
            self.segmenter.accept(window);
            segments.extend(self.segmenter.drain());
        }
        self.tail.drain(..complete);
        self.jobs(segments)
    }

    fn reset(&mut self) {
        self.segmenter.reset();
        self.tail.clear();
        if let Some(resampler) = &self.resampler {
            resampler.reset();
        }
    }

    fn jobs(&self, segments: Vec<Vec<f32>>) -> Vec<DecodeJob> {
        segments
            .into_iter()
            .filter(|samples| !samples.is_empty())
            .map(|samples| DecodeJob {
                identity: self.identity.clone(),
                samples,
            })
            .collect()
    }
}
