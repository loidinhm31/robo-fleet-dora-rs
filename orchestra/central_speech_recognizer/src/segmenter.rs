use crate::{config::SttConfig, native};
use eyre::Result;
use sherpa_onnx::VoiceActivityDetector;
use std::sync::Arc;

pub trait Segmenter: Send {
    fn accept(&mut self, samples: &[f32]);
    fn drain(&mut self) -> Vec<Vec<f32>>;
    fn flush(&mut self) -> Vec<Vec<f32>>;
    fn reset(&mut self);
}

pub type SegmenterFactory = Arc<dyn Fn() -> Result<Box<dyn Segmenter>> + Send + Sync>;

pub struct SherpaSegmenter {
    vad: VoiceActivityDetector,
}

impl SherpaSegmenter {
    pub fn new(vad: VoiceActivityDetector) -> Self {
        Self { vad }
    }
}

impl Segmenter for SherpaSegmenter {
    fn accept(&mut self, samples: &[f32]) {
        self.vad.accept_waveform(samples);
    }

    fn drain(&mut self) -> Vec<Vec<f32>> {
        let mut segments = Vec::new();
        while let Some(segment) = self.vad.front() {
            segments.push(segment.samples().to_vec());
            self.vad.pop();
        }
        segments
    }

    fn flush(&mut self) -> Vec<Vec<f32>> {
        self.vad.flush();
        self.drain()
    }

    fn reset(&mut self) {
        self.vad.reset();
        self.vad.clear();
    }
}

pub fn sherpa_factory(config: SttConfig) -> SegmenterFactory {
    Arc::new(move || {
        let vad = native::create_vad(&config)?;
        Ok(Box::new(SherpaSegmenter::new(vad)))
    })
}
