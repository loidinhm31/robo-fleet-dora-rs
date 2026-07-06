use std::time::{Duration, Instant};

use eyre::{eyre, Result};
use robo_rover_lib::TtsPriority;
use serde::Serialize;

pub const TTS_SAMPLE_RATE: u32 = 44_100;
pub const TTS_FRAME_SAMPLES_20_MS: usize = 882;

#[derive(Debug, Clone)]
pub struct TtsAudioChunk {
    pub command_id: String,
    pub priority: TtsPriority,
    pub frame_id: u64,
    pub timestamp_ms: u64,
    pub samples: Vec<f32>,
}

impl TtsAudioChunk {
    fn duration(&self) -> Duration {
        samples_duration(self.samples.len(), TTS_SAMPLE_RATE)
    }
}

pub trait Clock {
    fn now(&self) -> Duration;
}

#[derive(Debug, Clone)]
pub struct InstantClock {
    started: Instant,
}

impl Default for InstantClock {
    fn default() -> Self {
        Self {
            started: Instant::now(),
        }
    }
}

impl Clock for InstantClock {
    fn now(&self) -> Duration {
        self.started.elapsed()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PacingSnapshot {
    pub generated_frames: u64,
    pub generated_samples: u64,
    pub emitted_frames: u64,
    pub emitted_samples: u64,
    pub pending_depth: u64,
    pub pacing_lag_ms: u64,
}

#[derive(Debug)]
pub struct DueChunk {
    pub chunk: TtsAudioChunk,
    pub lag: Duration,
}

#[derive(Debug)]
pub struct TtsPacer<C> {
    clock: C,
    command_id: String,
    expected_frame_id: u64,
    pending: Option<TtsAudioChunk>,
    media_origin: Duration,
    next_deadline: Duration,
    emitted_media: Duration,
    final_partial_seen: bool,
    generated_frames: u64,
    generated_samples: u64,
    emitted_frames: u64,
    emitted_samples: u64,
    max_pacing_lag: Duration,
}

impl<C: Clock> TtsPacer<C> {
    pub fn new(command_id: impl Into<String>, clock: C) -> Self {
        let now = clock.now();
        Self {
            clock,
            command_id: command_id.into(),
            expected_frame_id: 0,
            pending: None,
            media_origin: now,
            next_deadline: now,
            emitted_media: Duration::ZERO,
            final_partial_seen: false,
            generated_frames: 0,
            generated_samples: 0,
            emitted_frames: 0,
            emitted_samples: 0,
            max_pacing_lag: Duration::ZERO,
        }
    }

    pub fn command_id(&self) -> &str {
        &self.command_id
    }

    pub fn is_full(&self) -> bool {
        self.pending.is_some()
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn timeout_until_due(&self, ceiling: Duration) -> Duration {
        if self.pending.is_none() {
            return ceiling;
        }
        let now = self.clock.now();
        if now >= self.next_deadline {
            Duration::ZERO
        } else {
            ceiling.min(self.next_deadline - now)
        }
    }

    pub fn accept(&mut self, chunk: TtsAudioChunk) -> Result<()> {
        if self.pending.is_some() {
            return Err(eyre!("TTS pacer pending slot is full"));
        }
        if chunk.command_id != self.command_id {
            return Err(eyre!("TTS chunk belongs to a stale command"));
        }
        if chunk.frame_id != self.expected_frame_id {
            return Err(eyre!(
                "TTS frame sequence gap: expected {}, got {}",
                self.expected_frame_id,
                chunk.frame_id
            ));
        }
        if chunk.samples.is_empty() {
            return Err(eyre!("TTS chunk is empty"));
        }
        if chunk.samples.len() > TTS_FRAME_SAMPLES_20_MS {
            return Err(eyre!(
                "TTS chunk has {} samples, maximum is {}",
                chunk.samples.len(),
                TTS_FRAME_SAMPLES_20_MS
            ));
        }
        if self.final_partial_seen {
            return Err(eyre!("TTS chunk arrived after final partial frame"));
        }
        if chunk.samples.len() < TTS_FRAME_SAMPLES_20_MS {
            self.final_partial_seen = true;
        }

        self.generated_frames = self.generated_frames.saturating_add(1);
        self.generated_samples = self
            .generated_samples
            .saturating_add(chunk.samples.len().try_into().unwrap_or(u64::MAX));
        self.expected_frame_id = self.expected_frame_id.saturating_add(1);
        self.pending = Some(chunk);
        Ok(())
    }

    pub fn pop_due(&mut self) -> Option<DueChunk> {
        let now = self.clock.now();
        if self.pending.is_none() || now < self.next_deadline {
            return None;
        }

        let chunk = self.pending.take()?;
        let duration = chunk.duration();
        let lag = now.saturating_sub(self.next_deadline);
        self.max_pacing_lag = self.max_pacing_lag.max(lag);
        self.emitted_frames = self.emitted_frames.saturating_add(1);
        self.emitted_samples = self
            .emitted_samples
            .saturating_add(chunk.samples.len().try_into().unwrap_or(u64::MAX));
        self.emitted_media = self.emitted_media.saturating_add(duration);

        let cumulative_deadline = self.media_origin.saturating_add(self.emitted_media);
        let no_burst_deadline = now.saturating_add(duration);
        self.next_deadline = cumulative_deadline.max(no_burst_deadline);

        Some(DueChunk { chunk, lag })
    }

    pub fn snapshot(&self) -> PacingSnapshot {
        PacingSnapshot {
            generated_frames: self.generated_frames,
            generated_samples: self.generated_samples,
            emitted_frames: self.emitted_frames,
            emitted_samples: self.emitted_samples,
            pending_depth: u64::from(self.pending.is_some()),
            pacing_lag_ms: self
                .max_pacing_lag
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
        }
    }

    pub fn clear_pending(&mut self) {
        self.pending = None;
    }
}

fn samples_duration(samples: usize, sample_rate: u32) -> Duration {
    Duration::from_secs_f64(samples as f64 / f64::from(sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    #[derive(Clone, Default)]
    struct FakeClock {
        now: Rc<Cell<Duration>>,
    }

    impl FakeClock {
        fn advance(&self, duration: Duration) {
            self.now.set(self.now.get() + duration);
        }
    }

    impl Clock for FakeClock {
        fn now(&self) -> Duration {
            self.now.get()
        }
    }

    fn chunk(frame_id: u64, samples: usize) -> TtsAudioChunk {
        TtsAudioChunk {
            command_id: "cmd".to_string(),
            priority: TtsPriority::Normal,
            frame_id,
            timestamp_ms: frame_id,
            samples: vec![0.0; samples],
        }
    }

    #[test]
    fn first_chunk_is_due_immediately_then_paces_twenty_ms() {
        let clock = FakeClock::default();
        let mut pacer = TtsPacer::new("cmd", clock.clone());

        pacer.accept(chunk(0, TTS_FRAME_SAMPLES_20_MS)).unwrap();
        assert!(pacer.pop_due().is_some());

        pacer.accept(chunk(1, TTS_FRAME_SAMPLES_20_MS)).unwrap();
        assert!(pacer.pop_due().is_none());
        clock.advance(Duration::from_millis(19));
        assert!(pacer.pop_due().is_none());
        clock.advance(Duration::from_millis(1));
        assert!(pacer.pop_due().is_some());
    }

    #[test]
    fn scheduler_delay_does_not_create_catch_up_burst() {
        let clock = FakeClock::default();
        let mut pacer = TtsPacer::new("cmd", clock.clone());

        pacer.accept(chunk(0, TTS_FRAME_SAMPLES_20_MS)).unwrap();
        assert!(pacer.pop_due().is_some());

        pacer.accept(chunk(1, TTS_FRAME_SAMPLES_20_MS)).unwrap();
        clock.advance(Duration::from_millis(70));
        assert!(pacer.pop_due().is_some());

        pacer.accept(chunk(2, TTS_FRAME_SAMPLES_20_MS)).unwrap();
        assert!(pacer.pop_due().is_none());
        clock.advance(Duration::from_millis(19));
        assert!(pacer.pop_due().is_none());
        clock.advance(Duration::from_millis(1));
        assert!(pacer.pop_due().is_some());
    }

    #[test]
    fn rejects_sequence_gap_and_post_partial_frames() {
        let clock = FakeClock::default();
        let mut pacer = TtsPacer::new("cmd", clock);

        let gap = pacer.accept(chunk(1, TTS_FRAME_SAMPLES_20_MS)).unwrap_err();
        assert!(gap.to_string().contains("sequence gap"));

        pacer.accept(chunk(0, 100)).unwrap();
        assert!(pacer.pop_due().is_some());
        let after_partial = pacer.accept(chunk(1, TTS_FRAME_SAMPLES_20_MS)).unwrap_err();
        assert!(after_partial.to_string().contains("final partial"));
    }

    #[test]
    fn reports_generated_emitted_and_pending_accounting() {
        let clock = FakeClock::default();
        let mut pacer = TtsPacer::new("cmd", clock);
        pacer.accept(chunk(0, TTS_FRAME_SAMPLES_20_MS)).unwrap();
        assert_eq!(pacer.snapshot().pending_depth, 1);
        assert!(pacer.pop_due().is_some());
        let snapshot = pacer.snapshot();
        assert_eq!(snapshot.generated_frames, 1);
        assert_eq!(snapshot.emitted_frames, 1);
        assert_eq!(snapshot.generated_samples, 882);
        assert_eq!(snapshot.emitted_samples, 882);
        assert_eq!(snapshot.pending_depth, 0);
    }
}
