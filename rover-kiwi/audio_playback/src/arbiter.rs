use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eyre::Result;

use crate::buffers::PlaybackBuffers;
use crate::playback_event::ArbiterEvent;
use crate::protocol::{AudioSource, SourceFrame};
use crate::resampler::SourceResampler;
use crate::tts_arbiter::{TtsArbiter, TtsArbiterStats};

const WALKIE_HOLD: Duration = Duration::from_millis(250);

pub struct SourceArbiter {
    output_rate: u32,
    buffers: Arc<PlaybackBuffers>,
    tts: TtsArbiter,
    walkie_resampler: Option<SourceResampler>,
    walkie_deadline: Option<Instant>,
}

impl SourceArbiter {
    pub fn new(
        output_rate: u32,
        buffers: Arc<PlaybackBuffers>,
        tts_stall_timeout: Duration,
    ) -> Self {
        Self {
            output_rate,
            buffers: buffers.clone(),
            tts: TtsArbiter::new(output_rate, buffers, tts_stall_timeout),
            walkie_resampler: None,
            walkie_deadline: None,
        }
    }

    pub fn accept(&mut self, frame: SourceFrame, now: Instant) -> Result<Option<ArbiterEvent>> {
        match frame.source {
            AudioSource::Tts => self.tts.accept(frame, now),
            AudioSource::Walkie => self.accept_walkie(frame, now),
        }
    }

    pub fn tick(&mut self, now: Instant) -> Option<ArbiterEvent> {
        if self.walkie_deadline.is_some_and(|deadline| now >= deadline) {
            if let Some(resampler) = self.walkie_resampler.as_mut() {
                resampler.reset();
            }
            self.buffers.finish_walkie();
            self.walkie_deadline = None;
            return Some(ArbiterEvent::WalkieEnded);
        }
        self.tts.tick(now)
    }

    pub fn finish_tts(&mut self, command_id: &str, now: Instant) -> Result<Option<ArbiterEvent>> {
        self.tts.finish(command_id, now)
    }

    pub fn abort_tts(&mut self, command_id: &str) {
        self.tts.abort(command_id);
        self.tts.prune_command_ids();
    }

    pub fn owns_tts(&self, command_id: &str) -> bool {
        self.tts.owns(command_id)
    }

    pub fn fail_playback(&mut self) -> Option<ArbiterEvent> {
        let event = self.tts.fail();
        self.buffers.clear_all();
        if let Some(resampler) = self.walkie_resampler.as_mut() {
            resampler.reset();
        }
        self.walkie_deadline = None;
        event
    }

    /// Cancels any accepted TTS before the output stream is dropped. The
    /// caller owns the terminal result; this method only removes local state.
    pub fn interrupt_for_lifecycle(&mut self) -> Option<String> {
        let command_id = self.tts.preempt();
        self.buffers.clear_all();
        if let Some(resampler) = self.walkie_resampler.as_mut() {
            resampler.reset();
        }
        self.walkie_deadline = None;
        command_id
    }

    pub fn prune_command_ids(&mut self) {
        self.tts.prune_command_ids();
    }

    pub fn command_ids(&self) -> &BTreeMap<u64, String> {
        self.tts.command_ids()
    }

    pub fn tts_stats(&self) -> TtsArbiterStats {
        self.tts.stats()
    }

    fn accept_walkie(&mut self, frame: SourceFrame, now: Instant) -> Result<Option<ArbiterEvent>> {
        let started = !self.buffers.walkie_is_active();
        let interrupted = started.then(|| self.tts.preempt()).flatten();
        if started {
            self.buffers.preempt_tts();
        }
        let rate_changed = self
            .walkie_resampler
            .as_ref()
            .is_some_and(|resampler| resampler.input_rate() != frame.sample_rate);
        if rate_changed {
            if let Some(resampler) = self.walkie_resampler.as_mut() {
                self.buffers.enqueue_walkie(&resampler.flush()?);
            }
        }
        if self.walkie_resampler.is_none() || rate_changed {
            self.walkie_resampler =
                Some(SourceResampler::new(frame.sample_rate, self.output_rate)?);
        }
        let output = self
            .walkie_resampler
            .as_mut()
            .expect("walkie resampler initialized")
            .process(&frame.samples)?;
        self.buffers.enqueue_walkie(&output);
        self.walkie_deadline = Some(now + WALKIE_HOLD);
        Ok(started.then_some(ArbiterEvent::WalkieStarted { interrupted }))
    }
}
