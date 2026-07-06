use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eyre::Result;

use crate::buffers::{PlaybackBuffers, SOURCE_TTS};
use crate::playback_event::ArbiterEvent;
use crate::protocol::SourceFrame;
use crate::resampler::SourceResampler;

const MAX_PENDING_FRAMES: usize = 3;

struct AwaitingPlayback {
    command_id: String,
    token: u64,
    consumed_target: u64,
}

struct PendingFrame {
    command_id: String,
    token: u64,
    samples: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TtsArbiterStats {
    pub pending_frames: usize,
    pub pending_samples: usize,
    pub pending_overflows: u64,
    pub pending_frames_cleared: u64,
    pub pending_samples_cleared: u64,
    pub stall_failures: u64,
    pub sequence_failures: u64,
}

pub struct TtsArbiter {
    output_rate: u32,
    buffers: Arc<PlaybackBuffers>,
    resampler: Option<SourceResampler>,
    command_ids: BTreeMap<u64, String>,
    current: Option<(String, u64)>,
    awaiting: Option<AwaitingPlayback>,
    blocked_command: Option<String>,
    pending: VecDeque<PendingFrame>,
    stall_started_at: Option<Instant>,
    stall_timeout: Duration,
    expected_frame_id: Option<u64>,
    synthesis_completed: bool,
    pending_overflows: u64,
    pending_frames_cleared: u64,
    pending_samples_cleared: u64,
    stall_failures: u64,
    sequence_failures: u64,
    next_token: u64,
}

impl TtsArbiter {
    pub fn new(output_rate: u32, buffers: Arc<PlaybackBuffers>, stall_timeout: Duration) -> Self {
        Self {
            output_rate,
            buffers,
            resampler: None,
            command_ids: BTreeMap::new(),
            current: None,
            awaiting: None,
            blocked_command: None,
            pending: VecDeque::new(),
            stall_started_at: None,
            stall_timeout,
            expected_frame_id: None,
            synthesis_completed: false,
            pending_overflows: 0,
            pending_frames_cleared: 0,
            pending_samples_cleared: 0,
            stall_failures: 0,
            sequence_failures: 0,
            next_token: 1,
        }
    }

    pub fn accept(&mut self, frame: SourceFrame, now: Instant) -> Result<Option<ArbiterEvent>> {
        if let Some(event) = self.tick(now) {
            return Ok(Some(event));
        }
        if self.buffers.walkie_is_active() {
            return Ok(Some(ArbiterEvent::TtsRejectedWhileWalkie));
        }
        let command_id = frame
            .command_id
            .expect("validated TTS frame has command ID");
        if self.blocked_command.as_deref() == Some(&command_id) {
            return Ok(None);
        }
        self.blocked_command = None;
        let command_changed = self
            .current
            .as_ref()
            .is_some_and(|(current, _)| current != &command_id);
        let rate_changed = self
            .resampler
            .as_ref()
            .is_some_and(|resampler| resampler.input_rate() != frame.sample_rate);
        if command_changed || (rate_changed && self.current.is_some()) {
            return Ok(Some(self.fail_current(command_id)));
        }
        if self.resampler.is_none() || rate_changed {
            self.resampler = Some(SourceResampler::new(frame.sample_rate, self.output_rate)?);
        }
        let token = match self.current.as_ref() {
            Some((current, token)) if current == &command_id => *token,
            _ => self.begin(command_id.clone()),
        };
        if !self.accepts_frame_id(frame.frame_id) {
            self.sequence_failures = self.sequence_failures.saturating_add(1);
            return Ok(Some(self.fail_current(command_id)));
        }
        let output = self
            .resampler
            .as_mut()
            .expect("TTS resampler initialized")
            .process(&frame.samples)?;
        Ok(self.enqueue_or_pending(output, token, &command_id, now))
    }

    pub fn finish(&mut self, command_id: &str, now: Instant) -> Result<Option<ArbiterEvent>> {
        if let Some(event) = self.tick(now) {
            return Ok(Some(event));
        }
        if self.blocked_command.as_deref() == Some(command_id) {
            self.blocked_command = None;
            return Ok(None);
        }
        if self
            .current
            .as_ref()
            .is_none_or(|(current, _)| current != command_id)
        {
            return Ok(None);
        }
        let token = self.current.as_ref().expect("current TTS exists").1;
        let tail = if let Some(resampler) = self.resampler.as_mut() {
            resampler.flush()?
        } else {
            Vec::new()
        };
        self.synthesis_completed = true;
        if let Some(event) = self.enqueue_or_pending(tail, token, command_id, now) {
            return Ok(Some(event));
        }
        self.finalize_completed_if_drained();
        Ok(None)
    }

    pub fn tick(&mut self, now: Instant) -> Option<ArbiterEvent> {
        if let Some(started) = self.stall_started_at {
            if now.duration_since(started) >= self.stall_timeout {
                self.stall_failures = self.stall_failures.saturating_add(1);
                let fallback = self
                    .current
                    .as_ref()
                    .map(|(command, _)| command.clone())
                    .unwrap_or_default();
                return Some(self.fail_current(fallback));
            }
        }
        self.drain_pending();
        self.finalize_completed_if_drained();
        let finished = self.awaiting.as_ref().is_some_and(|pending| {
            self.buffers.tts_retired_total() >= pending.consumed_target
                && self.buffers.tts_is_empty()
        });
        if !finished {
            return None;
        }
        let completed = self.awaiting.take().expect("finished playback is pending");
        self.command_ids.remove(&completed.token);
        Some(ArbiterEvent::TtsPlaybackCompleted {
            command_id: completed.command_id,
        })
    }

    pub fn preempt(&mut self) -> Option<String> {
        let command_id = self.active_command_id();
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.reset();
        }
        self.current = None;
        self.awaiting = None;
        self.blocked_command = None;
        self.clear_pending();
        self.stall_started_at = None;
        self.expected_frame_id = None;
        self.synthesis_completed = false;
        command_id
    }

    pub fn fail(&mut self) -> Option<ArbiterEvent> {
        let command_id = self.preempt();
        command_id.map(|command_id| ArbiterEvent::TtsPlaybackFailed { command_id })
    }

    pub fn abort(&mut self, command_id: &str) {
        let matches_current = self
            .current
            .as_ref()
            .is_some_and(|(current, _)| current == command_id);
        let matches_awaiting = self
            .awaiting
            .as_ref()
            .is_some_and(|pending| pending.command_id == command_id);
        let matches_pending = self
            .pending
            .iter()
            .any(|pending| pending.command_id == command_id);
        if matches_current || matches_awaiting || matches_pending {
            self.preempt();
            self.buffers.clear_tts();
        } else if self.blocked_command.as_deref() == Some(command_id) {
            self.blocked_command = None;
        }
    }

    pub fn prune_command_ids(&mut self) {
        if self.current.is_none()
            && self.awaiting.is_none()
            && self.pending.is_empty()
            && self.buffers.tts_is_empty()
        {
            self.command_ids.clear();
        }
    }

    pub fn command_ids(&self) -> &BTreeMap<u64, String> {
        &self.command_ids
    }

    pub fn owns(&self, command_id: &str) -> bool {
        self.current
            .as_ref()
            .is_some_and(|(current, _)| current == command_id)
            || self
                .awaiting
                .as_ref()
                .is_some_and(|pending| pending.command_id == command_id)
            || self
                .pending
                .iter()
                .any(|pending| pending.command_id == command_id)
            || self.blocked_command.as_deref() == Some(command_id)
    }

    pub fn stats(&self) -> TtsArbiterStats {
        TtsArbiterStats {
            pending_frames: self.pending.len(),
            pending_samples: self.pending.iter().map(|frame| frame.samples.len()).sum(),
            pending_overflows: self.pending_overflows,
            pending_frames_cleared: self.pending_frames_cleared,
            pending_samples_cleared: self.pending_samples_cleared,
            stall_failures: self.stall_failures,
            sequence_failures: self.sequence_failures,
        }
    }

    fn enqueue_or_pending(
        &mut self,
        output: Vec<f32>,
        token: u64,
        command_id: &str,
        now: Instant,
    ) -> Option<ArbiterEvent> {
        if !self.pending.is_empty() && !output.is_empty() {
            return self.push_pending(output, token, command_id, now);
        }
        if output.is_empty() || self.buffers.try_enqueue_tts_frame(&output, token) {
            if self.pending.is_empty() {
                self.stall_started_at = None;
            }
            return None;
        }
        self.push_pending(output, token, command_id, now)
    }

    fn push_pending(
        &mut self,
        output: Vec<f32>,
        token: u64,
        command_id: &str,
        now: Instant,
    ) -> Option<ArbiterEvent> {
        if self.pending.len() >= MAX_PENDING_FRAMES {
            self.pending_overflows = self.pending_overflows.saturating_add(1);
            return Some(self.fail_current(command_id.to_owned()));
        }
        self.pending.push_back(PendingFrame {
            command_id: command_id.to_owned(),
            token,
            samples: output,
        });
        self.stall_started_at.get_or_insert(now);
        None
    }

    fn drain_pending(&mut self) {
        while let Some(frame) = self.pending.front() {
            if !self
                .buffers
                .try_enqueue_tts_frame(&frame.samples, frame.token)
            {
                break;
            }
            self.pending.pop_front();
        }
        if self.pending.is_empty() {
            self.stall_started_at = None;
        }
    }

    fn finalize_completed_if_drained(&mut self) {
        if !self.synthesis_completed || !self.pending.is_empty() {
            return;
        }
        if let Some((command_id, token)) = self.current.take() {
            self.awaiting = Some(AwaitingPlayback {
                command_id,
                token,
                consumed_target: self.buffers.tts_enqueued_total(),
            });
        }
        self.synthesis_completed = false;
    }

    fn accepts_frame_id(&mut self, frame_id: u64) -> bool {
        match self.expected_frame_id {
            None if frame_id == 0 => {
                self.expected_frame_id = Some(1);
                true
            }
            Some(expected) if frame_id == expected => {
                self.expected_frame_id = Some(expected.saturating_add(1));
                true
            }
            _ => false,
        }
    }

    fn fail_current(&mut self, fallback_id: String) -> ArbiterEvent {
        let command_id = self.active_command_id().unwrap_or(fallback_id);
        self.buffers.clear_tts();
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.reset();
        }
        self.current = None;
        self.awaiting = None;
        self.clear_pending();
        self.stall_started_at = None;
        self.expected_frame_id = None;
        self.synthesis_completed = false;
        self.blocked_command = Some(command_id.clone());
        ArbiterEvent::TtsPlaybackFailed { command_id }
    }

    fn begin(&mut self, command_id: String) -> u64 {
        let token = self.next_token;
        self.next_token = self.next_token.saturating_add(1).max(1);
        self.command_ids.insert(token, command_id.clone());
        self.current = Some((command_id, token));
        self.expected_frame_id = None;
        self.synthesis_completed = false;
        token
    }

    fn active_command_id(&self) -> Option<String> {
        let (source, token) = self.buffers.active_consumption();
        if source == SOURCE_TTS {
            return self.command_ids.get(&token).cloned();
        }
        self.current.as_ref().map(|(id, _)| id.clone()).or_else(|| {
            self.awaiting
                .as_ref()
                .map(|pending| pending.command_id.clone())
        })
    }

    fn clear_pending(&mut self) {
        self.pending_frames_cleared = self
            .pending_frames_cleared
            .saturating_add(self.pending.len() as u64);
        let samples = self
            .pending
            .iter()
            .map(|frame| frame.samples.len() as u64)
            .sum::<u64>();
        self.pending_samples_cleared = self.pending_samples_cleared.saturating_add(samples);
        self.pending.clear();
    }
}
