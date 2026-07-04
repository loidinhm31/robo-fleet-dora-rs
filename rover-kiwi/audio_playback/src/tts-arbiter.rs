use std::collections::BTreeMap;
use std::sync::Arc;

use eyre::Result;

use crate::buffers::{PlaybackBuffers, SOURCE_TTS};
use crate::playback_event::ArbiterEvent;
use crate::protocol::SourceFrame;
use crate::resampler::SourceResampler;

struct AwaitingPlayback {
    command_id: String,
    token: u64,
    consumed_target: u64,
}

pub struct TtsArbiter {
    output_rate: u32,
    buffers: Arc<PlaybackBuffers>,
    resampler: Option<SourceResampler>,
    command_ids: BTreeMap<u64, String>,
    current: Option<(String, u64)>,
    awaiting: Option<AwaitingPlayback>,
    blocked_command: Option<String>,
    next_token: u64,
}

impl TtsArbiter {
    pub fn new(output_rate: u32, buffers: Arc<PlaybackBuffers>) -> Self {
        Self {
            output_rate,
            buffers,
            resampler: None,
            command_ids: BTreeMap::new(),
            current: None,
            awaiting: None,
            blocked_command: None,
            next_token: 1,
        }
    }

    pub fn accept(&mut self, frame: SourceFrame) -> Result<Option<ArbiterEvent>> {
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
        let output = self
            .resampler
            .as_mut()
            .expect("TTS resampler initialized")
            .process(&frame.samples)?;
        Ok(self.enqueue(&output, token, &command_id))
    }

    pub fn finish(&mut self, command_id: &str) -> Result<Option<ArbiterEvent>> {
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
        if let Some(event) = self.enqueue(&tail, token, command_id) {
            return Ok(Some(event));
        }
        self.current = None;
        self.awaiting = Some(AwaitingPlayback {
            command_id: command_id.to_owned(),
            token,
            consumed_target: self.buffers.tts_enqueued_total(),
        });
        Ok(None)
    }

    pub fn poll_completed(&mut self) -> Option<ArbiterEvent> {
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
        if matches_current || matches_awaiting {
            self.preempt();
            self.buffers.clear_tts();
        } else if self.blocked_command.as_deref() == Some(command_id) {
            self.blocked_command = None;
        }
    }

    pub fn prune_command_ids(&mut self) {
        if self.current.is_none() && self.awaiting.is_none() && self.buffers.tts_is_empty() {
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
            || self.blocked_command.as_deref() == Some(command_id)
    }

    fn enqueue(&mut self, output: &[f32], token: u64, command_id: &str) -> Option<ArbiterEvent> {
        if self.buffers.enqueue_tts(output, token) == output.len() {
            return None;
        }
        self.buffers.clear_tts();
        Some(self.fail_current(command_id.to_owned()))
    }

    fn fail_current(&mut self, fallback_id: String) -> ArbiterEvent {
        let command_id = self.active_command_id().unwrap_or(fallback_id);
        self.buffers.clear_tts();
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.reset();
        }
        self.current = None;
        self.awaiting = None;
        self.blocked_command = Some(command_id.clone());
        ArbiterEvent::TtsPlaybackFailed { command_id }
    }

    fn begin(&mut self, command_id: String) -> u64 {
        let token = self.next_token;
        self.next_token = self.next_token.saturating_add(1).max(1);
        self.command_ids.insert(token, command_id.clone());
        self.current = Some((command_id, token));
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
}
