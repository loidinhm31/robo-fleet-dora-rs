use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use dora_node_api::MetadataParameters;
use robo_rover_lib::{AudioFrameMetadata, PcmSampleFormat};
use uuid::Uuid;

#[path = "walkie-audio-metrics.rs"]
mod metrics;
#[path = "walkie-audio-protocol.rs"]
mod protocol;

use metrics::{log_walkie_metrics, WalkieMetrics};
pub use protocol::WalkieAudioFrameMetadata;
use protocol::{decode_walkie_frame, metadata_parameters};

const QUEUE_BUDGET: Duration = Duration::from_millis(40);
const STREAM_TTL: Duration = Duration::from_millis(250);

pub struct QueuedWalkieFrame {
    pub metadata: AudioFrameMetadata,
    pub samples: Vec<f32>,
    duration: Duration,
}

impl QueuedWalkieFrame {
    pub fn parameters(&self) -> MetadataParameters {
        metadata_parameters(self.metadata)
    }
}

struct StreamState {
    sample_rate: u32,
    channels: u16,
    format: PcmSampleFormat,
    next_frame_id: u64,
    last_seen: Instant,
}

#[derive(Default)]
pub struct WalkieIngress {
    streams: HashMap<(String, Uuid), StreamState>,
    queue: VecDeque<QueuedWalkieFrame>,
    queued_duration: Duration,
    high_water_duration: Duration,
    metrics: WalkieMetrics,
}

impl WalkieIngress {
    pub fn record_malformed_metadata(&mut self) {
        self.metrics.invalid_frames = self.metrics.invalid_frames.saturating_add(1);
    }

    pub fn admit(
        &mut self,
        socket_id: &str,
        wire: WalkieAudioFrameMetadata,
        attachments: Vec<Vec<u8>>,
        now: Instant,
    ) -> Result<(), String> {
        self.expire_streams(now);
        let result = self.validate_and_queue(socket_id, wire, attachments, now);
        if result.is_err() {
            self.metrics.invalid_frames = self.metrics.invalid_frames.saturating_add(1);
        }
        result
    }

    pub fn pop_front(&mut self) -> Option<QueuedWalkieFrame> {
        let frame = self.queue.pop_front()?;
        self.queued_duration = self.queued_duration.saturating_sub(frame.duration);
        Some(frame)
    }

    pub fn record_forwarded(&mut self) {
        self.metrics.forwarded_frames = self.metrics.forwarded_frames.saturating_add(1);
    }

    pub fn record_send_failure(&mut self) {
        self.metrics.send_failures = self.metrics.send_failures.saturating_add(1);
    }

    pub fn remove_socket(&mut self, socket_id: &str) {
        self.streams.retain(|(owner, _), _| owner != socket_id);
    }

    pub fn expire_streams(&mut self, now: Instant) {
        self.streams
            .retain(|_, state| now.duration_since(state.last_seen) <= STREAM_TTL);
    }

    pub fn metrics(&self) -> WalkieMetrics {
        WalkieMetrics {
            queue_frames: self.queue.len(),
            queue_duration_ms: self.queued_duration.as_secs_f64() * 1_000.0,
            queue_high_water_ms: self.high_water_duration.as_secs_f64() * 1_000.0,
            ..self.metrics
        }
    }

    pub fn log_metrics(&self, stage: &'static str) {
        log_walkie_metrics(self.metrics(), stage);
    }

    fn validate_and_queue(
        &mut self,
        socket_id: &str,
        wire: WalkieAudioFrameMetadata,
        attachments: Vec<Vec<u8>>,
        now: Instant,
    ) -> Result<(), String> {
        let (metadata, samples, duration) = decode_walkie_frame(wire, attachments)?;
        let key = (socket_id.to_owned(), metadata.stream_id);
        self.validate_sequence(&key, metadata, now)?;
        while self.queued_duration.saturating_add(duration) > QUEUE_BUDGET {
            let Some(dropped) = self.queue.pop_front() else {
                break;
            };
            self.queued_duration = self.queued_duration.saturating_sub(dropped.duration);
            self.metrics.overflow_dropped_frames =
                self.metrics.overflow_dropped_frames.saturating_add(1);
            self.metrics.overflow_dropped_samples = self
                .metrics
                .overflow_dropped_samples
                .saturating_add(u64::from(dropped.metadata.sample_count));
        }
        self.queue.push_back(QueuedWalkieFrame {
            metadata,
            samples,
            duration,
        });
        self.queued_duration = self.queued_duration.saturating_add(duration);
        self.high_water_duration = self.high_water_duration.max(self.queued_duration);
        self.metrics.received_frames = self.metrics.received_frames.saturating_add(1);
        Ok(())
    }

    fn validate_sequence(
        &mut self,
        key: &(String, Uuid),
        metadata: AudioFrameMetadata,
        now: Instant,
    ) -> Result<(), String> {
        if let Some(state) = self.streams.get_mut(key) {
            if state.sample_rate != metadata.sample_rate
                || state.channels != metadata.channels
                || state.format != metadata.format
            {
                return Err("walkie stream dimensions changed during a session".into());
            }
            if metadata.frame_id < state.next_frame_id {
                self.metrics.duplicate_frames = self.metrics.duplicate_frames.saturating_add(1);
                return Err("duplicate or regressed walkie frame ID".into());
            }
            if metadata.frame_id > state.next_frame_id {
                self.metrics.gap_events = self.metrics.gap_events.saturating_add(1);
                self.metrics.missing_frames = self
                    .metrics
                    .missing_frames
                    .saturating_add(metadata.frame_id - state.next_frame_id);
            }
            state.next_frame_id = metadata.frame_id.saturating_add(1);
            state.last_seen = now;
            return Ok(());
        }
        if metadata.frame_id != 0 {
            return Err("new walkie session must start at frame zero".into());
        }
        self.streams.insert(
            key.clone(),
            StreamState {
                sample_rate: metadata.sample_rate,
                channels: metadata.channels,
                format: metadata.format,
                next_frame_id: 1,
                last_seen: now,
            },
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "walkie-audio-tests.rs"]
mod tests;
