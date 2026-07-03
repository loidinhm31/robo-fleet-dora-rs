use crate::stt_protocol::VoiceCommandAudioFrame;
use crate::stt_stream_state::{mark_closing, BrowserStream, StreamPhase};
pub use crate::stt_stream_state::{FrameError, FrameErrorKind, SweepOutcome};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Debug, Default)]
pub struct StreamRegistry {
    streams: HashMap<Uuid, BrowserStream>,
}

impl StreamRegistry {
    pub fn start(
        &mut self,
        stream_id: Uuid,
        owner_socket: String,
        target_entity_id: String,
        sample_rate: u32,
        channels: u16,
        now: Instant,
    ) -> Result<(), String> {
        if self.streams.contains_key(&stream_id) {
            return Err("browser stream UUID is already owned".into());
        }
        self.streams.insert(
            stream_id,
            BrowserStream {
                owner_socket,
                target_entity_id,
                sample_rate,
                channels,
                next_frame_id: 0,
                phase: StreamPhase::Active,
                last_activity: now,
                expires_at: None,
            },
        );
        Ok(())
    }

    pub fn accept_frame(
        &mut self,
        owner_socket: &str,
        frame: &VoiceCommandAudioFrame,
        now: Instant,
    ) -> Result<String, FrameError> {
        let stream = self
            .streams
            .get_mut(&frame.stream_id)
            .ok_or_else(|| FrameError {
                kind: FrameErrorKind::Unknown,
                message: "unknown browser speech stream".into(),
            })?;
        if stream.owner_socket != owner_socket {
            return Err(FrameError {
                kind: FrameErrorKind::NotOwner,
                message: "browser speech stream belongs to another socket".into(),
            });
        }
        if stream.phase != StreamPhase::Active {
            return Err(FrameError {
                kind: FrameErrorKind::NotActive,
                message: "browser speech stream is closing".into(),
            });
        }
        if stream.sample_rate != frame.sample_rate || stream.channels != frame.channels {
            return Err(FrameError {
                kind: FrameErrorKind::MetadataChanged,
                message: "browser speech stream format changed after start".into(),
            });
        }
        if stream.next_frame_id != frame.frame_id {
            return Err(FrameError {
                kind: FrameErrorKind::Sequence,
                message: format!(
                    "browser speech frame sequence mismatch: expected {}, got {}",
                    stream.next_frame_id, frame.frame_id
                ),
            });
        }
        stream.next_frame_id = stream.next_frame_id.saturating_add(1);
        stream.last_activity = now;
        Ok(stream.target_entity_id.clone())
    }

    pub fn close(
        &mut self,
        stream_id: Uuid,
        owner_socket: &str,
        now: Instant,
        closing_ttl: Duration,
    ) -> Result<bool, String> {
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or_else(|| "unknown browser speech stream".to_string())?;
        if stream.owner_socket != owner_socket {
            return Err("browser speech stream belongs to another socket".into());
        }
        Ok(mark_closing(stream, now, closing_ttl))
    }

    pub fn close_owner(
        &mut self,
        owner_socket: &str,
        now: Instant,
        closing_ttl: Duration,
    ) -> Vec<Uuid> {
        self.streams
            .iter_mut()
            .filter_map(|(id, stream)| {
                (stream.owner_socket == owner_socket && mark_closing(stream, now, closing_ttl))
                    .then_some(*id)
            })
            .collect()
    }

    pub fn force_close(&mut self, stream_id: Uuid, now: Instant, closing_ttl: Duration) -> bool {
        self.streams
            .get_mut(&stream_id)
            .is_some_and(|stream| mark_closing(stream, now, closing_ttl))
    }

    pub fn remove(&mut self, stream_id: Uuid) {
        self.streams.remove(&stream_id);
    }

    pub fn route_browser_result(&self, stream_id: Uuid, target_entity_id: &str) -> Option<String> {
        let stream = self.streams.get(&stream_id)?;
        if stream.target_entity_id != target_entity_id {
            return None;
        }
        Some(stream.owner_socket.clone())
    }

    pub fn sweep(
        &mut self,
        now: Instant,
        idle_ttl: Duration,
        closing_ttl: Duration,
    ) -> SweepOutcome {
        let mut outcome = SweepOutcome::default();
        for (id, stream) in &mut self.streams {
            if stream.phase == StreamPhase::Active
                && now.saturating_duration_since(stream.last_activity) >= idle_ttl
                && mark_closing(stream, now, closing_ttl)
            {
                outcome.stop_streams.push(*id);
            }
        }
        let before = self.streams.len();
        self.streams.retain(|_, stream| {
            stream.phase != StreamPhase::Closing || stream.expires_at.is_none_or(|end| end > now)
        });
        outcome.expired_streams = before - self.streams.len();
        outcome
    }

    pub fn active_count(&self) -> usize {
        self.streams
            .values()
            .filter(|stream| stream.phase == StreamPhase::Active)
            .count()
    }

    pub fn len(&self) -> usize {
        self.streams.len()
    }

    #[cfg(test)]
    pub fn contains(&self, stream_id: Uuid) -> bool {
        self.streams.contains_key(&stream_id)
    }
}

#[cfg(test)]
mod tests;
