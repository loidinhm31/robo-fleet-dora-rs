use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SttBridgeMetrics {
    pub queue_drops: u64,
    pub terminated_streams: u64,
    pub expired_streams: u64,
    pub late_transcriptions: u64,
    pub status_requests: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamPhase {
    Active,
    Closing,
}

#[derive(Debug, Clone)]
pub(crate) struct BrowserStream {
    pub owner_socket: String,
    pub target_entity_id: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub next_frame_id: u64,
    pub phase: StreamPhase,
    pub last_activity: Instant,
    pub expires_at: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameErrorKind {
    Unknown,
    NotOwner,
    NotActive,
    MetadataChanged,
    Sequence,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FrameError {
    pub kind: FrameErrorKind,
    pub message: String,
}

impl FrameError {
    pub fn terminates_owner_stream(&self) -> bool {
        !matches!(
            self.kind,
            FrameErrorKind::Unknown | FrameErrorKind::NotOwner
        )
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct SweepOutcome {
    pub stop_streams: Vec<uuid::Uuid>,
    pub expired_streams: usize,
}

pub(crate) fn mark_closing(stream: &mut BrowserStream, now: Instant, ttl: Duration) -> bool {
    if stream.phase == StreamPhase::Closing {
        return false;
    }
    stream.phase = StreamPhase::Closing;
    stream.expires_at = Some(now + ttl);
    true
}
