use std::time::{Duration, Instant};

use robo_rover_lib::{PlaybackState, PlaybackStateKind};

const SUPPRESSION_TAIL: Duration = Duration::from_millis(400);

pub struct CaptureGate {
    capture_enabled_by_user: bool,
    playback_suppressed: bool,
    tail_until: Option<Instant>,
    last_producer_instance_id: Option<String>,
    last_sequence_id: Option<u64>,
    accepted_states: u64,
    stale_states: u64,
    suppression_entries: u64,
    tail_entries: u64,
    unavailable_while_active: u64,
}

impl CaptureGate {
    pub fn new(capture_enabled_by_user: bool) -> Self {
        Self {
            capture_enabled_by_user,
            playback_suppressed: false,
            tail_until: None,
            last_producer_instance_id: None,
            last_sequence_id: None,
            accepted_states: 0,
            stale_states: 0,
            suppression_entries: 0,
            tail_entries: 0,
            unavailable_while_active: 0,
        }
    }

    pub fn set_user_enabled(&mut self, enabled: bool) {
        self.capture_enabled_by_user = enabled;
    }

    pub fn observe_playback(&mut self, state: &PlaybackState, now: Instant) {
        if self.last_producer_instance_id.as_deref() != Some(&state.producer_instance_id) {
            self.last_producer_instance_id = Some(state.producer_instance_id.clone());
            self.last_sequence_id = None;
        }
        if self
            .last_sequence_id
            .is_some_and(|last_sequence_id| state.sequence_id <= last_sequence_id)
        {
            self.stale_states = self.stale_states.saturating_add(1);
            return;
        }
        self.last_sequence_id = Some(state.sequence_id);
        self.accepted_states = self.accepted_states.saturating_add(1);
        match state.state {
            PlaybackStateKind::Active => {
                if !self.playback_suppressed {
                    self.suppression_entries = self.suppression_entries.saturating_add(1);
                }
                self.playback_suppressed = true;
                self.tail_until = None;
            }
            PlaybackStateKind::Idle => {
                if self.playback_suppressed {
                    self.playback_suppressed = false;
                    self.tail_until = Some(now + SUPPRESSION_TAIL);
                    self.tail_entries = self.tail_entries.saturating_add(1);
                }
            }
            PlaybackStateKind::Unavailable => {
                if self.playback_suppressed {
                    self.unavailable_while_active = self.unavailable_while_active.saturating_add(1);
                    self.playback_suppressed = false;
                    self.tail_until = Some(now + SUPPRESSION_TAIL);
                    self.tail_entries = self.tail_entries.saturating_add(1);
                }
            }
        }
    }

    pub fn can_publish(&mut self, now: Instant) -> bool {
        if self.tail_until.is_some_and(|deadline| now >= deadline) {
            self.tail_until = None;
        }
        self.capture_enabled_by_user && !self.playback_suppressed && self.tail_until.is_none()
    }

    pub fn metrics(&self) -> CaptureGateMetrics {
        CaptureGateMetrics {
            accepted_states: self.accepted_states,
            stale_states: self.stale_states,
            suppression_entries: self.suppression_entries,
            tail_entries: self.tail_entries,
            unavailable_while_active: self.unavailable_while_active,
            playback_suppressed: self.playback_suppressed,
            tail_active: self.tail_until.is_some(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureGateMetrics {
    pub accepted_states: u64,
    pub stale_states: u64,
    pub suppression_entries: u64,
    pub tail_entries: u64,
    pub unavailable_while_active: u64,
    pub playback_suppressed: bool,
    pub tail_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{PlaybackSource, VoiceReasonCode};

    fn state(kind: PlaybackStateKind, sequence_id: u64) -> PlaybackState {
        PlaybackState {
            entity_id: "rover-kiwi".into(),
            producer_instance_id: "550e8400-e29b-41d4-a716-446655440001".into(),
            sequence_id,
            state: kind,
            source: (kind == PlaybackStateKind::Active).then_some(PlaybackSource::Walkie),
            command_id: None,
            timestamp: 1,
            reason_code: (kind == PlaybackStateKind::Unavailable)
                .then_some(VoiceReasonCode::PlaybackUnavailable),
            detail: None,
        }
    }

    #[test]
    fn active_playback_and_tail_suppress_publication() {
        let now = Instant::now();
        let mut gate = CaptureGate::new(true);
        gate.observe_playback(&state(PlaybackStateKind::Active, 1), now);
        assert!(!gate.can_publish(now));

        gate.observe_playback(&state(PlaybackStateKind::Idle, 2), now);
        assert!(!gate.can_publish(now + Duration::from_millis(399)));
        assert!(gate.can_publish(now + Duration::from_millis(400)));
    }

    #[test]
    fn manual_enablement_is_independent_from_playback_state() {
        let now = Instant::now();
        let mut gate = CaptureGate::new(false);
        gate.observe_playback(&state(PlaybackStateKind::Idle, 1), now);
        assert!(!gate.can_publish(now));
        gate.set_user_enabled(true);
        assert!(gate.can_publish(now));
    }

    #[test]
    fn stale_or_duplicate_states_do_not_clear_active_suppression() {
        let now = Instant::now();
        let mut gate = CaptureGate::new(true);

        gate.observe_playback(&state(PlaybackStateKind::Active, 3), now);
        gate.observe_playback(&state(PlaybackStateKind::Idle, 2), now);
        gate.observe_playback(&state(PlaybackStateKind::Unavailable, 3), now);

        assert!(!gate.can_publish(now));
        assert_eq!(gate.metrics().stale_states, 2);
    }

    #[test]
    fn unavailable_while_active_uses_bounded_tail() {
        let now = Instant::now();
        let mut gate = CaptureGate::new(true);

        gate.observe_playback(&state(PlaybackStateKind::Active, 1), now);
        gate.observe_playback(&state(PlaybackStateKind::Unavailable, 2), now);

        assert!(!gate.can_publish(now + Duration::from_millis(399)));
        assert!(gate.can_publish(now + Duration::from_millis(400)));
        assert_eq!(gate.metrics().unavailable_while_active, 1);
    }

    #[test]
    fn new_producer_instance_resets_sequence_ordering() {
        let now = Instant::now();
        let mut gate = CaptureGate::new(true);

        gate.observe_playback(&state(PlaybackStateKind::Active, 3), now);
        let restarted_idle = PlaybackState {
            producer_instance_id: "550e8400-e29b-41d4-a716-446655440002".into(),
            ..state(PlaybackStateKind::Idle, 0)
        };
        gate.observe_playback(&restarted_idle, now);

        assert!(!gate.can_publish(now + Duration::from_millis(399)));
        assert!(gate.can_publish(now + Duration::from_millis(400)));
    }

    #[test]
    fn repeated_active_callbacks_coalesce_into_one_suppression_window() {
        let now = Instant::now();
        let mut gate = CaptureGate::new(true);

        gate.observe_playback(&state(PlaybackStateKind::Active, 1), now);
        gate.observe_playback(&state(PlaybackStateKind::Active, 2), now + Duration::from_millis(80));
        gate.observe_playback(
            &state(PlaybackStateKind::Active, 3),
            now + Duration::from_millis(160),
        );
        gate.observe_playback(&state(PlaybackStateKind::Idle, 4), now + Duration::from_millis(240));

        assert_eq!(gate.metrics().suppression_entries, 1);
        assert_eq!(gate.metrics().tail_entries, 1);
        assert!(!gate.can_publish(now + Duration::from_millis(639)));
        assert!(gate.can_publish(now + Duration::from_millis(640)));
    }
}
