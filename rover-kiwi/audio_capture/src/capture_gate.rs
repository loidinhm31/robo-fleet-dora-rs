use std::time::{Duration, Instant};

use robo_rover_lib::{PlaybackState, PlaybackStateKind};

const SUPPRESSION_TAIL: Duration = Duration::from_millis(400);

pub struct CaptureGate {
    capture_enabled_by_user: bool,
    playback_suppressed: bool,
    tail_until: Option<Instant>,
}

impl CaptureGate {
    pub fn new(capture_enabled_by_user: bool) -> Self {
        Self {
            capture_enabled_by_user,
            playback_suppressed: false,
            tail_until: None,
        }
    }

    pub fn set_user_enabled(&mut self, enabled: bool) {
        self.capture_enabled_by_user = enabled;
    }

    pub fn observe_playback(&mut self, state: &PlaybackState, now: Instant) {
        match state.state {
            PlaybackStateKind::Active => {
                self.playback_suppressed = true;
                self.tail_until = None;
            }
            PlaybackStateKind::Idle | PlaybackStateKind::Unavailable => {
                if self.playback_suppressed {
                    self.playback_suppressed = false;
                    self.tail_until = Some(now + SUPPRESSION_TAIL);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{PlaybackSource, VoiceReasonCode};

    fn state(kind: PlaybackStateKind) -> PlaybackState {
        PlaybackState {
            entity_id: "rover-kiwi".into(),
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
        gate.observe_playback(&state(PlaybackStateKind::Active), now);
        assert!(!gate.can_publish(now));

        gate.observe_playback(&state(PlaybackStateKind::Idle), now);
        assert!(!gate.can_publish(now + Duration::from_millis(399)));
        assert!(gate.can_publish(now + Duration::from_millis(400)));
    }

    #[test]
    fn manual_enablement_is_independent_from_playback_state() {
        let now = Instant::now();
        let mut gate = CaptureGate::new(false);
        gate.observe_playback(&state(PlaybackStateKind::Idle), now);
        assert!(!gate.can_publish(now));
        gate.set_user_enabled(true);
        assert!(gate.can_publish(now));
    }
}
