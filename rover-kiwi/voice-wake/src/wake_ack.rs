use robo_rover_lib::{PlaybackStateKind, PowerProfile, PowerState, PowerStatus};

pub const WAKE_ACK_PCM: &[u8] = include_bytes!("../assets/i-am-on.pcm");
pub const WAKE_ACK_SAMPLE_RATE: u32 = 44_100;

#[derive(Debug, Default)]
pub struct WakeAckGate {
    pending_demand_id: Option<String>,
}

impl WakeAckGate {
    pub fn arm(&mut self, demand_id: String) {
        self.pending_demand_id = Some(demand_id);
    }

    pub fn ready(
        &mut self,
        status: &PowerStatus,
        playback_state: Option<PlaybackStateKind>,
    ) -> Option<String> {
        (status.state == PowerState::Active
            && status.effective_profile == PowerProfile::NormalRover
            && matches!(playback_state, Some(PlaybackStateKind::Idle)))
        .then(|| self.pending_demand_id.take())
        .flatten()
    }

    pub fn clear(&mut self) {
        self.pending_demand_id = None;
    }
}

pub fn samples() -> Vec<f32> {
    WAKE_ACK_PCM
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("exact f32 chunks")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_is_f32_mono_and_nonempty() {
        assert!(!WAKE_ACK_PCM.is_empty());
        assert_eq!(WAKE_ACK_PCM.len() % 4, 0);
        assert!(samples().iter().all(|sample| sample.is_finite()));
    }
}
