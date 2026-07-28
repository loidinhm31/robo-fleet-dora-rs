use robo_rover_lib::{
    LifecycleRole, PlaybackState, PlaybackStateKind, PowerCommand, PowerCommandAction,
    PowerCommandResult, PowerDemand, PowerDemandAction, PowerDemandPriority, PowerDemandSource,
    PowerProfile, PowerState, PowerStatus, POWER_PROTOCOL_VERSION,
};
use uuid::Uuid;

use crate::{debounce::WakeDebounce, wake_ack::WakeAckGate};

const KWS_TTL_MS: u64 = 300_000;
const COMMAND_TTL_MS: u64 = 60_000;

pub struct WakeController {
    entity_id: String,
    status: Option<PowerStatus>,
    playback_state: Option<PlaybackStateKind>,
    debounce: WakeDebounce,
    pending_command: Option<(String, String)>,
    wake_ack: WakeAckGate,
}

impl WakeController {
    pub fn new(entity_id: String) -> Self {
        Self {
            entity_id,
            status: None,
            playback_state: None,
            debounce: WakeDebounce::default(),
            pending_command: None,
            wake_ack: WakeAckGate::default(),
        }
    }

    pub fn observe_status(&mut self, status: PowerStatus) {
        if status.role == LifecycleRole::Rover && status.entity_id == self.entity_id {
            self.status = Some(status);
        }
    }

    pub fn observe_playback(&mut self, playback: PlaybackState) {
        if playback.entity_id == self.entity_id {
            self.playback_state = Some(playback.state);
        }
    }

    pub fn observe_result(&mut self, result: PowerCommandResult) {
        let Some((command_id, _)) = self.pending_command.as_ref() else {
            return;
        };
        if result.command_id != command_id.as_str() {
            return;
        }
        let (_, demand_id) = self
            .pending_command
            .take()
            .expect("pending command checked");
        if result.accepted {
            self.wake_ack.arm(demand_id);
        }
    }

    pub fn listens(&self) -> bool {
        self.status.as_ref().is_some_and(|status| {
            status.state == PowerState::IdleListening
                && status.effective_profile == PowerProfile::IdleListening
        })
    }

    pub fn wake_command(&mut self, now_ms: u64) -> Option<PowerCommand> {
        let status = self.status.as_ref()?;
        if !self.listens() || !self.debounce.accept(now_ms) {
            return None;
        }
        let bucket = now_ms / KWS_TTL_MS;
        let demand_id = Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!("robo-fleet:kws:{}:{bucket}", self.entity_id).as_bytes(),
        )
        .to_string();
        let command_id = Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!("robo-fleet:kws-command:{demand_id}").as_bytes(),
        )
        .to_string();
        let demand = PowerDemand {
            protocol_version: POWER_PROTOCOL_VERSION,
            demand_id: demand_id.clone(),
            action: PowerDemandAction::Acquire,
            source: PowerDemandSource::Kws,
            priority: PowerDemandPriority::High,
            role: LifecycleRole::Rover,
            entity_id: self.entity_id.clone(),
            required_profile: PowerProfile::NormalRover,
            authority: status.authority,
            issued_at_ms: now_ms,
            not_before_ms: now_ms,
            expires_at_ms: now_ms.saturating_add(KWS_TTL_MS),
            renew_sequence: 1,
        };
        self.pending_command = Some((command_id.clone(), demand_id));
        Some(PowerCommand {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id,
            role: LifecycleRole::Rover,
            entity_id: self.entity_id.clone(),
            authority: status.authority,
            action: PowerCommandAction::RegisterDemand { demand },
            issued_at_ms: now_ms,
            not_before_ms: now_ms,
            expires_at_ms: now_ms.saturating_add(COMMAND_TTL_MS),
            detail: Some("local_kws_hey_kiwi".into()),
        })
    }

    pub fn ready_wake_ack(&mut self) -> Option<String> {
        self.status
            .as_ref()
            .and_then(|status| self.wake_ack.ready(status, self.playback_state))
    }

    pub fn reset(&mut self) {
        self.debounce.reset();
        self.pending_command = None;
        self.wake_ack.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{PowerAuthority, PowerPolicy};

    fn idle_status() -> PowerStatus {
        PowerStatus {
            protocol_version: POWER_PROTOCOL_VERSION,
            role: LifecycleRole::Rover,
            entity_id: "rover-kiwi".into(),
            authority: PowerAuthority {
                epoch: 2,
                sequence: 3,
            },
            policy: PowerPolicy::Auto,
            requested_profile: PowerProfile::IdleListening,
            effective_profile: PowerProfile::IdleListening,
            state: PowerState::IdleListening,
            transition_id: None,
            reason_code: None,
            detail: None,
            active_reservations: vec![],
            updated_at_ms: 1,
        }
    }

    #[test]
    fn only_idle_listening_creates_a_bounded_kws_demand() {
        let mut controller = WakeController::new("rover-kiwi".into());
        controller.observe_status(idle_status());
        let command = controller.wake_command(300_001).unwrap();
        let PowerCommandAction::RegisterDemand { demand } = command.action else {
            panic!("expected demand command");
        };
        assert_eq!(demand.source, PowerDemandSource::Kws);
        assert_eq!(demand.required_profile, PowerProfile::NormalRover);
        assert_eq!(demand.expires_at_ms - demand.issued_at_ms, KWS_TTL_MS);
        assert!(controller.wake_command(300_002).is_none());
    }

    #[test]
    fn accepted_kws_demand_plays_once_after_normal_rover_is_ready() {
        let mut controller = WakeController::new("rover-kiwi".into());
        controller.observe_status(idle_status());
        let command = controller.wake_command(300_001).unwrap();
        let demand_id = match &command.action {
            PowerCommandAction::RegisterDemand { demand } => demand.demand_id.clone(),
            _ => unreachable!(),
        };
        controller.observe_result(PowerCommandResult {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id: command.command_id,
            accepted: true,
            authority: PowerAuthority {
                epoch: 2,
                sequence: 4,
            },
            reason_code: None,
            detail: None,
        });
        let mut ready = idle_status();
        ready.effective_profile = PowerProfile::NormalRover;
        ready.requested_profile = PowerProfile::NormalRover;
        ready.state = PowerState::Active;
        controller.observe_status(ready);
        assert_eq!(controller.ready_wake_ack(), None);
        controller.observe_playback(PlaybackState {
            entity_id: "rover-kiwi".into(),
            producer_instance_id: "playback".into(),
            sequence_id: 1,
            state: PlaybackStateKind::Idle,
            source: None,
            command_id: None,
            timestamp: 1,
            reason_code: None,
            detail: None,
        });
        assert_eq!(controller.ready_wake_ack(), Some(demand_id));
        assert_eq!(controller.ready_wake_ack(), None);
    }
}
