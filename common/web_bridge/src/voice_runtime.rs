use std::collections::{BTreeMap, BTreeSet};

use robo_rover_lib::{
    TtsConfigCommand, TtsConfigState, TtsConfigUpdate, TtsRuntimeConfig, VoiceReasonCode,
    VoiceState, VoiceStatus,
};

pub enum ConfigUpdateOutcome {
    Accepted {
        command: TtsConfigCommand,
        state: TtsConfigState,
    },
    Stale {
        state: TtsConfigState,
    },
}

#[derive(Debug, Clone)]
pub struct VoiceRuntimeState {
    desired_revision: u64,
    desired_config: TtsRuntimeConfig,
    active_rovers: BTreeSet<String>,
    rover_statuses: BTreeMap<String, VoiceStatus>,
}

impl VoiceRuntimeState {
    pub fn new(active_rovers: Vec<String>, desired_config: TtsRuntimeConfig) -> Self {
        Self {
            desired_revision: 0,
            desired_config,
            active_rovers: active_rovers.into_iter().collect(),
            rover_statuses: BTreeMap::new(),
        }
    }

    pub fn config_state(&self, timestamp: u64) -> TtsConfigState {
        let rovers = self
            .active_rovers
            .iter()
            .map(|entity_id| self.config_status(entity_id, timestamp))
            .collect::<Vec<_>>();
        let applied_rovers = rovers
            .iter()
            .filter(|status| {
                status.applied_revision == self.desired_revision
                    && status.state != VoiceState::Unavailable
                    && status.applied_config == self.desired_config
            })
            .count() as u32;
        let state = TtsConfigState {
            desired_revision: self.desired_revision,
            desired_config: self.desired_config.clone(),
            applied_rovers,
            active_rovers: rovers.len() as u32,
            rovers,
            timestamp,
        };
        debug_assert!(state.validate().is_ok());
        state
    }

    pub fn active_voice_statuses(&self, timestamp: u64) -> Vec<VoiceStatus> {
        self.active_rovers
            .iter()
            .map(|entity_id| {
                self.rover_statuses
                    .get(entity_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        unavailable_status(entity_id, &self.desired_config, timestamp)
                    })
            })
            .collect()
    }

    #[cfg(test)]
    pub fn handle_config_update(
        &mut self,
        update: TtsConfigUpdate,
        timestamp: u64,
    ) -> ConfigUpdateOutcome {
        match self.preview_config_update(update, timestamp) {
            ConfigUpdateOutcome::Accepted { command, .. } => self.commit_config_command(command),
            ConfigUpdateOutcome::Stale { state } => ConfigUpdateOutcome::Stale { state },
        }
    }

    pub fn preview_config_update(
        &self,
        update: TtsConfigUpdate,
        timestamp: u64,
    ) -> ConfigUpdateOutcome {
        if update.base_revision != self.desired_revision {
            return ConfigUpdateOutcome::Stale {
                state: self.config_state(timestamp),
            };
        }

        let mut preview = self.clone();
        preview.desired_revision = preview.desired_revision.saturating_add(1);
        preview.desired_config = update.config.clone();
        ConfigUpdateOutcome::Accepted {
            command: TtsConfigCommand {
                revision: preview.desired_revision,
                config: update.config,
            },
            state: preview.config_state(timestamp),
        }
    }

    pub fn commit_config_command(&mut self, command: TtsConfigCommand) -> ConfigUpdateOutcome {
        self.desired_revision = command.revision;
        self.desired_config = command.config.clone();
        ConfigUpdateOutcome::Accepted {
            command,
            state: self.config_state(current_timestamp_ms()),
        }
    }

    pub fn record_voice_status(&mut self, status: VoiceStatus) -> bool {
        if status.applied_revision > self.desired_revision {
            return false;
        }

        let should_replace = match self.rover_statuses.get(&status.entity_id) {
            None => true,
            Some(existing) if status.applied_revision > existing.applied_revision => true,
            Some(existing)
                if status.applied_revision == existing.applied_revision
                    && status.timestamp >= existing.timestamp =>
            {
                true
            }
            _ => false,
        };

        if should_replace {
            self.rover_statuses.insert(status.entity_id.clone(), status);
        }

        should_replace
    }

    pub fn sync_active_rovers(&mut self, active_rovers: Vec<String>) -> bool {
        let next: BTreeSet<String> = active_rovers.into_iter().collect();
        if self.active_rovers == next {
            return false;
        }
        self.rover_statuses
            .retain(|entity_id, _| next.contains(entity_id));
        self.active_rovers = next;
        true
    }

    fn config_status(&self, entity_id: &str, timestamp: u64) -> VoiceStatus {
        match self.rover_statuses.get(entity_id) {
            Some(status) if status.applied_revision <= self.desired_revision => status.clone(),
            _ => unavailable_status(entity_id, &self.desired_config, timestamp),
        }
    }
}

fn unavailable_status(
    entity_id: &str,
    desired_config: &TtsRuntimeConfig,
    timestamp: u64,
) -> VoiceStatus {
    VoiceStatus {
        entity_id: entity_id.to_string(),
        state: VoiceState::Unavailable,
        applied_revision: 0,
        applied_config: desired_config.clone(),
        active_command_id: None,
        timestamp,
        reason_code: Some(VoiceReasonCode::VoiceNotReady),
        detail: Some("awaiting voice status".to_string()),
    }
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{TtsLanguage, VoiceState};

    fn timestamp(ms: u64) -> u64 {
        ms
    }

    fn runtime() -> VoiceRuntimeState {
        VoiceRuntimeState::new(
            vec!["rover-a".to_string(), "rover-b".to_string()],
            TtsRuntimeConfig::default(),
        )
    }

    fn ready_status(entity_id: &str, revision: u64, timestamp: u64) -> VoiceStatus {
        VoiceStatus {
            entity_id: entity_id.to_string(),
            state: VoiceState::Ready,
            applied_revision: revision,
            applied_config: TtsRuntimeConfig::default(),
            active_command_id: None,
            timestamp,
            reason_code: None,
            detail: None,
        }
    }

    #[test]
    fn stale_update_returns_current_state_without_mutating() {
        let mut state = runtime();
        let outcome = state.handle_config_update(
            TtsConfigUpdate {
                base_revision: 1,
                config: TtsRuntimeConfig::default(),
            },
            timestamp(10),
        );

        match outcome {
            ConfigUpdateOutcome::Accepted { .. } => panic!("expected stale update"),
            ConfigUpdateOutcome::Stale { state } => {
                assert_eq!(state.desired_revision, 0);
            }
        }
    }

    #[test]
    fn accepted_update_increments_revision_and_preserves_partial_convergence() {
        let mut state = runtime();
        let outcome = state.handle_config_update(
            TtsConfigUpdate {
                base_revision: 0,
                config: TtsRuntimeConfig {
                    language: TtsLanguage::Vi,
                    ..TtsRuntimeConfig::default()
                },
            },
            timestamp(20),
        );

        let accepted = match outcome {
            ConfigUpdateOutcome::Accepted { command, state } => (command, state),
            ConfigUpdateOutcome::Stale { .. } => panic!("expected accepted update"),
        };

        assert_eq!(accepted.0.revision, 1);
        assert_eq!(accepted.1.desired_revision, 1);
        assert_eq!(accepted.1.applied_rovers, 0);
        assert_eq!(accepted.1.active_rovers, 2);
    }

    #[test]
    fn older_status_cannot_regress_applied_revision() {
        let mut state = runtime();
        let _ = state.handle_config_update(
            TtsConfigUpdate {
                base_revision: 0,
                config: TtsRuntimeConfig::default(),
            },
            timestamp(20),
        );
        assert!(state.record_voice_status(ready_status("rover-a", 1, timestamp(30))));
        assert!(!state.record_voice_status(ready_status("rover-a", 0, timestamp(40))));

        let config = state.config_state(timestamp(50));
        assert_eq!(config.rovers[0].applied_revision, 1);
    }

    #[test]
    fn newer_timestamp_replaces_same_revision_status() {
        let mut state = runtime();
        let _ = state.handle_config_update(
            TtsConfigUpdate {
                base_revision: 0,
                config: TtsRuntimeConfig::default(),
            },
            timestamp(20),
        );
        assert!(state.record_voice_status(ready_status("rover-a", 1, timestamp(30))));
        let mut speaking = ready_status("rover-a", 1, timestamp(31));
        speaking.state = VoiceState::Speaking;
        speaking.active_command_id = Some(uuid::Uuid::new_v4().to_string());
        assert!(state.record_voice_status(speaking.clone()));

        let config = state.config_state(timestamp(40));
        assert_eq!(config.rovers[0].state, VoiceState::Speaking);
        assert_eq!(
            config.rovers[0].active_command_id,
            speaking.active_command_id
        );
    }

    #[test]
    fn config_state_uses_placeholder_for_statuses_ahead_of_authority() {
        let mut state = runtime();
        assert!(!state.record_voice_status(ready_status("rover-a", 3, timestamp(30))));

        let config = state.config_state(timestamp(50));
        assert_eq!(config.rovers[0].state, VoiceState::Unavailable);
        assert_eq!(
            config.rovers[0].reason_code,
            Some(VoiceReasonCode::VoiceNotReady)
        );

        let latest = state.active_voice_statuses(timestamp(50));
        assert_eq!(latest[0].state, VoiceState::Unavailable);
        assert_eq!(latest[0].applied_revision, 0);
    }

    #[test]
    fn active_rover_sync_updates_config_state() {
        let mut state = runtime();
        assert!(state.sync_active_rovers(vec!["rover-b".to_string()]));
        let config = state.config_state(timestamp(60));
        assert_eq!(config.active_rovers, 1);
        assert_eq!(config.rovers.len(), 1);
        assert_eq!(config.rovers[0].entity_id, "rover-b");
    }

    #[test]
    fn revision_zero_placeholders_do_not_count_as_applied() {
        let state = runtime();
        let config = state.config_state(timestamp(10));
        assert_eq!(config.desired_revision, 0);
        assert_eq!(config.applied_rovers, 0);
        assert_eq!(config.active_rovers, 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn revision_zero_requires_config_equality_to_count_as_applied() {
        let mut state = runtime();
        let mut mismatched = ready_status("rover-a", 0, timestamp(10));
        mismatched.applied_config.language = TtsLanguage::Vi;
        assert!(state.record_voice_status(mismatched));

        let config = state.config_state(timestamp(11));
        assert_eq!(config.applied_rovers, 0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn future_status_revision_is_rejected() {
        let mut state = runtime();
        assert!(!state.record_voice_status(ready_status("rover-a", 1, timestamp(10))));
        let statuses = state.active_voice_statuses(timestamp(11));
        assert_eq!(statuses[0].state, VoiceState::Unavailable);
    }

    #[test]
    fn reactivation_requires_fresh_status() {
        let mut state = runtime();
        let _ = state.handle_config_update(
            TtsConfigUpdate {
                base_revision: 0,
                config: TtsRuntimeConfig::default(),
            },
            timestamp(20),
        );
        assert!(state.record_voice_status(ready_status("rover-a", 1, timestamp(30))));
        assert!(state.sync_active_rovers(vec!["rover-b".to_string()]));
        assert!(state.sync_active_rovers(vec!["rover-a".to_string(), "rover-b".to_string()]));

        let config = state.config_state(timestamp(40));
        let rover_a = config
            .rovers
            .iter()
            .find(|status| status.entity_id == "rover-a")
            .expect("rover-a should be present");
        assert_eq!(rover_a.state, VoiceState::Unavailable);
        assert_eq!(config.applied_rovers, 0);
    }
}
