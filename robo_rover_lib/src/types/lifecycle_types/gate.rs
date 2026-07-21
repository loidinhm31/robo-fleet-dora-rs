use super::{
    LifecycleCommand, LifecycleComponentState, LifecycleComponentStatus, LifecycleDesiredState,
    LifecycleEffectiveState, LifecycleReasonCode, LifecycleStatus, LifecycleTarget,
    LIFECYCLE_PROTOCOL_VERSION,
};

/// Per-workload admission gate. Nodes own teardown, but must use this gate to
/// reject stale transitions and keep new work out while resources are released.
#[derive(Debug, Clone)]
pub struct LifecycleGate {
    target: LifecycleTarget,
    admission_open: bool,
    last_epoch: u64,
    last_revision: u64,
    last_desired_state: Option<LifecycleDesiredState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleTransition {
    Quiesce,
    Resume,
}

impl LifecycleGate {
    pub fn new(target: LifecycleTarget) -> Self {
        Self {
            target,
            admission_open: true,
            last_epoch: 0,
            last_revision: 0,
            last_desired_state: None,
        }
    }

    pub fn admission_open(&self) -> bool {
        self.admission_open
    }

    /// The authority currently controlling this workload. It remains visible
    /// while a transition is in progress so asynchronous acknowledgements can
    /// prove that they still belong to the active lifecycle command.
    pub fn authority(&self) -> (u64, u64) {
        (self.last_epoch, self.last_revision)
    }

    pub fn desired_state(&self) -> Option<LifecycleDesiredState> {
        self.last_desired_state
    }

    /// Returns `None` for a duplicate transition and rejects foreign or stale
    /// authority commands. Closing admission happens before node-owned cleanup.
    pub fn begin(
        &mut self,
        command: &LifecycleCommand,
    ) -> Result<Option<LifecycleTransition>, String> {
        command.validate()?;
        if command.target != self.target {
            return Err("lifecycle command target does not match this node".into());
        }
        let version = (
            command.manager_epoch,
            command.expected_revision.saturating_add(1),
        );
        let last = (self.last_epoch, self.last_revision);
        if version < last {
            return Err("lifecycle command authority is stale".into());
        }
        if version == last {
            return if self.last_desired_state == Some(command.desired_state) {
                Ok(None)
            } else {
                Err("lifecycle command payload changed at the same authority revision".into())
            };
        }
        self.last_epoch = version.0;
        self.last_revision = version.1;
        self.last_desired_state = Some(command.desired_state);
        let transition = match command.desired_state {
            LifecycleDesiredState::Quiesced => {
                self.admission_open = false;
                LifecycleTransition::Quiesce
            }
            LifecycleDesiredState::Running => LifecycleTransition::Resume,
        };
        Ok(Some(transition))
    }

    pub fn complete(&mut self, transition: LifecycleTransition) {
        self.admission_open = matches!(transition, LifecycleTransition::Resume);
    }

    pub fn status(
        &self,
        state: LifecycleComponentState,
        reason_code: Option<LifecycleReasonCode>,
        updated_at_ms: u64,
    ) -> LifecycleStatus {
        LifecycleStatus {
            protocol_version: LIFECYCLE_PROTOCOL_VERSION,
            manager_epoch: self.last_epoch,
            target: self.target.clone(),
            revision: self.last_revision,
            desired_state: self.last_desired_state.unwrap_or_else(|| {
                if self.admission_open {
                    LifecycleDesiredState::Running
                } else {
                    LifecycleDesiredState::Quiesced
                }
            }),
            effective_state: match state {
                LifecycleComponentState::Running => LifecycleEffectiveState::Running,
                LifecycleComponentState::Cancelling => LifecycleEffectiveState::Cancelling,
                LifecycleComponentState::Quiescing => LifecycleEffectiveState::Quiescing,
                LifecycleComponentState::Quiesced => LifecycleEffectiveState::Quiesced,
                LifecycleComponentState::Resuming => LifecycleEffectiveState::Resuming,
                LifecycleComponentState::Degraded => LifecycleEffectiveState::Degraded,
                LifecycleComponentState::Failed | LifecycleComponentState::Unsupported => {
                    LifecycleEffectiveState::Failed
                }
            },
            components: vec![LifecycleComponentStatus {
                node_id: self.target.node_id.clone(),
                state,
                reason_code,
            }],
            updated_at_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LifecycleRole, LifecycleTarget};

    fn command(revision: u64, desired_state: LifecycleDesiredState) -> LifecycleCommand {
        LifecycleCommand {
            protocol_version: 1,
            request_id: "550e8400-e29b-41d4-a716-446655440000".into(),
            manager_epoch: 7,
            target: LifecycleTarget {
                role: LifecycleRole::Rover,
                entity_id: "rover-kiwi".into(),
                node_id: "gst-camera".into(),
            },
            desired_state,
            expected_revision: revision,
            issued_at_ms: 100,
            expires_at_ms: 1_000,
        }
    }

    #[test]
    fn closes_admission_before_quiesce_and_reopens_only_after_resume() {
        let mut gate =
            LifecycleGate::new(command(0, LifecycleDesiredState::Running).target.clone());
        assert_eq!(
            gate.begin(&command(0, LifecycleDesiredState::Quiesced))
                .unwrap(),
            Some(LifecycleTransition::Quiesce)
        );
        assert!(!gate.admission_open());
        gate.complete(LifecycleTransition::Quiesce);
        assert!(!gate.admission_open());
        assert_eq!(
            gate.begin(&command(1, LifecycleDesiredState::Running))
                .unwrap(),
            Some(LifecycleTransition::Resume)
        );
        assert!(!gate.admission_open());
        gate.complete(LifecycleTransition::Resume);
        assert!(gate.admission_open());
    }

    #[test]
    fn ignores_duplicate_and_rejects_stale_authority() {
        let mut gate =
            LifecycleGate::new(command(0, LifecycleDesiredState::Running).target.clone());
        gate.begin(&command(0, LifecycleDesiredState::Quiesced))
            .unwrap();
        assert_eq!(
            gate.begin(&command(0, LifecycleDesiredState::Quiesced))
                .unwrap(),
            None
        );
        assert!(gate
            .begin(&command(0, LifecycleDesiredState::Running))
            .is_err());
    }

    #[test]
    fn transition_status_keeps_the_command_desired_state_before_completion() {
        let mut gate =
            LifecycleGate::new(command(0, LifecycleDesiredState::Running).target.clone());
        gate.begin(&command(0, LifecycleDesiredState::Quiesced))
            .unwrap();
        gate.complete(LifecycleTransition::Quiesce);
        gate.begin(&command(1, LifecycleDesiredState::Running))
            .unwrap();

        let status = gate.status(LifecycleComponentState::Resuming, None, 10);
        assert_eq!(status.desired_state, LifecycleDesiredState::Running);
        assert!(!gate.admission_open());
    }
}
