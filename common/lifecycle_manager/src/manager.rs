use robo_rover_lib::{
    LifecycleCapability, LifecycleCommand, LifecycleCommandResult, LifecycleComponentState,
    LifecycleComponentStatus, LifecycleDesiredState, LifecycleEffectiveState, LifecycleReasonCode,
    LifecycleStatus, LifecycleTarget, LifecycleWakeLease, LifecycleWakeLeaseAction,
    LIFECYCLE_PROTOCOL_VERSION,
};
use std::collections::{BTreeMap, BTreeSet};

const TRANSITION_TIMEOUT_MS: u64 = 30_000;
const MAX_CACHED_REQUESTS: usize = 1_024;

#[derive(Debug, Clone)]
struct CachedRequest {
    command: LifecycleCommand,
    result: LifecycleCommandResult,
}

#[derive(Debug, Clone)]
struct ManagedTarget {
    capability: LifecycleCapability,
    desired_state: LifecycleDesiredState,
    revision: u64,
    effective_state: LifecycleEffectiveState,
    transition_deadline_ms: Option<u64>,
    authority_epoch: u64,
    /// A deadline-expired transition cannot accept a late success status at
    /// the same authority version.
    timed_out_authority: Option<(u64, u64)>,
}

pub struct LifecycleManager {
    epoch: u64,
    targets: BTreeMap<LifecycleTarget, ManagedTarget>,
    requests: BTreeMap<String, CachedRequest>,
    wake_leases: BTreeMap<LifecycleTarget, BTreeMap<String, u64>>,
    revoked_wake_leases: BTreeSet<(LifecycleTarget, String)>,
}

impl LifecycleManager {
    pub fn new(epoch: u64, capabilities: Vec<LifecycleCapability>) -> Result<Self, String> {
        if epoch == 0 {
            return Err("lifecycle manager epoch must be positive".into());
        }
        let mut targets = BTreeMap::new();
        for capability in capabilities {
            capability.target.validate()?;
            if targets.contains_key(&capability.target) {
                return Err("duplicate lifecycle capability target".into());
            }
            targets.insert(
                capability.target.clone(),
                ManagedTarget {
                    capability,
                    desired_state: LifecycleDesiredState::Running,
                    revision: 0,
                    effective_state: LifecycleEffectiveState::Running,
                    transition_deadline_ms: None,
                    authority_epoch: epoch,
                    timed_out_authority: None,
                },
            );
        }
        Ok(Self {
            epoch,
            targets,
            requests: BTreeMap::new(),
            wake_leases: BTreeMap::new(),
            revoked_wake_leases: BTreeSet::new(),
        })
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn capabilities(&self) -> Vec<LifecycleCapability> {
        self.targets
            .values()
            .map(|item| item.capability.clone())
            .collect()
    }

    pub fn status(&self, target: &LifecycleTarget, now_ms: u64) -> Option<LifecycleStatus> {
        self.targets
            .get(target)
            .map(|item| self.status_for(target, item, now_ms))
    }

    pub fn statuses(&self, now_ms: u64) -> Vec<LifecycleStatus> {
        self.targets
            .iter()
            .map(|(target, item)| self.status_for(target, item, now_ms))
            .collect()
    }

    pub fn apply(&mut self, command: LifecycleCommand, now_ms: u64) -> LifecycleCommandResult {
        self.requests
            .retain(|_, cached| cached.command.expires_at_ms > now_ms);
        if let Some(cached) = self.requests.get(&command.request_id) {
            return if cached.command == command {
                cached.result.clone()
            } else {
                self.rejected(
                    &command,
                    LifecycleReasonCode::DuplicateMismatch,
                    "request id payload changed",
                )
            };
        }
        if self.requests.len() >= MAX_CACHED_REQUESTS {
            return self.rejected(
                &command,
                LifecycleReasonCode::Internal,
                "lifecycle request cache is full",
            );
        }
        let result = self.admit(&command, now_ms);
        let request_id = command.request_id.clone();
        self.requests.insert(
            request_id.clone(),
            CachedRequest {
                command,
                result: result.clone(),
            },
        );
        result
    }

    /// Applies a command relayed by the Orchestra authority. This manager never
    /// owns the revision: it only mirrors the authoritative desired state and
    /// reports its local application status using the Orchestra epoch.
    pub fn apply_relayed(
        &mut self,
        command: LifecycleCommand,
        now_ms: u64,
    ) -> LifecycleCommandResult {
        self.requests
            .retain(|_, cached| cached.command.expires_at_ms > now_ms);
        if let Some(cached) = self.requests.get(&command.request_id) {
            return if cached.command == command {
                cached.result.clone()
            } else {
                self.rejected(
                    &command,
                    LifecycleReasonCode::DuplicateMismatch,
                    "request id payload changed",
                )
            };
        }
        if self.requests.len() >= MAX_CACHED_REQUESTS {
            return self.rejected(
                &command,
                LifecycleReasonCode::Internal,
                "lifecycle request cache is full",
            );
        }
        if let Err(error) = command.validate() {
            return self.rejected(&command, LifecycleReasonCode::InvalidRequest, &error);
        }
        if command.expires_at_ms <= now_ms {
            return self.rejected(&command, LifecycleReasonCode::Expired, "command expired");
        }
        let Some(target) = self.targets.get_mut(&command.target) else {
            return self.rejected(
                &command,
                LifecycleReasonCode::InvalidTarget,
                "target is not server-advertised",
            );
        };
        if !target.capability.supported || target.capability.always_on {
            return self.rejected(
                &command,
                LifecycleReasonCode::Unsupported,
                "target cannot be lifecycle-controlled",
            );
        }
        let has_remote_authority = target.authority_epoch != self.epoch || target.revision != 0;
        if has_remote_authority
            && (command.manager_epoch < target.authority_epoch
                || (command.manager_epoch == target.authority_epoch
                    && command.expected_revision < target.revision))
        {
            return self.rejected(
                &command,
                LifecycleReasonCode::StaleEpoch,
                "relayed authority epoch or revision is stale",
            );
        }
        // The Orchestra revision is authoritative. A relayed snapshot can move
        // this local reporter forward after restart without creating a second CAS authority.
        target.authority_epoch = command.manager_epoch;
        target.revision = command.expected_revision.saturating_add(1);
        target.timed_out_authority = None;
        target.desired_state = command.desired_state;
        if command.desired_state == LifecycleDesiredState::Quiesced {
            self.revoke_wake_leases(&command.target);
        }
        self.reconcile_target(&command.target, now_ms);
        let result = LifecycleCommandResult {
            protocol_version: LIFECYCLE_PROTOCOL_VERSION,
            request_id: command.request_id.clone(),
            accepted: true,
            manager_epoch: command.manager_epoch,
            revision: command.expected_revision.saturating_add(1),
            reason_code: None,
            detail: None,
        };
        self.requests.insert(
            command.request_id.clone(),
            CachedRequest {
                command,
                result: result.clone(),
            },
        );
        result
    }

    /// Phase 3 adapters must return a status carrying the authority epoch and
    /// revision that caused their transition. Older acknowledgements are ignored.
    pub fn apply_component_status(&mut self, status: &LifecycleStatus) -> bool {
        if status.validate().is_err() {
            return false;
        }
        let Some(target) = self.targets.get(&status.target) else {
            return false;
        };
        if target.authority_epoch != status.manager_epoch || target.revision != status.revision {
            return false;
        }
        if target.desired_state != status.desired_state {
            return false;
        }
        if target.timed_out_authority == Some((status.manager_epoch, status.revision)) {
            return false;
        }
        let state = status
            .components
            .iter()
            .find(|component| component.node_id == status.target.node_id)
            .map(|component| component.state);
        if let Some(state) = state {
            self.component_applied(&status.target, state, status.updated_at_ms);
            true
        } else {
            false
        }
    }

    pub fn acquire_wake_lease(
        &mut self,
        lease_id: String,
        target: &LifecycleTarget,
        expires_at_ms: u64,
        now_ms: u64,
    ) -> Result<(), String> {
        if lease_id.is_empty() || lease_id.len() > 128 || expires_at_ms <= now_ms {
            return Err("invalid lifecycle wake lease".into());
        }
        self.targets.get(target).ok_or("unknown lifecycle target")?;
        if self
            .revoked_wake_leases
            .contains(&(target.clone(), lease_id.clone()))
        {
            return Err("wake lease was revoked by a user pause".into());
        }
        self.wake_leases
            .entry(target.clone())
            .or_default()
            .insert(lease_id, expires_at_ms);
        self.reconcile_target(target, now_ms);
        Ok(())
    }

    /// Scheduler-only wake leases are independent of browser CAS commands.
    /// They temporarily affect effective state but never mutate user desired state.
    pub fn apply_wake_lease(
        &mut self,
        lease: LifecycleWakeLease,
        now_ms: u64,
    ) -> Result<(), String> {
        lease.validate()?;
        match lease.action {
            LifecycleWakeLeaseAction::Acquire => {
                self.acquire_wake_lease(lease.lease_id, &lease.target, lease.expires_at_ms, now_ms)
            }
            LifecycleWakeLeaseAction::Release => {
                if !self.targets.contains_key(&lease.target) {
                    return Err("unknown lifecycle target".into());
                }
                self.release_wake_lease(&lease.lease_id, &lease.target, now_ms);
                Ok(())
            }
        }
    }

    pub fn release_wake_lease(&mut self, lease_id: &str, target: &LifecycleTarget, now_ms: u64) {
        self.revoked_wake_leases
            .remove(&(target.clone(), lease_id.to_owned()));
        if let Some(leases) = self.wake_leases.get_mut(target) {
            leases.remove(lease_id);
        }
        self.reconcile_target(target, now_ms);
    }

    pub fn component_applied(
        &mut self,
        target: &LifecycleTarget,
        state: LifecycleComponentState,
        now_ms: u64,
    ) {
        let Some(item) = self.targets.get_mut(target) else {
            return;
        };
        item.effective_state = match state {
            LifecycleComponentState::Running => LifecycleEffectiveState::Running,
            LifecycleComponentState::Quiesced => LifecycleEffectiveState::Quiesced,
            LifecycleComponentState::Failed => LifecycleEffectiveState::Failed,
            LifecycleComponentState::Degraded => LifecycleEffectiveState::Degraded,
            LifecycleComponentState::Cancelling => LifecycleEffectiveState::Cancelling,
            LifecycleComponentState::Quiescing => LifecycleEffectiveState::Quiescing,
            LifecycleComponentState::Resuming => LifecycleEffectiveState::Resuming,
            LifecycleComponentState::Unsupported => LifecycleEffectiveState::Failed,
        };
        item.transition_deadline_ms = None;
        let _ = now_ms;
    }

    pub fn tick(&mut self, now_ms: u64) {
        let affected: BTreeSet<_> = self
            .wake_leases
            .iter_mut()
            .filter_map(|(target, leases)| {
                leases.retain(|_, expiry| *expiry > now_ms);
                Some(target.clone())
            })
            .collect();
        for target in affected {
            self.reconcile_target(&target, now_ms);
        }
        for item in self.targets.values_mut() {
            if item
                .transition_deadline_ms
                .is_some_and(|deadline| deadline <= now_ms)
            {
                item.effective_state = LifecycleEffectiveState::Failed;
                item.transition_deadline_ms = None;
                item.timed_out_authority = Some((item.authority_epoch, item.revision));
            }
        }
    }

    fn admit(&mut self, command: &LifecycleCommand, now_ms: u64) -> LifecycleCommandResult {
        if let Err(error) = command.validate() {
            return self.rejected(command, LifecycleReasonCode::InvalidRequest, &error);
        }
        if command.manager_epoch != self.epoch {
            return self.rejected(
                command,
                LifecycleReasonCode::StaleEpoch,
                "manager epoch changed",
            );
        }
        if command.expires_at_ms <= now_ms {
            return self.rejected(command, LifecycleReasonCode::Expired, "command expired");
        }
        let Some(item) = self.targets.get(&command.target) else {
            return self.rejected(
                command,
                LifecycleReasonCode::InvalidTarget,
                "target is not server-advertised",
            );
        };
        if !item.capability.supported || item.capability.always_on {
            return self.rejected(
                command,
                LifecycleReasonCode::Unsupported,
                "target cannot be lifecycle-controlled",
            );
        }
        if command.expected_revision != item.revision {
            return self.rejected(
                command,
                LifecycleReasonCode::Conflict,
                "expected revision is stale",
            );
        }
        let revision = {
            let item = self
                .targets
                .get_mut(&command.target)
                .expect("target checked above");
            item.desired_state = command.desired_state;
            item.revision = item.revision.saturating_add(1);
            item.timed_out_authority = None;
            item.revision
        };
        if command.desired_state == LifecycleDesiredState::Quiesced {
            self.revoke_wake_leases(&command.target);
        }
        self.reconcile_target(&command.target, now_ms);
        LifecycleCommandResult {
            protocol_version: LIFECYCLE_PROTOCOL_VERSION,
            request_id: command.request_id.clone(),
            accepted: true,
            manager_epoch: self.epoch,
            revision,
            reason_code: None,
            detail: None,
        }
    }

    fn revoke_wake_leases(&mut self, target: &LifecycleTarget) {
        if let Some(leases) = self.wake_leases.remove(target) {
            self.revoked_wake_leases.extend(
                leases
                    .into_keys()
                    .map(|lease_id| (target.clone(), lease_id)),
            );
        }
    }

    fn reconcile_target(&mut self, target: &LifecycleTarget, now_ms: u64) {
        let has_wake_lease = self
            .wake_leases
            .get(target)
            .is_some_and(|leases| !leases.is_empty());
        let Some(item) = self.targets.get_mut(target) else {
            return;
        };
        if item.timed_out_authority == Some((item.authority_epoch, item.revision)) {
            return;
        }
        let desired = if has_wake_lease {
            LifecycleDesiredState::Running
        } else {
            item.desired_state
        };
        match desired {
            LifecycleDesiredState::Running
                if item.effective_state != LifecycleEffectiveState::Running =>
            {
                item.effective_state = LifecycleEffectiveState::Resuming;
                item.transition_deadline_ms = Some(now_ms.saturating_add(TRANSITION_TIMEOUT_MS));
            }
            LifecycleDesiredState::Quiesced
                if item.effective_state != LifecycleEffectiveState::Quiesced =>
            {
                item.effective_state = LifecycleEffectiveState::Quiescing;
                item.transition_deadline_ms = Some(now_ms.saturating_add(TRANSITION_TIMEOUT_MS));
            }
            _ => {}
        }
    }

    fn status_for(
        &self,
        target: &LifecycleTarget,
        item: &ManagedTarget,
        now_ms: u64,
    ) -> LifecycleStatus {
        let component_state = match item.effective_state {
            LifecycleEffectiveState::Running => LifecycleComponentState::Running,
            LifecycleEffectiveState::Cancelling => LifecycleComponentState::Cancelling,
            LifecycleEffectiveState::Quiescing => LifecycleComponentState::Quiescing,
            LifecycleEffectiveState::Quiesced => LifecycleComponentState::Quiesced,
            LifecycleEffectiveState::Resuming => LifecycleComponentState::Resuming,
            LifecycleEffectiveState::Degraded => LifecycleComponentState::Degraded,
            LifecycleEffectiveState::Failed | LifecycleEffectiveState::Superseded => {
                LifecycleComponentState::Failed
            }
        };
        LifecycleStatus {
            protocol_version: LIFECYCLE_PROTOCOL_VERSION,
            manager_epoch: item.authority_epoch,
            target: target.clone(),
            revision: item.revision,
            desired_state: item.desired_state,
            effective_state: item.effective_state,
            components: vec![LifecycleComponentStatus {
                node_id: target.node_id.clone(),
                state: component_state,
                reason_code: None,
            }],
            updated_at_ms: now_ms,
        }
    }

    fn rejected(
        &self,
        command: &LifecycleCommand,
        reason_code: LifecycleReasonCode,
        detail: &str,
    ) -> LifecycleCommandResult {
        LifecycleCommandResult {
            protocol_version: LIFECYCLE_PROTOCOL_VERSION,
            request_id: command.request_id.clone(),
            accepted: false,
            manager_epoch: self.epoch,
            revision: self
                .targets
                .get(&command.target)
                .map_or(0, |item| item.revision),
            reason_code: Some(reason_code),
            detail: Some(detail.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::LifecycleRole;

    fn target() -> LifecycleTarget {
        LifecycleTarget {
            role: LifecycleRole::Rover,
            entity_id: "r1".into(),
            node_id: "camera".into(),
        }
    }
    fn command(
        epoch: u64,
        revision: u64,
        desired_state: LifecycleDesiredState,
    ) -> LifecycleCommand {
        LifecycleCommand {
            protocol_version: 1,
            request_id: "f4f3e2d1-c0b9-48a7-9615-141312111000".into(),
            manager_epoch: epoch,
            target: target(),
            desired_state,
            expected_revision: revision,
            issued_at_ms: 1,
            expires_at_ms: 10_000,
        }
    }

    #[test]
    fn duplicate_and_conflicting_commands_are_safe() {
        let mut manager = LifecycleManager::new(
            9,
            vec![LifecycleCapability {
                target: target(),
                supported: true,
                always_on: false,
            }],
        )
        .unwrap();
        let accepted = manager.apply(command(9, 0, LifecycleDesiredState::Quiesced), 2);
        assert!(accepted.accepted);
        assert_eq!(
            manager.apply(command(9, 0, LifecycleDesiredState::Quiesced), 2),
            accepted
        );
        let mut stale = command(9, 0, LifecycleDesiredState::Running);
        stale.request_id = "f4f3e2d1-c0b9-48a7-9615-141312111001".into();
        assert_eq!(
            manager.apply(stale, 2).reason_code,
            Some(LifecycleReasonCode::Conflict)
        );
    }

    #[test]
    fn final_wake_lease_release_reconciles_latest_user_state() {
        let mut manager = LifecycleManager::new(
            9,
            vec![LifecycleCapability {
                target: target(),
                supported: true,
                always_on: false,
            }],
        )
        .unwrap();
        assert!(
            manager
                .apply(command(9, 0, LifecycleDesiredState::Quiesced), 2)
                .accepted
        );
        manager
            .acquire_wake_lease("occurrence-a".into(), &target(), 100, 3)
            .unwrap();
        assert_eq!(
            manager.status(&target(), 3).unwrap().effective_state,
            LifecycleEffectiveState::Resuming
        );
        manager.release_wake_lease("occurrence-a", &target(), 4);
        assert_eq!(
            manager.status(&target(), 4).unwrap().effective_state,
            LifecycleEffectiveState::Quiescing
        );

        manager
            .acquire_wake_lease("occurrence-b".into(), &target(), 100, 5)
            .unwrap();
        let mut user_pause = command(9, 1, LifecycleDesiredState::Quiesced);
        user_pause.request_id = "f4f3e2d1-c0b9-48a7-9615-141312111005".into();
        assert!(manager.apply(user_pause, 6).accepted);
        assert!(manager
            .acquire_wake_lease("occurrence-b".into(), &target(), 100, 7)
            .is_err());
        manager.release_wake_lease("occurrence-b", &target(), 8);
    }

    #[test]
    fn stale_epoch_expiry_and_transition_timeout_never_report_quiesced() {
        let mut manager = LifecycleManager::new(
            9,
            vec![LifecycleCapability {
                target: target(),
                supported: true,
                always_on: false,
            }],
        )
        .unwrap();
        let mut stale = command(8, 0, LifecycleDesiredState::Quiesced);
        stale.request_id = "f4f3e2d1-c0b9-48a7-9615-141312111002".into();
        assert_eq!(
            manager.apply(stale, 2).reason_code,
            Some(LifecycleReasonCode::StaleEpoch)
        );
        let mut expired = command(9, 0, LifecycleDesiredState::Quiesced);
        expired.request_id = "f4f3e2d1-c0b9-48a7-9615-141312111003".into();
        expired.expires_at_ms = 2;
        assert_eq!(
            manager.apply(expired, 2).reason_code,
            Some(LifecycleReasonCode::Expired)
        );
        assert!(
            manager
                .apply(command(9, 0, LifecycleDesiredState::Quiesced), 2)
                .accepted
        );
        manager.tick(30_002);
        assert_eq!(
            manager.status(&target(), 30_002).unwrap().effective_state,
            LifecycleEffectiveState::Failed
        );
    }

    #[test]
    fn timeout_rejects_late_quiesced_status_until_newer_authority() {
        let mut manager = LifecycleManager::new(
            9,
            vec![LifecycleCapability {
                target: target(),
                supported: true,
                always_on: false,
            }],
        )
        .unwrap();
        assert!(
            manager
                .apply(command(9, 0, LifecycleDesiredState::Quiesced), 2)
                .accepted
        );
        manager.tick(30_002);

        let mut late = manager.status(&target(), 30_002).unwrap();
        late.components[0].state = LifecycleComponentState::Quiesced;
        assert!(!manager.apply_component_status(&late));
        assert_eq!(
            manager.status(&target(), 30_002).unwrap().effective_state,
            LifecycleEffectiveState::Failed
        );

        let mut resume = command(9, 1, LifecycleDesiredState::Running);
        resume.request_id = "f4f3e2d1-c0b9-48a7-9615-141312111006".into();
        resume.issued_at_ms = 30_003;
        resume.expires_at_ms = 40_003;
        assert!(manager.apply(resume, 30_003).accepted);
        let mut current = manager.status(&target(), 30_003).unwrap();
        current.components[0].state = LifecycleComponentState::Running;
        assert!(manager.apply_component_status(&current));
        assert_eq!(
            manager.status(&target(), 30_003).unwrap().effective_state,
            LifecycleEffectiveState::Running
        );
    }

    #[test]
    fn component_status_cannot_contradict_current_desired_state() {
        let mut manager = LifecycleManager::new(
            9,
            vec![LifecycleCapability {
                target: target(),
                supported: true,
                always_on: false,
            }],
        )
        .unwrap();
        assert!(
            manager
                .apply(command(9, 0, LifecycleDesiredState::Quiesced), 2)
                .accepted
        );
        let mut contradictory = manager.status(&target(), 2).unwrap();
        contradictory.desired_state = LifecycleDesiredState::Running;
        contradictory.components[0].state = LifecycleComponentState::Running;

        assert!(!manager.apply_component_status(&contradictory));
        assert_eq!(
            manager.status(&target(), 2).unwrap().effective_state,
            LifecycleEffectiveState::Quiescing
        );
    }

    #[test]
    fn relayed_command_preserves_orchestra_epoch_and_rejects_stale_component_status() {
        let mut manager = LifecycleManager::new(
            90,
            vec![LifecycleCapability {
                target: target(),
                supported: true,
                always_on: false,
            }],
        )
        .unwrap();
        let relayed_command = command(7, 0, LifecycleDesiredState::Quiesced);
        assert!(manager.apply_relayed(relayed_command, 2).accepted);
        let status = manager.status(&target(), 2).unwrap();
        assert_eq!(status.manager_epoch, 7);
        let mut stale = status.clone();
        stale.manager_epoch = 6;
        stale.components[0].state = LifecycleComponentState::Quiesced;
        assert!(!manager.apply_component_status(&stale));
        assert_eq!(
            manager.status(&target(), 2).unwrap().effective_state,
            LifecycleEffectiveState::Quiescing
        );
        let mut replay = command(7, 5, LifecycleDesiredState::Running);
        replay.request_id = "f4f3e2d1-c0b9-48a7-9615-141312111004".into();
        assert!(manager.apply_relayed(replay, 3).accepted);
        assert_eq!(manager.status(&target(), 3).unwrap().revision, 6);

        let mut wrong_component = manager.status(&target(), 3).unwrap();
        wrong_component.components[0].node_id = "other-workload".into();
        assert!(!manager.apply_component_status(&wrong_component));
    }

    #[test]
    fn expired_request_cache_entries_do_not_exhaust_admission() {
        let mut manager = LifecycleManager::new(
            9,
            vec![LifecycleCapability {
                target: target(),
                supported: true,
                always_on: false,
            }],
        )
        .unwrap();
        for index in 0..MAX_CACHED_REQUESTS {
            let mut cached = command(9, 0, LifecycleDesiredState::Quiesced);
            cached.request_id = format!("f4f3e2d1-c0b9-48a7-9615-{index:012x}");
            cached.expires_at_ms = 100;
            manager.apply(cached, 2);
        }
        let mut after_expiry = command(9, 1, LifecycleDesiredState::Running);
        after_expiry.request_id = "f4f3e2d1-c0b9-48a7-9615-ffffffffffff".into();
        after_expiry.issued_at_ms = 101;
        after_expiry.expires_at_ms = 200;
        assert!(manager.apply(after_expiry, 101).accepted);
    }
}
