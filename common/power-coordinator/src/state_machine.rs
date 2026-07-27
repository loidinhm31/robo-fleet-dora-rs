use crate::readiness::fresh_low_cpu;
use crate::{
    plan_transition, target, CoordinatorConfig, DemandLedger, ProfileCatalog, TransitionPlan,
};
use robo_rover_lib::{
    occurrence_requires_protection, LifecycleCommand, LifecycleCommandOrigin,
    LifecycleCommandResult, LifecycleEffectiveState, LifecycleRole, LifecycleStatus,
    PowerAuthority, PowerAuthorityDecision, PowerAuthoritySnapshot, PowerCommand,
    PowerCommandAction, PowerCommandResult, PowerPolicy, PowerProfile, PowerReasonCode,
    PowerSnapshotGate, PowerState, PowerStatus, PowerTransition, ProtectedWorkSnapshot,
    RecordingOccurrence, ResourceSnapshot, MAX_PROTECTED_WORK_ITEMS, POWER_PROTOCOL_VERSION,
};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
pub struct CoordinatorTime {
    pub wall_ms: u64,
    pub monotonic_ms: u64,
}

#[derive(Debug, Clone)]
struct PendingTransition {
    id: String,
    target_profile: PowerProfile,
    plan: TransitionPlan,
    stage: usize,
    issued: bool,
    deadline_ms: u64,
    retry_after_ms: u64,
    retries: u8,
}

#[derive(Debug, Clone)]
struct IssuedLifecycleCommand {
    node_id: String,
    manager_epoch: u64,
    expected_revision: u64,
    transition_id: String,
}

#[derive(Debug, Clone)]
struct CachedPowerCommand {
    command: PowerCommand,
    result: PowerCommandResult,
}

#[derive(Debug, Clone, Copy)]
struct ProtectedOperation {
    updated_at_ms: u64,
    active: bool,
}

const MAX_CACHED_POWER_COMMANDS: usize = 1_024;

#[derive(Debug, Clone)]
pub struct CoordinatorEffects {
    pub status: PowerStatus,
    pub transition: Option<PowerTransition>,
    pub lifecycle_commands: Vec<LifecycleCommand>,
    pub query_lifecycle_status: bool,
}

#[derive(Clone)]
pub struct PowerCoordinator {
    config: CoordinatorConfig,
    catalog: ProfileCatalog,
    ledger: DemandLedger,
    authority: PowerAuthority,
    policy: PowerPolicy,
    requested: PowerProfile,
    effective: PowerProfile,
    state: PowerState,
    resources: Option<ResourceSnapshot>,
    lifecycle: BTreeMap<String, LifecycleStatus>,
    command_targets: BTreeMap<String, IssuedLifecycleCommand>,
    command_replays: BTreeMap<String, CachedPowerCommand>,
    authority_gate: Option<PowerSnapshotGate>,
    authority_gate_reason: Option<PowerReasonCode>,
    idle_since_ms: Option<u64>,
    last_resource_sequence: Option<u64>,
    low_samples: u32,
    awake_since_ms: u64,
    protected_operations: BTreeMap<String, ProtectedOperation>,
    protected_work_capacity_blocked: bool,
    protected_snapshot_at_ms: u64,
    pending: Option<PendingTransition>,
    announced: Option<PowerTransition>,
    journal_capacity_unsafe: bool,
}

impl PowerCoordinator {
    pub fn new(config: CoordinatorConfig, epoch: u64) -> Result<Self, String> {
        let catalog = ProfileCatalog::for_role(config.role)?;
        let normal = normal_profile(config.role);
        let effective = initial_profile(config.role);
        Ok(Self {
            config,
            catalog,
            ledger: DemandLedger::default(),
            authority: PowerAuthority {
                epoch: epoch.max(1),
                sequence: 1,
            },
            policy: PowerPolicy::Awake,
            requested: normal,
            effective,
            state: PowerState::Waking,
            resources: None,
            lifecycle: BTreeMap::new(),
            command_targets: BTreeMap::new(),
            command_replays: BTreeMap::new(),
            authority_gate: None,
            authority_gate_reason: None,
            idle_since_ms: None,
            last_resource_sequence: None,
            low_samples: 0,
            awake_since_ms: 0,
            protected_operations: BTreeMap::new(),
            protected_work_capacity_blocked: false,
            protected_snapshot_at_ms: 0,
            pending: None,
            announced: None,
            journal_capacity_unsafe: false,
        })
    }

    pub fn observe_resources(&mut self, snapshot: ResourceSnapshot) {
        if snapshot.role == role_as_resource(self.config.role)
            && snapshot.entity_id == self.config.entity_id
            && snapshot.validate().is_ok()
        {
            if snapshot.sampled_at_ms >= 0
                && self.resources.as_ref().is_none_or(|current| {
                    snapshot.sequence > current.sequence
                        && snapshot.sampled_at_ms >= current.sampled_at_ms
                })
            {
                self.resources = Some(snapshot);
            }
        }
    }

    pub fn observe_lifecycle(&mut self, status: LifecycleStatus) {
        if status.target.role == self.config.role
            && status.target.entity_id == self.config.entity_id
            && status.validate().is_ok()
        {
            if self
                .lifecycle
                .get(&status.target.node_id)
                .is_some_and(|current| {
                    (
                        current.manager_epoch,
                        current.revision,
                        current.updated_at_ms,
                    ) > (status.manager_epoch, status.revision, status.updated_at_ms)
                })
            {
                return;
            }
            self.lifecycle.insert(status.target.node_id.clone(), status);
        }
    }

    pub fn observe_lifecycle_result(&mut self, result: LifecycleCommandResult) {
        if result.validate().is_err() {
            return;
        }
        let Some(issued) = self.command_targets.get(&result.request_id) else {
            return;
        };
        if !result.accepted
            || result.manager_epoch != issued.manager_epoch
            || result.revision != issued.expected_revision.saturating_add(1)
        {
            return;
        }
        let issued = self
            .command_targets
            .remove(&result.request_id)
            .expect("issued lifecycle command was checked");
        let Some(status) = self.lifecycle.get_mut(&issued.node_id) else {
            return;
        };
        if (status.manager_epoch, status.revision) > (result.manager_epoch, result.revision) {
            return;
        }
        status.manager_epoch = result.manager_epoch;
        status.revision = result.revision;
        if let Some(pending) = self.pending.as_mut() {
            let current_stage = &pending.plan.stages[pending.stage];
            if pending.issued
                && pending.id != issued.transition_id
                && current_stage.nodes.contains(&issued.node_id)
            {
                pending.issued = false;
            }
        }
    }

    pub fn set_protected_operation(&mut self, active: bool) {
        self.observe_protected_operation("legacy", active);
    }

    pub fn observe_protected_operation(&mut self, operation_id: impl Into<String>, active: bool) {
        self.apply_protected_operation(operation_id.into(), active, u64::MAX);
    }

    pub fn observe_protected_occurrence(&mut self, occurrence: RecordingOccurrence) {
        if occurrence.validate().is_err() || occurrence.updated_at_ms < 0 {
            return;
        }
        self.apply_protected_operation(
            occurrence.occurrence_id,
            occurrence_requires_protection(occurrence.state),
            occurrence.updated_at_ms as u64,
        );
    }

    pub fn observe_protected_work_snapshot(&mut self, snapshot: ProtectedWorkSnapshot) {
        if snapshot.validate().is_err() || snapshot.generated_at_ms < self.protected_snapshot_at_ms
        {
            return;
        }

        // A snapshot replaces the sender's view atomically. Retain only updates that
        // arrived after the snapshot was generated, so a full local map cannot cause
        // the snapshot reconciliation to discard every protected operation.
        let mut reconciled = snapshot
            .occurrences
            .into_iter()
            .map(|occurrence| {
                (
                    occurrence.occurrence_id,
                    ProtectedOperation {
                        updated_at_ms: occurrence.updated_at_ms as u64,
                        active: occurrence_requires_protection(occurrence.state),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (operation_id, operation) in &self.protected_operations {
            // Equal timestamps have no causal ordering guarantee across the relay.
            // Retaining them biases safely toward keeping Auto quiesce blocked until
            // a later snapshot can authoritatively clear the operation.
            if operation.updated_at_ms >= snapshot.generated_at_ms {
                reconciled.insert(operation_id.clone(), operation.clone());
            }
        }
        if reconciled.len() > MAX_PROTECTED_WORK_ITEMS {
            self.protected_work_capacity_blocked = true;
            return;
        }
        self.protected_operations = reconciled;
        self.protected_work_capacity_blocked = false;
        self.protected_snapshot_at_ms = snapshot.generated_at_ms;
    }

    fn apply_protected_operation(
        &mut self,
        operation_id: String,
        active: bool,
        updated_at_ms: u64,
    ) {
        if self
            .protected_operations
            .get(&operation_id)
            .is_some_and(|current| current.updated_at_ms > updated_at_ms)
        {
            return;
        }
        if !self.protected_operations.contains_key(&operation_id)
            && self.protected_operations.len() >= MAX_PROTECTED_WORK_ITEMS
        {
            self.protected_work_capacity_blocked = true;
            return;
        }
        self.protected_operations.insert(
            operation_id,
            ProtectedOperation {
                updated_at_ms,
                active,
            },
        );
    }

    pub fn set_journal_capacity_unsafe(&mut self, unsafe_capacity: bool) {
        self.journal_capacity_unsafe = unsafe_capacity;
    }

    pub fn current_status(&self, now_ms: u64) -> PowerStatus {
        self.status(now_ms)
    }

    pub fn next_authority(&self) -> PowerAuthority {
        self.authority
            .next_sequence()
            .or_else(|| self.authority.next_epoch())
            .unwrap_or(self.authority)
    }

    pub fn authority_epoch(&self) -> u64 {
        self.authority.epoch
    }

    /// Produces the bounded, target-scoped observation used by a remote
    /// authority gate. It contains no policy command and is safe to repeat.
    pub fn authority_snapshot(&self, now_ms: u64) -> PowerAuthoritySnapshot {
        PowerAuthoritySnapshot {
            protocol_version: POWER_PROTOCOL_VERSION,
            snapshot_id: Uuid::new_v4().hyphenated().to_string(),
            role: self.config.role,
            entity_id: self.config.entity_id.clone(),
            authority: self.authority,
            state: self.state,
            effective_profile: self.effective,
            captured_at_ms: now_ms,
            expires_at_ms: now_ms.saturating_add(30_000),
        }
    }

    /// Enables the snapshot-first remote authority boundary. Phase 04 supplies
    /// the transport input; this state-machine API remains transport agnostic.
    pub fn require_authority_snapshot(
        &mut self,
        remote_role: LifecycleRole,
        remote_entity_id: String,
    ) -> Result<(), String> {
        self.authority_gate = Some(PowerSnapshotGate::new(remote_role, remote_entity_id)?);
        self.authority_gate_reason = Some(PowerReasonCode::SnapshotMissing);
        Ok(())
    }

    /// Invalid snapshots are intentionally converted to bounded telemetry and
    /// never escape the coordinator event loop.
    pub fn observe_authority_snapshot(
        &mut self,
        snapshot: PowerAuthoritySnapshot,
        now_ms: u64,
    ) -> PowerAuthorityDecision {
        let Some(gate) = self.authority_gate.as_mut() else {
            return PowerAuthorityDecision::ObserveOnly;
        };
        let observed_authority = snapshot.authority;
        match gate.observe(snapshot, now_ms) {
            Ok(()) => {
                if observed_authority.epoch >= self.authority.epoch {
                    let Some(next_epoch) = observed_authority.next_epoch() else {
                        self.authority_gate_reason = Some(PowerReasonCode::InvalidAuthority);
                        return PowerAuthorityDecision::ObserveOnly;
                    };
                    self.authority = next_epoch;
                }
                self.authority_gate_reason = None;
                PowerAuthorityDecision::ObserveOnly
            }
            Err(error) => {
                self.authority_gate_reason = Some(snapshot_reason(&error));
                PowerAuthorityDecision::ObserveOnly
            }
        }
    }

    /// Returns the one-shot authorization Phase 04 must obtain before mapping
    /// a profile decision onto a bridge or direct-mode command.
    pub fn authorize_remote_profile_command(
        &mut self,
        proposed_authority: PowerAuthority,
        now_ms: u64,
    ) -> PowerAuthorityDecision {
        let Some(gate) = self.authority_gate.as_mut() else {
            return PowerAuthorityDecision::ObserveOnly;
        };
        let decision = gate.consume_profile_authority(proposed_authority, now_ms);
        if decision == PowerAuthorityDecision::ObserveOnly {
            self.authority_gate_reason =
                Some(if gate.state(now_ms) == PowerState::AuthorityUnknown {
                    PowerReasonCode::SnapshotStale
                } else {
                    PowerReasonCode::StaleAuthority
                });
        }
        decision
    }

    pub fn command_will_apply(
        &self,
        command: &PowerCommand,
        now: CoordinatorTime,
    ) -> Result<(), String> {
        command.validates_for(self.config.role, &self.config.entity_id)?;
        if !self.authority.accepts_command_authority(command.authority) {
            return Err("power command authority is not the exact successor".into());
        }
        if command.expires_at_ms <= now.wall_ms || command.not_before_ms > now.wall_ms {
            return Err("power command is not active".into());
        }
        let mut ledger = self.ledger.clone();
        match &command.action {
            PowerCommandAction::SetPolicy { policy }
                if *policy == PowerPolicy::Sleep && self.journal_capacity_unsafe =>
            {
                Err("journal CapacityExceeded: sleep is inhibited".into())
            }
            PowerCommandAction::SetPolicy { .. } => Ok(()),
            PowerCommandAction::RegisterDemand { demand }
                if demand.authority != command.authority =>
            {
                Err("embedded demand authority differs".into())
            }
            PowerCommandAction::RegisterDemand { demand } => ledger
                .apply(demand.clone(), now.wall_ms)
                .map(|_| ())
                .map_err(|reason| format!("{reason:?}")),
            PowerCommandAction::ReleaseDemand { demand_id } => ledger
                .release(&self.config.entity_id, demand_id)
                .map(|_| ())
                .map_err(|reason| format!("{reason:?}")),
            PowerCommandAction::RegisterReservation { reservation }
                if reservation.authority != command.authority =>
            {
                Err("embedded reservation authority differs".into())
            }
            PowerCommandAction::RegisterReservation { reservation } => ledger
                .register_reservation(reservation.clone(), now.wall_ms)
                .map(|_| ())
                .map_err(|reason| format!("{reason:?}")),
            PowerCommandAction::ReleaseReservation { reservation_id } => ledger
                .release_reservation(reservation_id, now.wall_ms)
                .map(|_| ())
                .map_err(|reason| format!("{reason:?}")),
        }
    }

    pub fn apply_command(
        &mut self,
        command: PowerCommand,
        now: CoordinatorTime,
    ) -> PowerCommandResult {
        if let Some(result) = self.replay_result(&command, now.wall_ms) {
            return result;
        }
        if self.command_replays.len() >= MAX_CACHED_POWER_COMMANDS {
            return rejected_command(
                command,
                self.authority,
                PowerReasonCode::CapacityExceeded,
                "power command replay cache is full",
            );
        }
        let command_id = command.command_id.clone();
        let accepted = command
            .validates_for(self.config.role, &self.config.entity_id)
            .and_then(|_| {
                if !self.authority.accepts_command_authority(command.authority) {
                    return Err("power command authority is not the exact successor".into());
                }
                if command.expires_at_ms <= now.wall_ms {
                    return Err("power command expired".into());
                }
                if command.not_before_ms > now.wall_ms {
                    return Err("power command is not active".into());
                }
                if command.authority != self.authority {
                    self.authority = command.authority;
                }
                match &command.action {
                    PowerCommandAction::SetPolicy { policy } => {
                        if *policy == PowerPolicy::Sleep && self.journal_capacity_unsafe {
                            return Err("journal CapacityExceeded: sleep is inhibited".into());
                        }
                        self.policy = *policy;
                        Ok(())
                    }
                    PowerCommandAction::RegisterDemand { demand }
                        if demand.authority != command.authority =>
                    {
                        Err("embedded demand authority differs".into())
                    }
                    PowerCommandAction::RegisterDemand { demand } => self
                        .ledger
                        .apply(demand.clone(), now.wall_ms)
                        .map(|_| ())
                        .map_err(|reason| format!("{reason:?}")),
                    PowerCommandAction::ReleaseDemand { demand_id } => self
                        .ledger
                        .release(&self.config.entity_id, demand_id)
                        .map(|_| ())
                        .map_err(|reason| format!("{reason:?}")),
                    PowerCommandAction::RegisterReservation { reservation }
                        if reservation.authority != command.authority =>
                    {
                        Err("embedded reservation authority differs".into())
                    }
                    PowerCommandAction::RegisterReservation { reservation } => self
                        .ledger
                        .register_reservation(reservation.clone(), now.wall_ms)
                        .map(|_| ())
                        .map_err(|reason| format!("{reason:?}")),
                    PowerCommandAction::ReleaseReservation { reservation_id } => self
                        .ledger
                        .release_reservation(reservation_id, now.wall_ms)
                        .map(|_| ())
                        .map_err(|reason| format!("{reason:?}")),
                }
            });
        let result = match accepted {
            Ok(()) => PowerCommandResult {
                protocol_version: POWER_PROTOCOL_VERSION,
                command_id: command_id.clone(),
                accepted: true,
                authority: self.bump_authority(),
                reason_code: None,
                detail: None,
            },
            Err(detail) => PowerCommandResult {
                protocol_version: POWER_PROTOCOL_VERSION,
                command_id: command_id.clone(),
                accepted: false,
                authority: self.authority,
                reason_code: Some(reason_from_error(&detail)),
                detail: Some(detail),
            },
        };
        self.command_replays.insert(
            command_id,
            CachedPowerCommand {
                command,
                result: result.clone(),
            },
        );
        result
    }

    /// Returns an immutable prior command result without creating a new
    /// journal intent. Expired cache entries cannot consume admission space.
    pub fn replay_result(
        &mut self,
        command: &PowerCommand,
        now_ms: u64,
    ) -> Option<PowerCommandResult> {
        self.command_replays
            .retain(|_, cached| cached.command.expires_at_ms > now_ms);
        self.command_replays.get(&command.command_id).map(|cached| {
            if cached.command == *command {
                cached.result.clone()
            } else {
                rejected_command(
                    command.clone(),
                    self.authority,
                    PowerReasonCode::DuplicateMismatch,
                    "command id payload changed",
                )
            }
        })
    }

    pub fn tick(&mut self, now: CoordinatorTime) -> CoordinatorEffects {
        if self
            .authority_gate
            .as_ref()
            .is_some_and(|gate| gate.state(now.wall_ms) == PowerState::AuthorityUnknown)
        {
            self.state = PowerState::AuthorityUnknown;
            self.authority_gate_reason
                .get_or_insert(PowerReasonCode::SnapshotMissing);
            return CoordinatorEffects {
                status: self.status(now.wall_ms),
                transition: None,
                lifecycle_commands: vec![],
                query_lifecycle_status: true,
            };
        }
        self.reduce(now);
        let commands = self.advance_transition(now);
        CoordinatorEffects {
            status: self.status(now.wall_ms),
            transition: self.announced.take(),
            lifecycle_commands: commands,
            query_lifecycle_status: true,
        }
    }

    fn reduce(&mut self, now: CoordinatorTime) {
        let active_profiles = self
            .ledger
            .active_demands(now.wall_ms)
            .map(|demand| demand.required_profile)
            .collect::<Vec<_>>();
        let reservation_active = self
            .ledger
            .active_reservations(now.wall_ms)
            .next()
            .is_some();
        let normal = normal_profile(self.config.role);
        let target = match self.policy {
            PowerPolicy::Awake => {
                self.cancel_idle();
                normal
            }
            PowerPolicy::Sleep if reservation_active => {
                self.cancel_idle();
                PowerProfile::ScheduledCapture
            }
            PowerPolicy::Sleep => {
                self.cancel_idle();
                low_power_profile(self.config.role)
            }
            PowerPolicy::Auto
                if self.protected_work_capacity_blocked
                    || self.protected_operations.values().any(|item| item.active)
                    || !active_profiles.is_empty()
                    || reservation_active =>
            {
                self.cancel_idle();
                self.least_profile(active_profiles, reservation_active)
            }
            PowerPolicy::Auto => self.auto_target(now, normal),
        };
        self.requested = target;
        let cancelled_pending = self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.target_profile != self.requested);
        if cancelled_pending {
            self.pending = None;
        }
        if self.pending.is_none() && (self.requested != self.effective || cancelled_pending) {
            let from = cancelled_pending.then_some(PowerProfile::Dormant);
            self.start_transition(now, from.unwrap_or(self.effective));
        }
        if self.pending.is_none()
            && !(self.policy == PowerPolicy::Auto
                && self.idle_since_ms.is_some()
                && self.requested == normal_profile(self.config.role)
                && self.effective == normal_profile(self.config.role))
        {
            self.state = state_for(self.policy, self.effective, self.requested);
        }
    }

    fn auto_target(&mut self, now: CoordinatorTime, normal: PowerProfile) -> PowerProfile {
        if self.journal_capacity_unsafe {
            self.cancel_idle();
            self.state = PowerState::Active;
            return normal;
        }
        if now.monotonic_ms.saturating_sub(self.awake_since_ms) < self.config.min_awake_ms {
            self.state = PowerState::Active;
            return normal;
        }
        let Some(started) = self.idle_since_ms else {
            self.idle_since_ms = Some(now.monotonic_ms);
            self.low_samples = 0;
            self.state = PowerState::IdlePending;
            return normal;
        };
        let low = fresh_low_cpu(
            self.resources.as_ref(),
            &self.catalog,
            &self.config,
            normal,
            now.wall_ms,
        );
        let fresh_sequence = self
            .resources
            .as_ref()
            .map(|item| item.sequence)
            .filter(|sequence| Some(*sequence) != self.last_resource_sequence);
        if low {
            if let Some(sequence) = fresh_sequence {
                self.last_resource_sequence = Some(sequence);
                self.low_samples = self.low_samples.saturating_add(1);
            }
        } else {
            self.idle_since_ms = Some(now.monotonic_ms);
            self.low_samples = 0;
            self.state = PowerState::Active;
            return normal;
        }
        if now.monotonic_ms.saturating_sub(started) < self.config.idle_grace_ms
            || self.low_samples < self.config.required_low_samples
        {
            self.state = PowerState::IdlePending;
            normal
        } else {
            auto_idle_profile(self.config.role)
        }
    }

    fn least_profile(&self, profiles: Vec<PowerProfile>, reservation_active: bool) -> PowerProfile {
        profiles
            .into_iter()
            .chain(reservation_active.then_some(PowerProfile::ScheduledCapture))
            .max_by_key(|profile| self.catalog.rank(*profile))
            .unwrap_or_else(|| normal_profile(self.config.role))
    }

    fn cancel_idle(&mut self) {
        self.idle_since_ms = None;
        self.low_samples = 0;
    }

    fn start_transition(&mut self, now: CoordinatorTime, from: PowerProfile) {
        let plan = plan_transition(&self.catalog, from, self.requested);
        if plan.stages.is_empty() {
            self.effective = self.requested;
            return;
        }
        if self.catalog.rank(self.requested) < self.catalog.rank(from) {
            // A sleep/quiesce epoch fences transient control replay. A wake
            // must arrive as a newly authorized command, never from cache.
            self.command_replays.clear();
        }
        let id = Uuid::new_v4().hyphenated().to_string();
        self.bump_authority();
        self.announced = Some(PowerTransition {
            protocol_version: POWER_PROTOCOL_VERSION,
            transition_id: id.clone(),
            role: self.config.role,
            entity_id: self.config.entity_id.clone(),
            authority: self.authority,
            requested_profile: self.requested,
            effective_profile: self.effective,
            state: state_for(self.policy, from, self.requested),
            issued_at_ms: now.wall_ms,
            deadline_at_ms: now
                .wall_ms
                .saturating_add(self.config.transition_timeout_ms),
        });
        self.pending = Some(PendingTransition {
            id,
            target_profile: self.requested,
            plan,
            stage: 0,
            issued: false,
            deadline_ms: now
                .monotonic_ms
                .saturating_add(self.config.transition_timeout_ms),
            retry_after_ms: now.monotonic_ms,
            retries: 0,
        });
        self.state = state_for(self.policy, from, self.requested);
    }

    fn advance_transition(&mut self, now: CoordinatorTime) -> Vec<LifecycleCommand> {
        let Some(mut pending) = self.pending.take() else {
            return vec![];
        };
        let desired = pending.plan.stages[pending.stage].desired_state;
        if pending.issued && self.stage_terminal(&pending, desired) {
            pending.stage += 1;
            pending.issued = false;
        }
        if pending.stage == pending.plan.stages.len() {
            self.effective = pending.target_profile;
            self.bump_authority();
            if self.effective == normal_profile(self.config.role) {
                self.awake_since_ms = now.monotonic_ms;
            }
            self.state = state_for(self.policy, self.effective, self.requested);
            return vec![];
        }
        if pending.issued
            && (self.stage_failed(&pending) || now.monotonic_ms >= pending.deadline_ms)
        {
            if pending.retries >= self.config.max_transition_retries {
                self.state = PowerState::Failed;
                self.pending = Some(pending);
                return vec![];
            }
            pending.retries += 1;
            pending.issued = false;
            pending.retry_after_ms = now.monotonic_ms.saturating_add(
                self.config
                    .retry_backoff_ms
                    .saturating_mul(pending.retries as u64),
            );
            self.state = PowerState::Degraded;
        }
        let commands = if !pending.issued && now.monotonic_ms >= pending.retry_after_ms {
            self.issue_stage(&mut pending, desired, now)
        } else {
            vec![]
        };
        self.pending = Some(pending);
        commands
    }

    fn issue_stage(
        &mut self,
        pending: &mut PendingTransition,
        desired: robo_rover_lib::LifecycleDesiredState,
        now: CoordinatorTime,
    ) -> Vec<LifecycleCommand> {
        let stage = &pending.plan.stages[pending.stage].nodes;
        let statuses = stage
            .iter()
            .map(|node| self.lifecycle.get(node))
            .collect::<Option<Vec<_>>>();
        let Some(statuses) = statuses else {
            return vec![];
        };
        pending.issued = true;
        pending.deadline_ms = now
            .monotonic_ms
            .saturating_add(self.config.transition_timeout_ms);
        let commands: Vec<_> = statuses
            .into_iter()
            .map(|status| LifecycleCommand {
                protocol_version: 1,
                request_id: Uuid::new_v4().hyphenated().to_string(),
                manager_epoch: status.manager_epoch,
                target: target(
                    self.config.role,
                    &self.config.entity_id,
                    &status.target.node_id,
                ),
                desired_state: desired,
                expected_revision: status.revision,
                issued_at_ms: now.wall_ms,
                expires_at_ms: now.wall_ms.saturating_add(60_000),
                origin: LifecycleCommandOrigin::Coordinator,
                transition_id: Some(pending.id.clone()),
            })
            .collect();
        for command in &commands {
            self.command_targets.insert(
                command.request_id.clone(),
                IssuedLifecycleCommand {
                    node_id: command.target.node_id.clone(),
                    manager_epoch: command.manager_epoch,
                    expected_revision: command.expected_revision,
                    transition_id: pending.id.clone(),
                },
            );
        }
        commands
    }

    fn stage_terminal(
        &self,
        pending: &PendingTransition,
        desired: robo_rover_lib::LifecycleDesiredState,
    ) -> bool {
        pending.plan.stages[pending.stage].nodes.iter().all(|node| {
            self.lifecycle.get(node).is_some_and(|status| {
                status.transition_id.as_deref() == Some(&pending.id)
                    && status.desired_state == desired
                    && matches!(
                        (desired, status.effective_state),
                        (
                            robo_rover_lib::LifecycleDesiredState::Running,
                            LifecycleEffectiveState::Running
                        ) | (
                            robo_rover_lib::LifecycleDesiredState::Quiesced,
                            LifecycleEffectiveState::Quiesced
                        )
                    )
            })
        })
    }

    fn stage_failed(&self, pending: &PendingTransition) -> bool {
        pending.plan.stages[pending.stage].nodes.iter().any(|node| {
            self.lifecycle.get(node).is_some_and(|status| {
                status.transition_id.as_deref() == Some(&pending.id)
                    && matches!(
                        status.effective_state,
                        LifecycleEffectiveState::Degraded | LifecycleEffectiveState::Failed
                    )
            })
        })
    }

    fn status(&self, now_ms: u64) -> PowerStatus {
        PowerStatus {
            protocol_version: POWER_PROTOCOL_VERSION,
            role: self.config.role,
            entity_id: self.config.entity_id.clone(),
            authority: self.authority,
            policy: self.policy,
            requested_profile: self.requested,
            effective_profile: self.effective,
            state: self.state,
            transition_id: (self.state != PowerState::AuthorityUnknown)
                .then(|| self.pending.as_ref().map(|item| item.id.clone()))
                .flatten(),
            reason_code: if self.state == PowerState::AuthorityUnknown {
                self.authority_gate_reason
            } else {
                (self.state == PowerState::Failed).then_some(PowerReasonCode::Timeout)
            },
            detail: None,
            updated_at_ms: now_ms,
        }
    }

    fn bump_authority(&mut self) -> PowerAuthority {
        self.authority = self
            .authority
            .next_sequence()
            .or_else(|| self.authority.next_epoch())
            .unwrap_or(self.authority);
        self.authority
    }
}

fn normal_profile(role: robo_rover_lib::LifecycleRole) -> PowerProfile {
    match role {
        robo_rover_lib::LifecycleRole::Rover => PowerProfile::NormalRover,
        robo_rover_lib::LifecycleRole::Orchestra => PowerProfile::OrchestraSpeech,
    }
}
fn initial_profile(role: robo_rover_lib::LifecycleRole) -> PowerProfile {
    match role {
        robo_rover_lib::LifecycleRole::Rover => PowerProfile::Dormant,
        robo_rover_lib::LifecycleRole::Orchestra => PowerProfile::OrchestraSpeech,
    }
}
fn low_power_profile(role: robo_rover_lib::LifecycleRole) -> PowerProfile {
    match role {
        robo_rover_lib::LifecycleRole::Rover => PowerProfile::Dormant,
        robo_rover_lib::LifecycleRole::Orchestra => PowerProfile::OrchestraIdle,
    }
}
fn auto_idle_profile(role: robo_rover_lib::LifecycleRole) -> PowerProfile {
    match role {
        robo_rover_lib::LifecycleRole::Rover => PowerProfile::IdleListening,
        robo_rover_lib::LifecycleRole::Orchestra => PowerProfile::OrchestraIdle,
    }
}
fn role_as_resource(role: robo_rover_lib::LifecycleRole) -> robo_rover_lib::ResourceRole {
    match role {
        robo_rover_lib::LifecycleRole::Rover => robo_rover_lib::ResourceRole::Rover,
        robo_rover_lib::LifecycleRole::Orchestra => robo_rover_lib::ResourceRole::Orchestra,
    }
}
fn state_for(policy: PowerPolicy, effective: PowerProfile, requested: PowerProfile) -> PowerState {
    if requested != effective {
        if requested == PowerProfile::ScheduledCapture && policy == PowerPolicy::Sleep {
            PowerState::Prewarming
        } else if matches!(
            requested,
            PowerProfile::Dormant | PowerProfile::IdleListening | PowerProfile::OrchestraIdle
        ) {
            PowerState::Quiescing
        } else {
            PowerState::Waking
        }
    } else {
        match effective {
            // OrchestraIdle quiesces central speech but does not enter the
            // Rover-only Dormant state (which means KWS is disabled).
            PowerProfile::Dormant => PowerState::Dormant,
            PowerProfile::IdleListening => PowerState::IdleListening,
            _ => PowerState::Active,
        }
    }
}
fn reason_from_error(detail: &str) -> PowerReasonCode {
    if detail.contains("Expired") || detail.contains("expired") {
        PowerReasonCode::Expired
    } else if detail.contains("authority") {
        PowerReasonCode::StaleAuthority
    } else if detail.contains("Duplicate") {
        PowerReasonCode::DuplicateMismatch
    } else if detail.contains("Capacity") {
        PowerReasonCode::CapacityExceeded
    } else {
        PowerReasonCode::InvalidRequest
    }
}

fn snapshot_reason(error: &str) -> PowerReasonCode {
    if error.contains("target") {
        PowerReasonCode::InvalidTarget
    } else if error.contains("fresh") {
        PowerReasonCode::SnapshotStale
    } else if error.contains("authority") {
        PowerReasonCode::StaleAuthority
    } else {
        PowerReasonCode::InvalidRequest
    }
}

fn rejected_command(
    command: PowerCommand,
    authority: PowerAuthority,
    reason_code: PowerReasonCode,
    detail: &str,
) -> PowerCommandResult {
    PowerCommandResult {
        protocol_version: POWER_PROTOCOL_VERSION,
        command_id: command.command_id,
        accepted: false,
        authority,
        reason_code: Some(reason_code),
        detail: Some(detail.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{
        DomainResourceUsage, LifecycleCommandResult, LifecycleComponentState,
        LifecycleDesiredState, LifecycleRole, PowerAuthority, PowerDemand, PowerDemandAction,
        PowerDemandPriority, PowerDemandSource, ResourceRole, ResourceScope, ResourceSource,
    };
    fn time(wall: u64, mono: u64) -> CoordinatorTime {
        CoordinatorTime {
            wall_ms: wall,
            monotonic_ms: mono,
        }
    }
    fn coordinator() -> PowerCoordinator {
        PowerCoordinator::new(
            CoordinatorConfig::for_test(LifecycleRole::Rover, "rover"),
            1,
        )
        .unwrap()
    }
    fn snapshot(sequence: u64, cpu: Option<f32>, sampled: i64) -> ResourceSnapshot {
        let mut domains = BTreeMap::new();
        for domain in [
            "audio-capture",
            "edge-voice",
            "gst-camera",
            "audio-playback",
        ] {
            domains.insert(
                domain.into(),
                DomainResourceUsage {
                    cpu_usage_percent: cpu,
                    memory_rss_bytes: cpu.map(|_| 1),
                    process_count: 1,
                    configured_node_count: 1,
                    sampled_at_ms: sampled,
                },
            );
        }
        ResourceSnapshot {
            schema_version: 1,
            role: ResourceRole::Rover,
            entity_id: "rover".into(),
            scope: ResourceScope::Host,
            source: ResourceSource::Procfs,
            sequence,
            sampled_at_ms: sampled,
            sample_interval_ms: 1,
            cpu_usage_percent: cpu,
            cpu_capacity_cores: Some(1.0),
            memory_used_bytes: None,
            memory_available_bytes: None,
            memory_limit_bytes: None,
            nodes: BTreeMap::new(),
            domains,
        }
    }

    fn orchestra_snapshot(sequence: u64, sampled: i64) -> ResourceSnapshot {
        let mut domains = BTreeMap::new();
        domains.insert(
            "central-speech-recognizer".into(),
            DomainResourceUsage {
                cpu_usage_percent: Some(1.0),
                memory_rss_bytes: Some(1),
                process_count: 1,
                configured_node_count: 1,
                sampled_at_ms: sampled,
            },
        );
        ResourceSnapshot {
            schema_version: 1,
            role: ResourceRole::Orchestra,
            entity_id: "orchestra".into(),
            scope: ResourceScope::Host,
            source: ResourceSource::Procfs,
            sequence,
            sampled_at_ms: sampled,
            sample_interval_ms: 1,
            cpu_usage_percent: Some(1.0),
            cpu_capacity_cores: Some(1.0),
            memory_used_bytes: None,
            memory_available_bytes: None,
            memory_limit_bytes: None,
            nodes: BTreeMap::new(),
            domains,
        }
    }
    fn status(
        node: &str,
        transition_id: Option<String>,
        desired: LifecycleDesiredState,
        effective: LifecycleEffectiveState,
    ) -> LifecycleStatus {
        LifecycleStatus {
            protocol_version: 1,
            manager_epoch: 2,
            target: target(LifecycleRole::Rover, "rover", node),
            revision: 0,
            desired_state: desired,
            effective_state: effective,
            transition_id,
            components: vec![robo_rover_lib::LifecycleComponentStatus {
                node_id: node.into(),
                state: LifecycleComponentState::Running,
                reason_code: None,
            }],
            updated_at_ms: 1,
        }
    }

    fn demand() -> PowerDemand {
        PowerDemand {
            protocol_version: POWER_PROTOCOL_VERSION,
            demand_id: "f4f3e2d1-c0b9-48a7-9615-141312111000".into(),
            action: PowerDemandAction::Acquire,
            source: PowerDemandSource::Ui,
            priority: PowerDemandPriority::Normal,
            role: LifecycleRole::Rover,
            entity_id: "rover".into(),
            required_profile: PowerProfile::NormalRover,
            authority: PowerAuthority {
                epoch: 1,
                sequence: 1,
            },
            issued_at_ms: 1,
            not_before_ms: 1,
            expires_at_ms: 10_000,
            renew_sequence: 1,
        }
    }

    fn single_command(item: &mut PowerCoordinator, now: u64) -> LifecycleCommand {
        let effects = item.tick(time(now, now));
        assert_eq!(effects.lifecycle_commands.len(), 1);
        effects.lifecycle_commands.into_iter().next().unwrap()
    }
    #[test]
    fn auto_never_leaves_idle_pending_before_grace_and_fresh_samples() {
        let mut item = coordinator();
        item.policy = PowerPolicy::Auto;
        item.effective = PowerProfile::NormalRover;
        item.observe_resources(snapshot(1, Some(1.0), 1));
        assert_eq!(item.tick(time(1, 1)).status.state, PowerState::IdlePending);
        item.observe_resources(snapshot(2, Some(1.0), 300_000));
        assert_eq!(
            item.tick(time(300_000, 300_000)).status.state,
            PowerState::IdlePending
        );
        item.observe_resources(snapshot(3, Some(1.0), 300_001));
        assert_eq!(
            item.tick(time(300_001, 300_001)).status.requested_profile,
            PowerProfile::IdleListening
        );
    }
    #[test]
    fn stale_or_high_cpu_cancels_idle_immediately() {
        let mut item = coordinator();
        item.policy = PowerPolicy::Auto;
        item.effective = PowerProfile::NormalRover;
        item.observe_resources(snapshot(1, Some(50.0), 1));
        assert_eq!(item.tick(time(1, 1)).status.state, PowerState::IdlePending);
        assert_eq!(
            item.tick(time(20_000, 20_000)).status.requested_profile,
            PowerProfile::NormalRover
        );
    }
    #[test]
    fn protected_operation_blocks_auto_quiesce_until_released() {
        let mut item = coordinator();
        item.policy = PowerPolicy::Auto;
        item.effective = PowerProfile::NormalRover;
        item.observe_resources(snapshot(1, Some(1.0), 1));
        item.set_protected_operation(true);
        assert_eq!(
            item.tick(time(300_001, 300_001)).status.requested_profile,
            PowerProfile::NormalRover
        );
        item.set_protected_operation(false);
        assert_eq!(
            item.tick(time(300_002, 300_002)).status.state,
            PowerState::IdlePending
        );
    }

    #[test]
    fn protected_work_rejects_stale_updates_and_snapshot_clears_stale_state() {
        let mut item = coordinator();
        item.apply_protected_operation("recording-1".into(), true, 2);
        item.apply_protected_operation("recording-1".into(), false, 3);
        item.apply_protected_operation("recording-1".into(), true, 2);
        assert!(!item.protected_operations["recording-1"].active);

        item.apply_protected_operation("recording-2".into(), true, 4);
        item.observe_protected_work_snapshot(ProtectedWorkSnapshot {
            protocol_version: robo_rover_lib::PROTECTED_WORK_RELAY_PROTOCOL_VERSION,
            snapshot_id: "f4f3e2d1-c0b9-48a7-9615-141312111000".into(),
            entity_id: "rover".into(),
            generated_at_ms: 5,
            occurrences: vec![],
        });
        assert!(item.protected_operations.is_empty());
    }

    #[test]
    fn protected_work_snapshot_reconciles_a_full_map_without_dropping_the_gate() {
        let mut item = coordinator();
        item.policy = PowerPolicy::Auto;
        item.effective = PowerProfile::NormalRover;
        item.observe_resources(snapshot(1, Some(1.0), 1));
        for index in 0..MAX_PROTECTED_WORK_ITEMS {
            item.apply_protected_operation(format!("old-{index}"), true, 1);
        }

        let occurrence: RecordingOccurrence = serde_json::from_value(serde_json::json!({
            "occurrence_id": "f4f3e2d1-c0b9-48a7-9615-141312111000",
            "schedule_id": "f4f3e2d1-c0b9-48a7-9615-141312111001",
            "schedule_revision": 1,
            "entity_id": "rover",
            "planned_start_ms": 1,
            "planned_end_ms": 2,
            "dst_resolution": "exact",
            "state": "active",
            "retry_count": 0,
            "next_retry_at_ms": null,
            "group_id": null,
            "start_request_id": "f4f3e2d1-c0b9-48a7-9615-141312111002",
            "attempts": [],
            "last_error": null,
            "suppressed_by_manual": false,
            "created_at_ms": 1,
            "updated_at_ms": 2,
            "terminal_at_ms": null,
            "expires_at_ms": null
        }))
        .unwrap();
        item.observe_protected_work_snapshot(ProtectedWorkSnapshot {
            protocol_version: robo_rover_lib::PROTECTED_WORK_RELAY_PROTOCOL_VERSION,
            snapshot_id: "f4f3e2d1-c0b9-48a7-9615-141312111003".into(),
            entity_id: "rover".into(),
            generated_at_ms: 2,
            occurrences: vec![occurrence],
        });

        assert_eq!(item.protected_operations.len(), 1);
        assert!(item
            .protected_operations
            .values()
            .any(|operation| operation.active));
        assert_eq!(
            item.tick(time(300_001, 300_001)).status.requested_profile,
            PowerProfile::NormalRover
        );
    }

    #[test]
    fn protected_work_snapshot_does_not_clear_an_equal_timestamp_update() {
        let mut item = coordinator();
        item.apply_protected_operation("recording-1".into(), true, 5);
        item.observe_protected_work_snapshot(ProtectedWorkSnapshot {
            protocol_version: robo_rover_lib::PROTECTED_WORK_RELAY_PROTOCOL_VERSION,
            snapshot_id: "f4f3e2d1-c0b9-48a7-9615-141312111004".into(),
            entity_id: "rover".into(),
            generated_at_ms: 5,
            occurrences: vec![],
        });

        assert!(item.protected_operations["recording-1"].active);
    }

    #[test]
    fn orchestra_sleep_and_auto_use_the_contract_valid_idle_profile() {
        let mut item = PowerCoordinator::new(
            CoordinatorConfig::for_test(LifecycleRole::Orchestra, "orchestra"),
            1,
        )
        .unwrap();
        item.policy = PowerPolicy::Sleep;
        assert_eq!(
            item.tick(time(1, 1)).status.requested_profile,
            PowerProfile::OrchestraIdle
        );
        item.policy = PowerPolicy::Auto;
        item.effective = PowerProfile::OrchestraSpeech;
        item.observe_resources(orchestra_snapshot(1, 1));
        item.tick(time(1, 1));
        item.observe_resources(orchestra_snapshot(2, 300_001));
        item.tick(time(300_001, 300_001));
        item.observe_resources(orchestra_snapshot(3, 300_002));
        let status = item.tick(time(300_002, 300_002)).status;
        assert_eq!(status.requested_profile, PowerProfile::OrchestraIdle);
        assert!(status.validate().is_ok());
        assert_eq!(
            state_for(
                PowerPolicy::Auto,
                PowerProfile::OrchestraIdle,
                PowerProfile::OrchestraIdle
            ),
            PowerState::Active
        );
    }
    #[test]
    fn lifecycle_command_waits_for_authoritative_status_then_uses_transition_fence() {
        let mut item = coordinator();
        item.observe_lifecycle(status(
            "audio-capture",
            None,
            LifecycleDesiredState::Running,
            LifecycleEffectiveState::Running,
        ));
        let effects = item.tick(time(1, 1));
        assert_eq!(effects.lifecycle_commands.len(), 1);
        assert!(effects.lifecycle_commands[0].transition_id.is_some());
    }

    #[test]
    fn snapshot_gate_suppresses_effects_and_requires_one_fresh_authorization() {
        let mut item = coordinator();
        item.require_authority_snapshot(LifecycleRole::Rover, "rover-remote".into())
            .unwrap();
        let missing = item.tick(time(1, 1));
        assert_eq!(missing.status.state, PowerState::AuthorityUnknown);
        assert!(missing.lifecycle_commands.is_empty());

        let invalid = PowerAuthoritySnapshot {
            protocol_version: POWER_PROTOCOL_VERSION,
            snapshot_id: "f4f3e2d1-c0b9-48a7-9615-141312111010".into(),
            role: LifecycleRole::Rover,
            entity_id: "other-rover".into(),
            authority: PowerAuthority {
                epoch: 1,
                sequence: 1,
            },
            state: PowerState::Active,
            effective_profile: PowerProfile::NormalRover,
            captured_at_ms: 1,
            expires_at_ms: 100,
        };
        assert_eq!(
            item.observe_authority_snapshot(invalid, 2),
            PowerAuthorityDecision::ObserveOnly
        );
        assert_eq!(
            item.tick(time(2, 2)).status.reason_code,
            Some(PowerReasonCode::InvalidTarget)
        );

        let valid = PowerAuthoritySnapshot {
            entity_id: "rover-remote".into(),
            snapshot_id: "f4f3e2d1-c0b9-48a7-9615-141312111011".into(),
            ..PowerAuthoritySnapshot {
                protocol_version: POWER_PROTOCOL_VERSION,
                snapshot_id: "unused".into(),
                role: LifecycleRole::Rover,
                entity_id: "unused".into(),
                authority: PowerAuthority {
                    epoch: 1,
                    sequence: 1,
                },
                state: PowerState::Active,
                effective_profile: PowerProfile::NormalRover,
                captured_at_ms: 2,
                expires_at_ms: 100,
            }
        };
        assert_eq!(
            item.observe_authority_snapshot(valid, 2),
            PowerAuthorityDecision::ObserveOnly
        );
        assert_eq!(item.authority_epoch(), 2);
        assert_ne!(
            item.tick(time(2, 2)).status.state,
            PowerState::AuthorityUnknown
        );
        let proposed = PowerAuthority {
            epoch: 2,
            sequence: 1,
        };
        assert_eq!(
            item.authorize_remote_profile_command(proposed, 2),
            PowerAuthorityDecision::CommandAllowed
        );
        assert_eq!(
            item.authorize_remote_profile_command(proposed, 2),
            PowerAuthorityDecision::ObserveOnly
        );
    }

    #[test]
    fn authority_snapshot_is_scoped_and_has_a_bounded_freshness_window() {
        let item = coordinator();
        let snapshot = item.authority_snapshot(1_000);

        assert_eq!(snapshot.role, LifecycleRole::Rover);
        assert_eq!(snapshot.entity_id, "rover");
        assert_eq!(snapshot.expires_at_ms - snapshot.captured_at_ms, 30_000);
        assert!(snapshot.validate().is_ok());
    }

    #[test]
    fn command_id_replay_returns_original_result_and_rejects_changed_payload() {
        let mut item = coordinator();
        let command = PowerCommand {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id: "f4f3e2d1-c0b9-48a7-9615-141312111012".into(),
            role: LifecycleRole::Rover,
            entity_id: "rover".into(),
            authority: PowerAuthority {
                epoch: 1,
                sequence: 1,
            },
            action: PowerCommandAction::SetPolicy {
                policy: PowerPolicy::Auto,
            },
            issued_at_ms: 1,
            not_before_ms: 1,
            expires_at_ms: 100,
            detail: None,
        };
        let accepted = item.apply_command(command.clone(), time(2, 2));
        assert!(accepted.accepted);
        assert_eq!(item.apply_command(command.clone(), time(3, 3)), accepted);

        let mut changed = command;
        changed.action = PowerCommandAction::SetPolicy {
            policy: PowerPolicy::Sleep,
        };
        let rejected = item.apply_command(changed, time(3, 3));
        assert_eq!(
            rejected.reason_code,
            Some(PowerReasonCode::DuplicateMismatch)
        );
        assert_eq!(item.current_status(3).authority, accepted.authority);
    }

    #[test]
    fn rover_rejects_gapped_or_reordered_authority_but_accepts_exact_epoch_reconciliation() {
        let mut item = coordinator();
        let mut command = PowerCommand {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id: "f4f3e2d1-c0b9-48a7-9615-141312111021".into(),
            role: LifecycleRole::Rover,
            entity_id: "rover".into(),
            authority: PowerAuthority {
                epoch: 3,
                sequence: 9,
            },
            action: PowerCommandAction::SetPolicy {
                policy: PowerPolicy::Auto,
            },
            issued_at_ms: 1,
            not_before_ms: 1,
            expires_at_ms: 100,
            detail: None,
        };
        assert_eq!(
            item.apply_command(command.clone(), time(2, 2)).reason_code,
            Some(PowerReasonCode::StaleAuthority)
        );

        command.command_id = "f4f3e2d1-c0b9-48a7-9615-141312111022".into();
        command.authority = PowerAuthority {
            epoch: 2,
            sequence: 2,
        };
        assert!(!item.apply_command(command.clone(), time(2, 2)).accepted);

        command.command_id = "f4f3e2d1-c0b9-48a7-9615-141312111023".into();
        command.authority = PowerAuthority {
            epoch: 2,
            sequence: 1,
        };
        let accepted = item.apply_command(command, time(2, 2));
        assert!(accepted.accepted);
        assert_eq!(
            accepted.authority,
            PowerAuthority {
                epoch: 2,
                sequence: 2
            }
        );
    }

    #[test]
    fn quiesce_transition_clears_replay_cache_and_fences_old_commands() {
        let mut item = coordinator();
        item.effective = PowerProfile::NormalRover;
        item.requested = PowerProfile::NormalRover;
        let command = PowerCommand {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id: "f4f3e2d1-c0b9-48a7-9615-141312111020".into(),
            role: LifecycleRole::Rover,
            entity_id: "rover".into(),
            authority: item.authority,
            action: PowerCommandAction::SetPolicy {
                policy: PowerPolicy::Sleep,
            },
            issued_at_ms: 1,
            not_before_ms: 1,
            expires_at_ms: 2,
            detail: None,
        };

        assert!(item.apply_command(command.clone(), time(1, 1)).accepted);
        assert_eq!(item.command_replays.len(), 1);
        item.tick(time(1, 1));
        assert!(item.command_replays.is_empty());
        assert_eq!(
            item.apply_command(command, time(1, 1)).reason_code,
            Some(PowerReasonCode::StaleAuthority)
        );
    }

    #[test]
    fn expired_command_replay_is_pruned_before_new_admission() {
        let mut item = coordinator();
        let command = PowerCommand {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id: "f4f3e2d1-c0b9-48a7-9615-141312111013".into(),
            role: LifecycleRole::Rover,
            entity_id: "rover".into(),
            authority: PowerAuthority {
                epoch: 1,
                sequence: 1,
            },
            action: PowerCommandAction::SetPolicy {
                policy: PowerPolicy::Auto,
            },
            issued_at_ms: 1,
            not_before_ms: 1,
            expires_at_ms: 3,
            detail: None,
        };
        assert!(item.apply_command(command.clone(), time(2, 2)).accepted);
        assert!(item.replay_result(&command, 3).is_none());
        assert!(item.command_replays.is_empty());
        let replay = item.apply_command(command, time(3, 3));
        assert_eq!(replay.reason_code, Some(PowerReasonCode::StaleAuthority));
        assert_eq!(item.command_replays.len(), 1);
    }

    #[test]
    fn partial_barrier_never_reports_ready() {
        let mut item = coordinator();
        item.observe_lifecycle(status(
            "audio-capture",
            None,
            LifecycleDesiredState::Running,
            LifecycleEffectiveState::Running,
        ));
        let command = item.tick(time(1, 1)).lifecycle_commands.remove(0);
        item.observe_lifecycle(status(
            "audio-capture",
            command.transition_id,
            LifecycleDesiredState::Running,
            LifecycleEffectiveState::Running,
        ));
        let effects = item.tick(time(2, 2));
        assert!(effects.lifecycle_commands.is_empty());
        assert_ne!(effects.status.state, PowerState::Active);
    }

    #[test]
    fn lifecycle_status_never_regresses_its_epoch_or_revision() {
        let mut item = coordinator();
        let mut current = status(
            "audio-capture",
            None,
            LifecycleDesiredState::Running,
            LifecycleEffectiveState::Running,
        );
        current.revision = 2;
        current.updated_at_ms = 10;
        item.observe_lifecycle(current.clone());

        let mut stale = current;
        stale.revision = 1;
        stale.updated_at_ms = 20;
        item.observe_lifecycle(stale);

        assert_eq!(item.lifecycle["audio-capture"].revision, 2);
    }

    #[test]
    fn late_quiesce_acceptance_reissues_the_superseding_wake_without_timeout() {
        let mut item = coordinator();
        for node in [
            "audio-capture",
            "edge-voice",
            "gst-camera",
            "audio-playback",
        ] {
            item.observe_lifecycle(status(
                node,
                None,
                LifecycleDesiredState::Running,
                LifecycleEffectiveState::Running,
            ));
        }
        item.effective = PowerProfile::NormalRover;
        item.policy = PowerPolicy::Sleep;

        let first = single_command(&mut item, 1);
        assert_eq!(first.target.node_id, "audio-playback");
        item.observe_lifecycle(status(
            "audio-playback",
            first.transition_id,
            LifecycleDesiredState::Quiesced,
            LifecycleEffectiveState::Quiesced,
        ));
        let second = single_command(&mut item, 2);
        assert_eq!(second.target.node_id, "gst-camera");
        item.observe_lifecycle(status(
            "gst-camera",
            second.transition_id,
            LifecycleDesiredState::Quiesced,
            LifecycleEffectiveState::Quiesced,
        ));
        let third = single_command(&mut item, 3);
        assert_eq!(third.target.node_id, "edge-voice");
        item.observe_lifecycle(status(
            "edge-voice",
            third.transition_id,
            LifecycleDesiredState::Quiesced,
            LifecycleEffectiveState::Quiesced,
        ));
        let quiesce = single_command(&mut item, 4);
        assert_eq!(quiesce.target.node_id, "audio-capture");

        item.policy = PowerPolicy::Auto;
        item.ledger.apply(demand(), 5).unwrap();
        let mut replacement_effects = item.tick(time(5, 5));
        assert_eq!(
            replacement_effects.status.effective_profile,
            PowerProfile::NormalRover
        );
        assert_eq!(replacement_effects.status.state, PowerState::Waking);
        let replacement = replacement_effects.lifecycle_commands.remove(0);
        assert_eq!(replacement.target.node_id, "audio-capture");
        assert_eq!(replacement.desired_state, LifecycleDesiredState::Running);
        assert_eq!(replacement.expected_revision, 0);

        let mismatched = LifecycleCommandResult {
            protocol_version: 1,
            request_id: quiesce.request_id.clone(),
            accepted: true,
            manager_epoch: 2,
            revision: 2,
            reason_code: None,
            detail: None,
        };
        item.observe_lifecycle_result(mismatched);
        assert!(item.command_targets.contains_key(&quiesce.request_id));

        item.observe_lifecycle_result(LifecycleCommandResult {
            protocol_version: 1,
            request_id: quiesce.request_id,
            accepted: true,
            manager_epoch: 2,
            revision: 1,
            reason_code: None,
            detail: None,
        });
        assert!(!item.pending.as_ref().unwrap().issued);
        let retry = item.tick(time(6, 6)).lifecycle_commands.remove(0);
        assert_eq!(retry.target.node_id, "audio-capture");
        assert_eq!(retry.desired_state, LifecycleDesiredState::Running);
        assert_eq!(retry.expected_revision, 1);
    }

    #[test]
    fn fresh_restart_is_awake_and_waits_for_lifecycle_authority() {
        let mut item = coordinator();
        let effects = item.tick(time(1, 1));
        assert_eq!(effects.status.policy, PowerPolicy::Awake);
        assert_eq!(effects.status.requested_profile, PowerProfile::NormalRover);
        assert_eq!(effects.status.state, PowerState::Waking);
        assert!(effects.lifecycle_commands.is_empty());
    }
}
