use crate::readiness::fresh_low_cpu;
use crate::{
    plan_transition, target, CoordinatorConfig, DemandLedger, ProfileCatalog, TransitionPlan,
};
use robo_rover_lib::{
    LifecycleCommand, LifecycleCommandOrigin, LifecycleCommandResult, LifecycleEffectiveState,
    LifecycleStatus, PowerAuthority, PowerCommand, PowerCommandAction, PowerCommandResult,
    PowerPolicy, PowerProfile, PowerReasonCode, PowerState, PowerStatus, PowerTransition,
    ResourceSnapshot, POWER_PROTOCOL_VERSION,
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
pub struct CoordinatorEffects {
    pub status: PowerStatus,
    pub transition: Option<PowerTransition>,
    pub lifecycle_commands: Vec<LifecycleCommand>,
    pub query_lifecycle_status: bool,
}

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
    idle_since_ms: Option<u64>,
    last_resource_sequence: Option<u64>,
    low_samples: u32,
    awake_since_ms: u64,
    protected_operation: bool,
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
            idle_since_ms: None,
            last_resource_sequence: None,
            low_samples: 0,
            awake_since_ms: 0,
            protected_operation: false,
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
        self.protected_operation = active;
    }

    pub fn set_journal_capacity_unsafe(&mut self, unsafe_capacity: bool) {
        self.journal_capacity_unsafe = unsafe_capacity;
    }

    pub fn current_status(&self, now_ms: u64) -> PowerStatus {
        self.status(now_ms)
    }

    pub fn next_authority(&self) -> PowerAuthority {
        PowerAuthority {
            epoch: self.authority.epoch,
            sequence: self.authority.sequence.saturating_add(1),
        }
    }

    pub fn command_will_apply(
        &self,
        command: &PowerCommand,
        now: CoordinatorTime,
    ) -> Result<(), String> {
        command.validates_for(self.config.role, &self.config.entity_id)?;
        if command.authority != self.authority {
            return Err("power command authority is stale".into());
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
                .release_reservation(reservation_id)
                .map(|_| ())
                .map_err(|reason| format!("{reason:?}")),
        }
    }

    pub fn apply_command(
        &mut self,
        command: PowerCommand,
        now: CoordinatorTime,
    ) -> PowerCommandResult {
        let accepted = command
            .validates_for(self.config.role, &self.config.entity_id)
            .and_then(|_| {
                if command.authority != self.authority {
                    return Err("power command authority is stale".into());
                }
                if command.expires_at_ms <= now.wall_ms {
                    return Err("power command expired".into());
                }
                if command.not_before_ms > now.wall_ms {
                    return Err("power command is not active".into());
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
                        .release_reservation(reservation_id)
                        .map(|_| ())
                        .map_err(|reason| format!("{reason:?}")),
                }
            });
        match accepted {
            Ok(()) => PowerCommandResult {
                protocol_version: POWER_PROTOCOL_VERSION,
                command_id: command.command_id,
                accepted: true,
                authority: self.bump_authority(),
                reason_code: None,
                detail: None,
            },
            Err(detail) => PowerCommandResult {
                protocol_version: POWER_PROTOCOL_VERSION,
                command_id: command.command_id,
                accepted: false,
                authority: self.authority,
                reason_code: Some(reason_from_error(&detail)),
                detail: Some(detail),
            },
        }
    }

    pub fn tick(&mut self, now: CoordinatorTime) -> CoordinatorEffects {
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
                PowerProfile::Dormant
            }
            PowerPolicy::Auto
                if self.protected_operation
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
            PowerProfile::IdleListening
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
            transition_id: self.pending.as_ref().map(|item| item.id.clone()),
            reason_code: (self.state == PowerState::Failed).then_some(PowerReasonCode::Timeout),
            detail: None,
            updated_at_ms: now_ms,
        }
    }

    fn bump_authority(&mut self) -> PowerAuthority {
        self.authority.sequence = self.authority.sequence.saturating_add(1);
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
        } else if requested == PowerProfile::Dormant || requested == PowerProfile::IdleListening {
            PowerState::Quiescing
        } else {
            PowerState::Waking
        }
    } else {
        match effective {
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
