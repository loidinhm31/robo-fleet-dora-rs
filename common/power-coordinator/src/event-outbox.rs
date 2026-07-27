use crate::{
    outbox_event::{command_context, lifecycle_context, record, rejected},
    CoordinatorConfig, CoordinatorEffects, CoordinatorTime, EventJournal, JournalAppendClass,
    JournalConfig, JournalIntent, JournalRecord, PowerCoordinator,
};
use robo_rover_lib::{
    PowerCommand, PowerCommandAction, PowerCommandResult, PowerEventType, PowerStatus,
};

/// Returns external lifecycle effects only after their journal intent syncs.
pub struct DurablePowerCoordinator {
    coordinator: PowerCoordinator,
    journal: EventJournal,
    last_effective: Option<robo_rover_lib::PowerProfile>,
}

impl DurablePowerCoordinator {
    pub fn open(config: CoordinatorConfig, now_ms: u64) -> Result<Self, String> {
        let journal = EventJournal::open(JournalConfig {
            directory: config.journal_dir.clone().into(),
            max_bytes: config.journal_max_bytes,
            max_records: config.journal_max_records,
            wake_reserve_bytes: config.journal_wake_reserve_bytes,
            wake_reserve_records: config.journal_wake_reserve_records,
        })?;
        let mut item = Self {
            coordinator: PowerCoordinator::new(config, journal.next_epoch())?,
            journal,
            last_effective: None,
        };
        let status = item.coordinator.current_status(now_ms);
        item.journal.replace_boot_intent(record(
            JournalIntent::BootAwake,
            PowerEventType::PolicyChanged,
            None,
            status.clone(),
            now_ms,
            Default::default(),
        ))?;
        item.last_effective = Some(status.effective_profile);
        Ok(item)
    }

    pub fn apply_command(
        &mut self,
        command: PowerCommand,
        now: CoordinatorTime,
    ) -> PowerCommandResult {
        if let Some(result) = self.coordinator.replay_result(&command, now.wall_ms) {
            return result;
        }
        if self.coordinator.command_will_apply(&command, now).is_err() {
            return self.coordinator.apply_command(command, now);
        }
        let event_type = match command.action {
            PowerCommandAction::SetPolicy { .. } => PowerEventType::PolicyChanged,
            _ => PowerEventType::DemandChanged,
        };
        let class = command_append_class(&command);
        let mut predicted = self.coordinator.clone();
        let predicted_result = predicted.apply_command(command.clone(), now);
        if !predicted_result.accepted {
            return predicted_result;
        }
        if let Err(detail) = self.append_command_pair(
            &command,
            event_type,
            predicted.current_status(now.wall_ms),
            now.wall_ms,
            class,
        ) {
            return rejected(
                command,
                self.coordinator.current_status(now.wall_ms),
                detail,
            );
        }
        self.coordinator.apply_command(command, now)
    }

    pub fn observe_resources(&mut self, snapshot: robo_rover_lib::ResourceSnapshot) {
        self.coordinator.observe_resources(snapshot);
    }
    pub fn observe_lifecycle(&mut self, status: robo_rover_lib::LifecycleStatus) {
        self.coordinator.observe_lifecycle(status);
    }
    pub fn observe_lifecycle_result(&mut self, result: robo_rover_lib::LifecycleCommandResult) {
        self.coordinator.observe_lifecycle_result(result);
    }
    pub fn observe_protected_operation(&mut self, operation_id: impl Into<String>, active: bool) {
        self.coordinator
            .observe_protected_operation(operation_id, active);
    }
    pub fn observe_protected_occurrence(
        &mut self,
        occurrence: robo_rover_lib::RecordingOccurrence,
    ) {
        self.coordinator.observe_protected_occurrence(occurrence);
    }
    pub fn observe_protected_work_snapshot(
        &mut self,
        snapshot: robo_rover_lib::ProtectedWorkSnapshot,
    ) {
        self.coordinator.observe_protected_work_snapshot(snapshot);
    }
    pub fn acknowledge(&mut self, event_id: &str) -> Result<(), String> {
        self.journal.acknowledge(event_id)
    }
    pub fn compact(&mut self) -> Result<(), String> {
        self.journal.compact()
    }
    pub fn pending_records(&self) -> Vec<JournalRecord> {
        self.journal.pending().cloned().collect()
    }
    pub fn journal_health(&self) -> crate::JournalHealth {
        self.journal.health()
    }

    pub fn tick(&mut self, now: CoordinatorTime) -> Result<CoordinatorEffects, String> {
        self.coordinator
            .set_journal_capacity_unsafe(self.journal.capacity().unsafe_for_sleep);
        let effects = self.coordinator.tick(now);
        // A transition may be announced before lifecycle statuses are present.
        // Persist another requested record when commands are actually issued so
        // the durable history retains the exact target/revision fence.
        if effects.transition.is_some() || !effects.lifecycle_commands.is_empty() {
            let class = is_wake_to_safer(
                effects.status.effective_profile,
                effects.status.requested_profile,
            )
            .then_some(JournalAppendClass::WakeToSafer)
            .unwrap_or(JournalAppendClass::Normal);
            self.append_with_class(
                JournalIntent::Transition,
                PowerEventType::TransitionRequested,
                effects.status.transition_id.clone(),
                effects.status.clone(),
                now.wall_ms,
                class,
                lifecycle_context(&effects.lifecycle_commands),
            )?;
        }
        if self.last_effective != Some(effects.status.effective_profile) {
            let prior = self
                .last_effective
                .unwrap_or(effects.status.effective_profile);
            let class = is_wake_to_safer(prior, effects.status.effective_profile)
                .then_some(JournalAppendClass::WakeToSafer)
                .unwrap_or(JournalAppendClass::Normal);
            self.append_with_class(
                JournalIntent::TransitionApplied,
                PowerEventType::TransitionApplied,
                effects.status.transition_id.clone(),
                effects.status.clone(),
                now.wall_ms,
                class,
                Default::default(),
            )?;
            self.last_effective = Some(effects.status.effective_profile);
        }
        self.coordinator
            .set_journal_capacity_unsafe(self.journal.capacity().unsafe_for_sleep);
        Ok(effects)
    }

    fn append_command_pair(
        &mut self,
        command: &PowerCommand,
        event_type: PowerEventType,
        applied_status: PowerStatus,
        now_ms: u64,
        class: JournalAppendClass,
    ) -> Result<(), String> {
        let mut intent = record(
            JournalIntent::Command,
            event_type,
            None,
            applied_status.clone(),
            now_ms,
            command_context(command),
        );
        // An intent proves admission ordering but must never publish the
        // pre-apply snapshot as authoritative current state.
        intent.status = None;
        let applied = record(
            JournalIntent::CommandApplied,
            PowerEventType::CommandApplied,
            None,
            applied_status,
            now_ms,
            command_context(command),
        );
        self.journal
            .preflight_append(&[(&intent, class), (&applied, class)])?;
        self.journal.append(intent, class)?;
        self.journal.append(applied, class)
    }
    fn append_with_class(
        &mut self,
        intent: JournalIntent,
        event_type: PowerEventType,
        transition_id: Option<String>,
        status: PowerStatus,
        now_ms: u64,
        class: JournalAppendClass,
        context: robo_rover_lib::PowerEventContext,
    ) -> Result<(), String> {
        self.journal.append(
            record(intent, event_type, transition_id, status, now_ms, context),
            class,
        )
    }
}

fn command_append_class(command: &PowerCommand) -> JournalAppendClass {
    match command.action {
        PowerCommandAction::SetPolicy {
            policy: robo_rover_lib::PowerPolicy::Awake,
        }
        | PowerCommandAction::RegisterDemand { .. }
        | PowerCommandAction::RegisterReservation { .. } => JournalAppendClass::WakeToSafer,
        _ => JournalAppendClass::Normal,
    }
}

fn is_wake_to_safer(from: robo_rover_lib::PowerProfile, to: robo_rover_lib::PowerProfile) -> bool {
    matches!(
        to,
        robo_rover_lib::PowerProfile::NormalRover
            | robo_rover_lib::PowerProfile::OrchestraSpeech
            | robo_rover_lib::PowerProfile::ScheduledCapture
    ) && from != to
}
