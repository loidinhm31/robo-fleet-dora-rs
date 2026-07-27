use std::time::Duration;

use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event, MetadataParameters,
};
use eyre::Result;
use robo_rover_lib::{
    occurrence_requires_protection, scheduled_intent_id, AuthenticatedRecordingScheduleCommand,
    LifecycleRole, PowerCommand, PowerCommandAction, PowerCommandResult, PowerProfile,
    PowerReservation, PowerStatus, ProtectedWorkSnapshot, ProtectedWorkSnapshotRequest,
    RecordingCoordinatorFeedback, RecordingOccurrence, RecordingReconciliationRequest,
    RecordingReconciliationSnapshot, RecordingScheduleQuery, RecordingScheduleSnapshot,
    RecordingSchedulerReadiness, RecordingSchedulerStatus, ScheduledRecordingIntentAction,
    POWER_PROTOCOL_VERSION, RECORDING_SCHEDULE_PROTOCOL_VERSION,
};
use std::collections::BTreeMap;
use uuid::Uuid;

use crate::{
    clock::{Clock, SystemClock},
    config::SchedulerConfig,
    mongo_repository::MongoRepository,
    mongo_repository::OutboxRecord,
    node_intents::build_intent,
    node_persistence::{adopt, persist},
    reservation_command_outbox::{ReservationCommandAction, ReservationCommandOutboxRecord},
    runtime::SchedulerRuntime,
    service::ScheduleService,
};

/// Tracks the latest recorder reconciliation exchange. Dora output is an
/// ephemeral channel, so each successful periodic exchange republishes Ready;
/// a restarted web bridge must not depend on receiving the first Ready event.
#[derive(Debug)]
struct ReconciliationState {
    request_id: String,
    deadline_ms: i64,
    interval_ms: i64,
    awaiting_snapshot: bool,
    degraded: bool,
}

impl ReconciliationState {
    fn new(now_ms: i64, interval_ms: i64) -> Self {
        let mut state = Self {
            request_id: String::new(),
            deadline_ms: now_ms,
            interval_ms,
            awaiting_snapshot: false,
            degraded: false,
        };
        state.next_request(now_ms);
        state
    }

    fn next_request(&mut self, now_ms: i64) -> RecordingReconciliationRequest {
        self.request_id = uuid::Uuid::new_v4().to_string();
        self.deadline_ms = now_ms + self.interval_ms;
        self.awaiting_snapshot = true;
        RecordingReconciliationRequest {
            request_id: self.request_id.clone(),
            entity_id: None,
        }
    }

    fn accepts(&self, request_id: &str) -> bool {
        self.request_id == request_id
    }

    fn mark_degraded_if_overdue(&mut self, now_ms: i64) -> bool {
        if self.awaiting_snapshot && now_ms >= self.deadline_ms && !self.degraded {
            self.degraded = true;
            true
        } else {
            false
        }
    }

    fn mark_ready(&mut self) -> bool {
        let recovered = self.degraded;
        self.degraded = false;
        self.awaiting_snapshot = false;
        recovered
    }
}

pub fn run() -> Result<()> {
    let config = SchedulerConfig::from_env().map_err(eyre::Report::msg)?;
    let tokio = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let status = DataId::from("recording_scheduler_status".to_owned());
    publish_status(
        &mut node,
        &status,
        RecordingSchedulerReadiness::Initializing,
        "scheduler initialization pending",
    )?;

    // A Mongo/index/reconciliation outage must not take manual recording down
    // with the scheduler process. Dora keeps this node alive and buffers its
    // bounded input queues until initialization succeeds again.
    loop {
        match run_ready_session(&config, &tokio, &mut node, &mut events, &status) {
            Ok(()) => return Ok(()),
            Err(_) => {
                publish_status(
                    &mut node,
                    &status,
                    RecordingSchedulerReadiness::Degraded,
                    "Mongo, indexes, or reconciliation unavailable",
                )?;
                tracing::warn!(
                    scheduler_ready = false,
                    "recording scheduler degraded; retrying Mongo initialization"
                );
                std::thread::sleep(Duration::from_secs(5));
            }
        }
    }
}

fn run_ready_session(
    config: &SchedulerConfig,
    tokio: &tokio::runtime::Runtime,
    mut node: &mut DoraNode,
    events: &mut dora_node_api::EventStream,
    status: &DataId,
) -> Result<()> {
    let repository = tokio.block_on(async {
        let repository =
            MongoRepository::connect(&config.mongodb_uri, &config.mongodb_database).await?;
        repository.ensure_indexes().await?;
        Ok::<_, mongodb::error::Error>(repository)
    })?;
    let mut scheduler = load_scheduler(&tokio, &repository)?;
    let service = ScheduleService::new(SystemClock, config.clone(), repository.clone());
    let result = DataId::from("recording_schedule_command_result".to_owned());
    let snapshot_result = DataId::from("recording_schedule_snapshot".to_owned());
    let occurrence_status = DataId::from("recording_occurrence_status".to_owned());
    let protected_work_snapshot = DataId::from("protected_work_snapshot".to_owned());
    let intent = DataId::from("scheduled_recording_intent".to_owned());
    let power_command = DataId::from("scheduled_power_command".to_owned());
    let prewarm_metrics = DataId::from("recording_prewarm_metrics".to_owned());
    let reconcile = DataId::from("recording_reconciliation_request".to_owned());
    let manual_suppression_ack =
        DataId::from("recording_scheduler_manual_suppression_ack".to_owned());
    let reconcile_interval_ms = config.reconcile_seconds as i64 * 1_000;
    let now_ms = SystemClock.now_ms();
    let mut reconciliation = ReconciliationState::new(now_ms, reconcile_interval_ms);
    send(
        &mut node,
        &reconcile,
        &RecordingReconciliationRequest {
            request_id: reconciliation.request_id.clone(),
            entity_id: None,
        },
    )?;
    let mut next_refresh_ms = 0;
    let mut next_snapshot_ms = now_ms + reconcile_interval_ms;
    let mut initial_reconciliation_complete = false;
    let mut rover_power_statuses = BTreeMap::<String, PowerStatus>::new();
    replay_reservation_commands(tokio, &repository, node, &power_command, &mut scheduler)?;
    tracing::info!(
        scheduler_ready = false,
        horizon_days = config.horizon_days,
        "recording scheduler awaiting reconciliation"
    );
    loop {
        let event = events.recv_timeout(Duration::from_millis(250));
        if matches!(event, Some(Event::Stop(_))) {
            break;
        }
        if let Some(Event::Input { id, data, .. }) = event {
            match id.as_str() {
                "recording_schedule_command" => {
                    if let Some(value) = decode::<AuthenticatedRecordingScheduleCommand>(&*data) {
                        let response = tokio.block_on(service.execute(value));
                        tracing::info!(
                            event = "recording_scheduler_command",
                            request_id = %response.request_id,
                            accepted = response.accepted,
                            reason_code = ?response.reason_code,
                            "recording schedule command processed"
                        );
                        if response.reason_code
                            == Some(robo_rover_lib::RecordingScheduleReasonCode::Conflict)
                        {
                            tracing::warn!(
                                event = "recording_scheduler_conflict",
                                request_id = %response.request_id,
                                "recording schedule command rejected by revision conflict"
                            );
                        }
                        if response.accepted {
                            scheduler = load_scheduler(&tokio, &repository)?;
                            if initial_reconciliation_complete {
                                scheduler.complete_reconciliation();
                            }
                        }
                        send(&mut node, &result, &response)?;
                    }
                }
                "recording_schedule_query" => {
                    if let Some(query) = decode::<RecordingScheduleQuery>(&*data) {
                        if let Err(error) = query.validate() {
                            tracing::warn!(%error, "rejected recording schedule query");
                            continue;
                        }
                        let schedules = tokio
                            .block_on(repository.load_schedules())
                            .map_err(eyre::Report::msg)?
                            .into_iter()
                            .filter(|schedule| schedule.definition.entity_id == query.entity_id)
                            .collect();
                        send(
                            &mut node,
                            &snapshot_result,
                            &RecordingScheduleSnapshot {
                                protocol_version: RECORDING_SCHEDULE_PROTOCOL_VERSION,
                                request_id: query.request_id,
                                entity_id: query.entity_id.clone(),
                                schedules,
                            },
                        )?;
                        // The snapshot owns definitions; occurrence updates carry scheduler-owned
                        // next-run and lifecycle state without letting the browser infer recurrence.
                        for occurrence in scheduler
                            .occurrences
                            .values()
                            .filter(|value| value.entity_id == query.entity_id)
                        {
                            send(&mut node, &occurrence_status, occurrence)?;
                        }
                    }
                }
                "recording_scheduler_recorder_feedback" => {
                    if let Some(value) = decode::<RecordingCoordinatorFeedback>(&*data) {
                        tracing::info!(
                            event = "recording_scheduler_feedback",
                            accepted = value.accepted,
                            applied = value.applied,
                            retryable = value.retryable,
                            "recording coordinator feedback received"
                        );
                        let applied = scheduler.apply_feedback(value.clone());
                        if applied {
                            persist(&tokio, &repository, &mut scheduler)?;
                            tokio
                                .block_on(repository.acknowledge_intent(&value.intent_id))
                                .map_err(eyre::Report::msg)?;
                            if value.manual_suppression {
                                send(&mut node, &manual_suppression_ack, &value)?;
                            }
                            if let Some(occurrence) =
                                scheduler.occurrences.get(&value.occurrence_id)
                            {
                                send(&mut node, &occurrence_status, occurrence)?;
                            }
                            if value.manual_suppression && value.group_id.is_some() {
                                let group_id = value.group_id.as_deref();
                                for occurrence in scheduler
                                    .occurrences
                                    .values()
                                    .filter(|occurrence| occurrence.group_id.as_deref() == group_id)
                                {
                                    send(&mut node, &occurrence_status, occurrence)?;
                                }
                            }
                            if value.retryable {
                                tracing::warn!(
                                    event = "recording_scheduler_retry",
                                    intent_id = %value.intent_id,
                                    occurrence_id = %value.occurrence_id,
                                    group_id = ?value.group_id,
                                    reason_code = ?value.reason_code,
                                    retry_at_ms = ?scheduler.occurrences
                                        .get(&value.occurrence_id)
                                        .and_then(|occurrence| occurrence.next_retry_at_ms),
                                    "scheduled recording transition queued for retry"
                                );
                            } else if !value.accepted || !value.applied {
                                tracing::warn!(
                                    event = "recording_scheduler_transition_failed",
                                    intent_id = %value.intent_id,
                                    occurrence_id = %value.occurrence_id,
                                    group_id = ?value.group_id,
                                    reason_code = ?value.reason_code,
                                    "scheduled recording transition failed"
                                );
                            }
                            emit_power_releases(
                                tokio,
                                &repository,
                                node,
                                &power_command,
                                &rover_power_statuses,
                                &mut scheduler,
                            )?;
                        } else if scheduler.has_handled_intent(&value.intent_id) {
                            tokio
                                .block_on(repository.acknowledge_intent(&value.intent_id))
                                .map_err(eyre::Report::msg)?;
                            if value.manual_suppression {
                                // Re-echo an idempotent manual suppression after a
                                // bridge restart so it can finish the exact Stop.
                                send(&mut node, &manual_suppression_ack, &value)?;
                            }
                        } else {
                            tracing::debug!(
                                event = "recording_scheduler_feedback_ignored",
                                intent_id = %value.intent_id,
                                occurrence_id = %value.occurrence_id,
                                group_id = ?value.group_id,
                                "ignored stale or invalid recording coordinator feedback"
                            );
                        }
                    }
                }
                "recording_reconciliation_snapshot" => {
                    if let Some(value) = decode::<RecordingReconciliationSnapshot>(&*data) {
                        if !reconciliation.accepts(&value.request_id) {
                            continue;
                        }
                        adopt(&mut scheduler, value);
                        let first_snapshot = !initial_reconciliation_complete;
                        let recovered = reconciliation.mark_ready();
                        if first_snapshot {
                            scheduler.complete_reconciliation();
                            initial_reconciliation_complete = true;
                        }
                        // Status is intentionally replayed after every successful
                        // reconciliation because the web bridge can restart after
                        // the first ephemeral Dora output.
                        publish_status(
                            node,
                            status,
                            RecordingSchedulerReadiness::Ready,
                            if recovered {
                                "scheduler reconciliation recovered"
                            } else {
                                "scheduler reconciliation complete"
                            },
                        )?;
                        persist(&tokio, &repository, &mut scheduler)?;
                        if first_snapshot {
                            replay_pending(
                                &tokio,
                                &repository,
                                &mut node,
                                &intent,
                                &mut scheduler,
                            )?;
                        }
                        // The bridge is intentionally stateless across restart. Re-send
                        // every durable live group only after its recorder snapshot barrier
                        // has crossed the bridge, so matching sessions are adopted instead
                        // of blindly started a second time.
                        replay_desired_groups(&mut node, &intent, &scheduler)?;
                        replay_protected_occurrences(&mut node, &occurrence_status, &scheduler)?;
                        replay_protected_work_snapshots(
                            &mut node,
                            &protected_work_snapshot,
                            &scheduler,
                            SystemClock.now_ms() as u64,
                        )?;
                        replay_manual_suppression_acks(
                            &mut node,
                            &manual_suppression_ack,
                            &scheduler,
                        )?;
                        log_metrics(&scheduler, "reconciliation");
                    }
                }
                "protected_work_snapshot_request" => {
                    if let Some(request) = decode::<ProtectedWorkSnapshotRequest>(&*data) {
                        if request.validate().is_err() {
                            continue;
                        }
                        send(
                            &mut node,
                            &protected_work_snapshot,
                            &protected_work_snapshot_for(
                                &scheduler,
                                request.entity_id,
                                SystemClock.now_ms() as u64,
                            ),
                        )?;
                    }
                }
                "rover_power_status" => {
                    if let Some(status) = decode::<PowerStatus>(&*data) {
                        if status
                            .validates_for(LifecycleRole::Rover, &status.entity_id)
                            .is_ok()
                        {
                            rover_power_statuses.insert(status.entity_id.clone(), status.clone());
                            let changed = scheduler.observe_power_status(&status);
                            let state_changed = !changed.is_empty();
                            for group_id in changed {
                                if let Some(metrics) = scheduler.prewarm_metrics(&group_id) {
                                    send(&mut node, &prewarm_metrics, &metrics)?;
                                }
                            }
                            if state_changed {
                                persist(&tokio, &repository, &mut scheduler)?;
                            }
                        }
                    }
                }
                "rover_power_command_result" => {
                    if let Some(result) = decode::<PowerCommandResult>(&*data) {
                        if result.validate().is_err() {
                            tracing::warn!(command_id = %result.command_id, "rejected invalid rover power command result");
                            continue;
                        }
                        if let Some(command_id) = scheduler.apply_power_command_result(&result) {
                            scheduler.prune_released_reservation_tombstones();
                            persist(&tokio, &repository, &mut scheduler)?;
                            tokio
                                .block_on(repository.acknowledge_reservation_command(&command_id))
                                .map_err(eyre::Report::msg)?;
                        }
                    }
                }
                _ => {}
            }
        }
        let now_ms = SystemClock.now_ms();
        if reconciliation.mark_degraded_if_overdue(now_ms) {
            publish_status(
                node,
                status,
                RecordingSchedulerReadiness::Degraded,
                "scheduler reconciliation snapshot deadline elapsed",
            )?;
            tracing::warn!(
                event = "recording_scheduler_reconciliation_timeout",
                request_id = %reconciliation.request_id,
                deadline_ms = reconciliation.deadline_ms,
                "recording scheduler degraded while awaiting reconciliation snapshot"
            );
        }
        if now_ms >= next_refresh_ms {
            refresh(
                &tokio,
                &repository,
                &mut scheduler,
                now_ms + config.horizon_days * 86_400_000,
            )?;
            log_metrics(&scheduler, "refresh");
            next_refresh_ms = now_ms + config.reconcile_seconds as i64 * 1_000;
        }
        if !scheduler
            .prepare_future_reservations(now_ms.saturating_add(7 * 24 * 60 * 60 * 1_000))
            .is_empty()
        {
            persist(&tokio, &repository, &mut scheduler)?;
        }
        emit_power_reservations(
            tokio,
            &repository,
            node,
            &power_command,
            &rover_power_statuses,
            &mut scheduler,
        )?;
        if now_ms >= next_snapshot_ms {
            let request = reconciliation.next_request(now_ms);
            send(&mut node, &reconcile, &request)?;
            next_snapshot_ms = now_ms + reconcile_interval_ms;
        }
        emit_due(
            tokio,
            &repository,
            node,
            &intent,
            &occurrence_status,
            &mut scheduler,
        )?;
        emit_power_releases(
            tokio,
            &repository,
            node,
            &power_command,
            &rover_power_statuses,
            &mut scheduler,
        )?;
        emit_stops(
            tokio,
            &repository,
            node,
            &intent,
            &occurrence_status,
            &mut scheduler,
        )?;
    }
    log_metrics(&scheduler, "shutdown");
    Ok(())
}

fn load_scheduler(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
) -> Result<SchedulerRuntime<SystemClock>> {
    tokio
        .block_on(repository.recover_superseded_future(SystemClock.now_ms()))
        .map_err(eyre::Report::msg)?;
    let occurrences = tokio
        .block_on(repository.load_nonterminal_stored())
        .map_err(eyre::Report::msg)?;
    let groups = tokio
        .block_on(repository.load_groups())
        .map_err(eyre::Report::msg)?;
    let mut scheduler = SchedulerRuntime::from_persisted(SystemClock, occurrences, groups)
        .map_err(eyre::Report::msg)?;
    scheduler.restore_prewarm_estimators(
        tokio
            .block_on(repository.load_prewarm_estimators())
            .map_err(eyre::Report::msg)?,
    );
    let schedules = tokio
        .block_on(repository.load_schedules())
        .map_err(eyre::Report::msg)?;
    scheduler.hydrate_group_directories(&schedules);
    Ok(scheduler)
}

fn refresh(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    scheduler: &mut SchedulerRuntime<SystemClock>,
    through_ms: i64,
) -> Result<()> {
    for schedule in tokio
        .block_on(repository.load_schedules())
        .map_err(eyre::Report::msg)?
    {
        if schedule.definition.enabled {
            scheduler
                .materialize(&schedule, through_ms)
                .map_err(eyre::Report::msg)?;
        }
    }
    persist(tokio, repository, scheduler)
}

fn emit_due(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    node: &mut DoraNode,
    output: &DataId,
    occurrence_output: &DataId,
    scheduler: &mut SchedulerRuntime<SystemClock>,
) -> Result<()> {
    for occurrence_id in scheduler.due() {
        if !scheduler.reservation_ready_for(&occurrence_id) {
            continue;
        }
        let Some(current_occurrence) = scheduler.occurrences.get(&occurrence_id) else {
            continue;
        };
        if !tokio
            .block_on(
                repository.validates_recorder_admission(current_occurrence, SystemClock.now_ms()),
            )
            .map_err(eyre::Report::msg)?
        {
            tracing::warn!(occurrence_id = %occurrence_id, "rejected stale scheduler admission before recorder acquire");
            continue;
        }
        let Some(transition) = scheduler.begin_start(&occurrence_id) else {
            continue;
        };
        let Some(intent_id) = transition.intent_id else {
            persist(tokio, repository, scheduler)?;
            continue;
        };
        let occurrence = &scheduler.occurrences[&occurrence_id];
        let group = &scheduler.groups[occurrence.group_id.as_ref().unwrap()];
        let group_id = group.group_id.clone();
        let value = build_intent(
            occurrence,
            group,
            intent_id,
            ScheduledRecordingIntentAction::Acquire,
        );
        let occurrence_status = occurrence.clone();
        tokio
            .block_on(repository.persist_transition(&OutboxRecord {
                intent: value.clone(),
                occurrence: occurrence.clone(),
                group: group.clone(),
            }))
            .map_err(eyre::Report::msg)?;
        persist(tokio, repository, scheduler)?;
        tracing::info!(
            event = "recording_scheduler_due",
            occurrence_id = %occurrence_id,
            group_id = %group_id,
            "scheduled recording start intent emitted"
        );
        send(node, output, &value)?;
        send(node, occurrence_output, &occurrence_status)?;
    }
    Ok(())
}

fn emit_power_reservations(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    node: &mut DoraNode,
    output: &DataId,
    statuses: &BTreeMap<String, PowerStatus>,
    scheduler: &mut SchedulerRuntime<SystemClock>,
) -> Result<()> {
    for group_id in scheduler.pending_reservations() {
        let Some(group) = scheduler.groups.get(&group_id) else {
            continue;
        };
        let Some(status) = statuses.get(&group.entity_id) else {
            continue;
        };
        let Some(command) = reservation_command(group, status, false, SystemClock.now_ms()) else {
            continue;
        };
        let record = ReservationCommandOutboxRecord {
            group_id: group_id.clone(),
            reservation_id: group
                .power_reservation
                .as_ref()
                .expect("reservation checked")
                .reservation_id
                .clone(),
            action: ReservationCommandAction::Register,
            command: command.clone(),
            created_at_ms: SystemClock.now_ms(),
        };
        if !scheduler.mark_reservation_registering(&group_id, command.command_id.clone()) {
            continue;
        }
        persist(tokio, repository, scheduler)?;
        tokio
            .block_on(repository.persist_reservation_command(&record))
            .map_err(eyre::Report::msg)?;
        send(node, output, &command)?;
    }
    Ok(())
}

fn emit_power_releases(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    node: &mut DoraNode,
    output: &DataId,
    statuses: &BTreeMap<String, PowerStatus>,
    scheduler: &mut SchedulerRuntime<SystemClock>,
) -> Result<()> {
    for group_id in scheduler.reservations_to_release() {
        let Some(group) = scheduler.groups.get(&group_id) else {
            continue;
        };
        let Some(status) = statuses.get(&group.entity_id) else {
            continue;
        };
        let Some(command) = reservation_command(group, status, true, SystemClock.now_ms()) else {
            continue;
        };
        let record = ReservationCommandOutboxRecord {
            group_id: group_id.clone(),
            reservation_id: group
                .power_reservation
                .as_ref()
                .expect("reservation checked")
                .reservation_id
                .clone(),
            action: ReservationCommandAction::Release,
            command: command.clone(),
            created_at_ms: SystemClock.now_ms(),
        };
        if !scheduler.mark_reservation_releasing(&group_id, command.command_id.clone()) {
            continue;
        }
        persist(tokio, repository, scheduler)?;
        tokio
            .block_on(repository.persist_reservation_command(&record))
            .map_err(eyre::Report::msg)?;
        send(node, output, &command)?;
    }
    Ok(())
}

fn reservation_command(
    group: &crate::domain::RecordingGroup,
    status: &PowerStatus,
    release: bool,
    now_ms: i64,
) -> Option<PowerCommand> {
    let reservation = group.power_reservation.as_ref()?;
    let now_ms = u64::try_from(now_ms).ok()?;
    let command_id = Uuid::new_v5(
        &Uuid::parse_str(&reservation.reservation_id).ok()?,
        format!(
            "{}:{}",
            if release { "release" } else { "register" },
            reservation.command_attempt.saturating_add(1)
        )
        .as_bytes(),
    )
    .to_string();
    let action = if release {
        PowerCommandAction::ReleaseReservation {
            reservation_id: reservation.reservation_id.clone(),
        }
    } else {
        let not_before_ms = u64::try_from(reservation.prewarm_at_ms.max(now_ms as i64)).ok()?;
        let expires_at_ms = u64::try_from(reservation.expires_at_ms).ok()?;
        if expires_at_ms <= not_before_ms || expires_at_ms.saturating_sub(now_ms) > 604_800_000 {
            return None;
        }
        PowerCommandAction::RegisterReservation {
            reservation: PowerReservation {
                protocol_version: POWER_PROTOCOL_VERSION,
                reservation_id: reservation.reservation_id.clone(),
                role: LifecycleRole::Rover,
                entity_id: group.entity_id.clone(),
                authority: status.authority,
                required_profile: PowerProfile::ScheduledCapture,
                issued_at_ms: now_ms,
                not_before_ms,
                expires_at_ms,
            },
        }
    };
    Some(PowerCommand {
        protocol_version: POWER_PROTOCOL_VERSION,
        command_id,
        role: LifecycleRole::Rover,
        entity_id: group.entity_id.clone(),
        authority: status.authority,
        action,
        issued_at_ms: now_ms,
        not_before_ms: now_ms,
        expires_at_ms: now_ms.saturating_add(60_000),
        detail: None,
    })
}

fn emit_stops(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    node: &mut DoraNode,
    output: &DataId,
    occurrence_output: &DataId,
    scheduler: &mut SchedulerRuntime<SystemClock>,
) -> Result<()> {
    for occurrence_id in scheduler.due_stops() {
        let Some(transition) = scheduler.begin_stop(&occurrence_id) else {
            continue;
        };
        let Some(intent_id) = transition.intent_id else {
            persist(tokio, repository, scheduler)?;
            continue;
        };
        let occurrence = &scheduler.occurrences[&occurrence_id];
        let group = &scheduler.groups[occurrence.group_id.as_ref().unwrap()];
        let group_id = group.group_id.clone();
        let value = build_intent(
            occurrence,
            group,
            intent_id,
            ScheduledRecordingIntentAction::Release,
        );
        let occurrence_status = occurrence.clone();
        tokio
            .block_on(repository.persist_transition(&OutboxRecord {
                intent: value.clone(),
                occurrence: occurrence.clone(),
                group: group.clone(),
            }))
            .map_err(eyre::Report::msg)?;
        persist(tokio, repository, scheduler)?;
        tracing::info!(
            event = "recording_scheduler_stop",
            occurrence_id = %occurrence_id,
            group_id = %group_id,
            "scheduled recording stop intent emitted"
        );
        send(node, output, &value)?;
        send(node, occurrence_output, &occurrence_status)?;
    }
    Ok(())
}

fn replay_pending(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    node: &mut DoraNode,
    output: &DataId,
    scheduler: &mut SchedulerRuntime<SystemClock>,
) -> Result<()> {
    let records = tokio
        .block_on(repository.pending_outbox())
        .map_err(eyre::Report::msg)?;
    let recovery = scheduler.recover_outbox(&records);
    persist(tokio, repository, scheduler)?;
    for intent_id in recovery.acknowledge {
        tokio
            .block_on(repository.acknowledge_intent(&intent_id))
            .map_err(eyre::Report::msg)?;
    }
    for intent in recovery.replay {
        send(node, output, &intent)?;
    }
    Ok(())
}

fn replay_reservation_commands(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    node: &mut DoraNode,
    output: &DataId,
    scheduler: &mut SchedulerRuntime<SystemClock>,
) -> Result<()> {
    let records = tokio
        .block_on(repository.pending_reservation_commands())
        .map_err(eyre::Report::msg)?;
    let command_ids = records
        .iter()
        .map(|record| record.command.command_id.clone())
        .collect::<Vec<_>>();
    if scheduler.repair_reservation_outbox(&command_ids) {
        persist(tokio, repository, scheduler)?;
    }
    let now_ms = SystemClock.now_ms();
    for record in records {
        if record.is_expired(now_ms) {
            if scheduler.expire_reservation_command(record.command_id()) {
                persist(tokio, repository, scheduler)?;
            }
            tokio
                .block_on(repository.acknowledge_reservation_command(record.command_id()))
                .map_err(eyre::Report::msg)?;
            continue;
        }
        send(node, output, &record.command)?;
    }
    Ok(())
}

fn replay_desired_groups(
    node: &mut DoraNode,
    output: &DataId,
    scheduler: &SchedulerRuntime<SystemClock>,
) -> Result<()> {
    for group in scheduler
        .groups
        .values()
        .filter(|group| !group.owner_ids.is_empty())
    {
        let Some(occurrence) = scheduler.occurrences.values().find(|occurrence| {
            occurrence.group_id.as_deref() == Some(&group.group_id)
                && !occurrence.state.is_terminal()
        }) else {
            continue;
        };
        let intent_id = scheduled_intent_id(
            &occurrence.occurrence_id,
            group.generation,
            ScheduledRecordingIntentAction::Acquire,
        )
        .map_err(eyre::Report::msg)?;
        let intent = build_intent(
            occurrence,
            group,
            intent_id,
            ScheduledRecordingIntentAction::Acquire,
        );
        send(node, output, &intent)?;
    }
    Ok(())
}

/// Replays live protected work after reconciliation so a restarted coordinator
/// cannot infer that an in-progress recording is safe to quiesce.
fn replay_protected_occurrences(
    node: &mut DoraNode,
    output: &DataId,
    scheduler: &SchedulerRuntime<SystemClock>,
) -> Result<()> {
    for occurrence in scheduler
        .occurrences
        .values()
        .filter(|occurrence| is_protected_occurrence(occurrence))
    {
        send(node, output, occurrence)?;
    }
    Ok(())
}

fn replay_protected_work_snapshots(
    node: &mut DoraNode,
    output: &DataId,
    scheduler: &SchedulerRuntime<SystemClock>,
    generated_at_ms: u64,
) -> Result<()> {
    let entities = scheduler
        .occurrences
        .values()
        .map(|occurrence| occurrence.entity_id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    for entity_id in entities {
        send(
            node,
            output,
            &protected_work_snapshot_for(scheduler, entity_id, generated_at_ms),
        )?;
    }
    Ok(())
}

fn protected_work_snapshot_for(
    scheduler: &SchedulerRuntime<SystemClock>,
    entity_id: String,
    generated_at_ms: u64,
) -> ProtectedWorkSnapshot {
    ProtectedWorkSnapshot {
        protocol_version: robo_rover_lib::PROTECTED_WORK_RELAY_PROTOCOL_VERSION,
        snapshot_id: uuid::Uuid::new_v4().to_string(),
        entity_id: entity_id.clone(),
        generated_at_ms,
        occurrences: scheduler
            .occurrences
            .values()
            .filter(|occurrence| {
                occurrence.entity_id == entity_id
                    && occurrence_requires_protection(occurrence.state)
            })
            .cloned()
            .collect(),
    }
}

fn is_protected_occurrence(occurrence: &RecordingOccurrence) -> bool {
    occurrence_requires_protection(occurrence.state)
}

fn replay_manual_suppression_acks(
    node: &mut DoraNode,
    output: &DataId,
    scheduler: &SchedulerRuntime<SystemClock>,
) -> Result<()> {
    for group in scheduler
        .groups
        .values()
        .filter(|group| group.owner_ids.is_empty())
    {
        let Some(occurrence) = scheduler.occurrences.values().find(|occurrence| {
            occurrence.group_id.as_deref() == Some(&group.group_id)
                && occurrence.suppressed_by_manual
        }) else {
            continue;
        };
        send(
            node,
            output,
            &RecordingCoordinatorFeedback {
                intent_id: scheduled_intent_id(
                    &occurrence.occurrence_id,
                    group.generation,
                    ScheduledRecordingIntentAction::Acquire,
                )
                .map_err(eyre::Report::msg)?,
                occurrence_id: occurrence.occurrence_id.clone(),
                generation: group.generation,
                accepted: true,
                applied: true,
                retryable: false,
                group_id: Some(group.group_id.clone()),
                recording_id: None,
                recorder_state: None,
                manual_suppression: true,
                reason_code: None,
                detail: Some("replayed durable manual suppression".into()),
            },
        )?;
    }
    Ok(())
}

fn decode<T: serde::de::DeserializeOwned>(data: &dyn Array) -> Option<T> {
    data.as_any()
        .downcast_ref::<BinaryArray>()
        .and_then(|array| {
            (array.len() == 1 && !array.is_null(0))
                .then(|| serde_json::from_slice(array.value(0)).ok())
        })
        .flatten()
}

fn send<T: serde::Serialize>(node: &mut DoraNode, output: &DataId, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    node.send_output(
        output.clone(),
        MetadataParameters::default(),
        BinaryArray::from_vec(vec![bytes.as_slice()]),
    )?;
    Ok(())
}

fn publish_status(
    node: &mut DoraNode,
    output: &DataId,
    readiness: RecordingSchedulerReadiness,
    detail: &str,
) -> Result<()> {
    send(
        node,
        output,
        &RecordingSchedulerStatus {
            protocol_version: RECORDING_SCHEDULE_PROTOCOL_VERSION,
            readiness,
            detail: Some(detail.chars().take(256).collect()),
        },
    )
}

fn log_metrics(scheduler: &SchedulerRuntime<SystemClock>, phase: &str) {
    tracing::info!(
        event = "recording_scheduler_metrics",
        phase,
        occurrence_count = scheduler.occurrences.len(),
        group_count = scheduler.groups.len(),
        active_group_count = scheduler
            .groups
            .values()
            .filter(|group| !group.owner_ids.is_empty())
            .count(),
        "recording scheduler metrics snapshot"
    );
}

#[cfg(test)]
mod tests {
    use super::ReconciliationState;

    #[test]
    fn reconciliation_timeout_recovers_after_the_current_snapshot_arrives() {
        let mut state = ReconciliationState::new(1_000, 30_000);
        let request_id = state.request_id.clone();

        assert!(!state.mark_degraded_if_overdue(30_999));
        assert!(state.mark_degraded_if_overdue(31_000));
        assert!(!state.mark_degraded_if_overdue(31_001));
        assert!(state.accepts(&request_id));
        assert!(state.mark_ready());
        assert!(!state.mark_degraded_if_overdue(1_000_000));
        assert!(!state.mark_ready());
    }

    #[test]
    fn reconciliation_only_accepts_the_most_recent_request() {
        let mut state = ReconciliationState::new(0, 1_000);
        let obsolete = state.request_id.clone();
        let current = state.next_request(1_000).request_id;

        assert!(!state.accepts(&obsolete));
        assert!(state.accepts(&current));
        assert!(!state.mark_degraded_if_overdue(1_999));
        assert!(state.mark_degraded_if_overdue(2_000));
    }
}
