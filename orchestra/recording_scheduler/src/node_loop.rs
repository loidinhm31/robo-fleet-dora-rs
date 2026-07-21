use std::time::Duration;

use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event, MetadataParameters,
};
use eyre::Result;
use robo_rover_lib::{
    scheduled_intent_id, AuthenticatedRecordingScheduleCommand, RecordingCoordinatorFeedback,
    RecordingReconciliationRequest, RecordingReconciliationSnapshot, RecordingScheduleQuery,
    RecordingScheduleSnapshot, RecordingSchedulerReadiness, RecordingSchedulerStatus,
    ScheduledRecordingIntentAction, RECORDING_SCHEDULE_PROTOCOL_VERSION,
};

use crate::{
    clock::{Clock, SystemClock},
    config::SchedulerConfig,
    mongo_repository::MongoRepository,
    mongo_repository::OutboxRecord,
    node_intents::build_intent,
    node_persistence::{adopt, persist},
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
    let intent = DataId::from("scheduled_recording_intent".to_owned());
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
                        replay_manual_suppression_acks(
                            &mut node,
                            &manual_suppression_ack,
                            &scheduler,
                        )?;
                        log_metrics(&scheduler, "reconciliation");
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
