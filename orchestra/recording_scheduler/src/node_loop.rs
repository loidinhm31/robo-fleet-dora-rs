use std::time::Duration;

use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event, MetadataParameters,
};
use eyre::Result;
use robo_rover_lib::{
    AuthenticatedRecordingScheduleCommand, RecordingCoordinatorFeedback,
    RecordingReconciliationRequest, RecordingReconciliationSnapshot,
    ScheduledRecordingIntentAction,
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

pub fn run() -> Result<()> {
    let config = SchedulerConfig::from_env().map_err(eyre::Report::msg)?;
    let tokio = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let repository = tokio.block_on(async {
        let repository =
            MongoRepository::connect(&config.mongodb_uri, &config.mongodb_database).await?;
        repository.ensure_indexes().await?;
        Ok::<_, mongodb::error::Error>(repository)
    })?;
    let mut scheduler = load_scheduler(&tokio, &repository)?;
    let service = ScheduleService::new(SystemClock, config.clone(), repository.clone());
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let result = DataId::from("recording_schedule_command_result".to_owned());
    let intent = DataId::from("scheduled_recording_intent".to_owned());
    let reconcile = DataId::from("recording_reconciliation_request".to_owned());
    let mut reconciliation_id = uuid::Uuid::new_v4().to_string();
    send(
        &mut node,
        &reconcile,
        &RecordingReconciliationRequest {
            request_id: reconciliation_id.clone(),
            entity_id: None,
        },
    )?;
    let mut next_refresh_ms = 0;
    let mut next_snapshot_ms = SystemClock.now_ms() + config.reconcile_seconds as i64 * 1_000;
    let mut initial_reconciliation_complete = false;
    tracing::info!(
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
                        if response.accepted {
                            scheduler = load_scheduler(&tokio, &repository)?;
                            if initial_reconciliation_complete {
                                scheduler.complete_reconciliation();
                            }
                        }
                        send(&mut node, &result, &response)?;
                    }
                }
                "recording_scheduler_recorder_feedback" => {
                    if let Some(value) = decode::<RecordingCoordinatorFeedback>(&*data) {
                        if scheduler.apply_feedback(value.clone()) {
                            persist(&tokio, &repository, &mut scheduler)?;
                            tokio
                                .block_on(repository.acknowledge_intent(&value.intent_id))
                                .map_err(eyre::Report::msg)?;
                        } else if scheduler.has_handled_intent(&value.intent_id) {
                            tokio
                                .block_on(repository.acknowledge_intent(&value.intent_id))
                                .map_err(eyre::Report::msg)?;
                        }
                    }
                }
                "recording_reconciliation_snapshot" => {
                    if let Some(value) = decode::<RecordingReconciliationSnapshot>(&*data) {
                        if value.request_id != reconciliation_id {
                            continue;
                        }
                        adopt(&mut scheduler, value);
                        let first_snapshot = !initial_reconciliation_complete;
                        if first_snapshot {
                            scheduler.complete_reconciliation();
                            initial_reconciliation_complete = true;
                        }
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
                        tracing::info!("recording scheduler reconciliation applied");
                    }
                }
                _ => {}
            }
        }
        let now_ms = SystemClock.now_ms();
        if now_ms >= next_refresh_ms {
            refresh(
                &tokio,
                &repository,
                &mut scheduler,
                now_ms + config.horizon_days * 86_400_000,
            )?;
            next_refresh_ms = now_ms + config.reconcile_seconds as i64 * 1_000;
        }
        if now_ms >= next_snapshot_ms {
            reconciliation_id = uuid::Uuid::new_v4().to_string();
            send(
                &mut node,
                &reconcile,
                &RecordingReconciliationRequest {
                    request_id: reconciliation_id.clone(),
                    entity_id: None,
                },
            )?;
            next_snapshot_ms = now_ms + config.reconcile_seconds as i64 * 1_000;
        }
        emit_due(&tokio, &repository, &mut node, &intent, &mut scheduler)?;
        emit_stops(&tokio, &repository, &mut node, &intent, &mut scheduler)?;
    }
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
        let value = build_intent(
            occurrence,
            group,
            intent_id,
            ScheduledRecordingIntentAction::Acquire,
        );
        tokio
            .block_on(repository.persist_transition(&OutboxRecord {
                intent: value.clone(),
                occurrence: occurrence.clone(),
                group: group.clone(),
            }))
            .map_err(eyre::Report::msg)?;
        persist(tokio, repository, scheduler)?;
        send(node, output, &value)?;
    }
    Ok(())
}

fn emit_stops(
    tokio: &tokio::runtime::Runtime,
    repository: &MongoRepository,
    node: &mut DoraNode,
    output: &DataId,
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
        let value = build_intent(
            occurrence,
            group,
            intent_id,
            ScheduledRecordingIntentAction::Release,
        );
        tokio
            .block_on(repository.persist_transition(&OutboxRecord {
                intent: value.clone(),
                occurrence: occurrence.clone(),
                group: group.clone(),
            }))
            .map_err(eyre::Report::msg)?;
        persist(tokio, repository, scheduler)?;
        send(node, output, &value)?;
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
