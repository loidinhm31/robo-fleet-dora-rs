use bson::doc;
use mongodb::{options::ClientOptions, Client};
use recording_scheduler::{
    clock::FakeClock,
    domain::RecordingGroup,
    mongo_repository::{MongoRepository, OutboxRecord},
    runtime::SchedulerRuntime,
};
use robo_rover_lib::{
    scheduled_intent_id, DstResolution, RecordingLocalStart, RecordingOccurrence,
    RecordingOccurrenceState, RecordingSchedule, RecordingScheduleDefinition,
    RecordingScheduleRecurrence, ScheduledRecordingIntent, ScheduledRecordingIntentAction,
};

#[test]
fn local_mongo_persists_occurrences_groups_and_outbox_idempotently() {
    let Ok(uri) = std::env::var("SCHEDULER_TEST_MONGODB_URI") else {
        return;
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let database_name = format!("recording_scheduler_test_{}", uuid::Uuid::new_v4());
        let client = Client::with_options(ClientOptions::parse(&uri).await.unwrap()).unwrap();
        let database = client.database(&database_name);
        let repository = MongoRepository::from_database(database.clone());
        repository.ensure_indexes().await.unwrap();

        repository.insert_schedule(&schedule(1)).await.unwrap();
        let occurrence = occurrence();
        assert_eq!(
            repository.save_occurrence(&occurrence, 0).await.unwrap(),
            Some(1)
        );
        assert_eq!(
            repository.save_occurrence(&occurrence, 0).await.unwrap(),
            None
        );
        assert_eq!(
            repository.save_occurrence(&occurrence, 1).await.unwrap(),
            Some(2)
        );
        assert_eq!(
            repository.load_nonterminal().await.unwrap(),
            vec![occurrence.clone()]
        );
        repository
            .cancel_future(&occurrence.schedule_id, occurrence.schedule_revision, 0)
            .await
            .unwrap();
        assert_eq!(
            repository.save_occurrence(&occurrence, 2).await.unwrap(),
            None
        );
        assert!(repository.load_nonterminal().await.unwrap().is_empty());

        let group = RecordingGroup::new(
            "kiwi-1",
            occurrence.planned_start_ms,
            occurrence.planned_end_ms,
            "scheduled".into(),
        )
        .unwrap();
        assert_eq!(repository.save_group(&group, 0).await.unwrap(), Some(1));
        assert_eq!(repository.save_group(&group, 0).await.unwrap(), None);
        let intent = ScheduledRecordingIntent {
            intent_id: scheduled_intent_id(
                &occurrence.occurrence_id,
                1,
                ScheduledRecordingIntentAction::Acquire,
            )
            .unwrap(),
            occurrence_id: occurrence.occurrence_id.clone(),
            group_id: group.group_id.clone(),
            generation: 1,
            entity_id: occurrence.entity_id.clone(),
            start_request_id: occurrence.start_request_id.clone(),
            planned_start_ms: occurrence.planned_start_ms,
            planned_end_ms: occurrence.planned_end_ms,
            relative_directory: "scheduled".into(),
            reservation_id: None,
            action: ScheduledRecordingIntentAction::Acquire,
        };
        let mut pending_occurrence = occurrence.clone();
        pending_occurrence.group_id = Some(group.group_id.clone());
        let record = OutboxRecord {
            intent: intent.clone(),
            occurrence: pending_occurrence,
            group: group.clone(),
        };
        repository.persist_transition(&record).await.unwrap();
        repository.persist_transition(&record).await.unwrap();
        assert_eq!(
            repository.pending_outbox().await.unwrap(),
            vec![record.clone()]
        );
        repository
            .acknowledge_intent(&intent.intent_id)
            .await
            .unwrap();
        assert!(repository.pending_outbox().await.unwrap().is_empty());

        database
            .collection::<bson::Document>("recording_occurrences")
            .find_one(doc! {"occurrence_id": occurrence.occurrence_id}, None)
            .await
            .unwrap()
            .unwrap();
        database.drop(None).await.unwrap();
    });
}

#[test]
fn local_mongo_recovers_update_and_delete_cancellation_races() {
    let Ok(uri) = std::env::var("SCHEDULER_TEST_MONGODB_URI") else {
        return;
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let database_name = format!("recording_scheduler_test_{}", uuid::Uuid::new_v4());
        let client = Client::with_options(ClientOptions::parse(&uri).await.unwrap()).unwrap();
        let database = client.database(&database_name);
        let repository = MongoRepository::from_database(database.clone());
        repository.ensure_indexes().await.unwrap();

        let initial = schedule(1);
        repository.insert_schedule(&initial).await.unwrap();
        let stale = occurrence();
        repository.save_occurrence(&stale, 0).await.unwrap();
        let updated = schedule(2);
        assert!(repository.replace_schedule_cas(&updated, 1).await.unwrap());
        repository.recover_superseded_future(0).await.unwrap();
        // Simulates an old materializer that passed its last read before the CAS.
        let late = RecordingOccurrence {
            occurrence_id: "00000000-0000-0000-0000-000000000303".into(),
            ..stale.clone()
        };
        assert_eq!(repository.save_occurrence(&late, 0).await.unwrap(), None);
        assert!(repository.load_nonterminal().await.unwrap().is_empty());

        let enabled_occurrence = RecordingOccurrence {
            occurrence_id: "00000000-0000-0000-0000-000000000304".into(),
            schedule_revision: 2,
            ..stale
        };
        repository
            .save_occurrence(&enabled_occurrence, 0)
            .await
            .unwrap();
        let disabled = RecordingSchedule {
            revision: 3,
            definition: RecordingScheduleDefinition {
                enabled: false,
                ..updated.definition.clone()
            },
            ..updated.clone()
        };
        assert!(repository.replace_schedule_cas(&disabled, 2).await.unwrap());
        repository.recover_superseded_future(0).await.unwrap();
        assert!(repository.load_nonterminal().await.unwrap().is_empty());

        let deleted = RecordingSchedule {
            revision: 4,
            ..disabled
        };
        assert!(repository
            .tombstone_schedule_cas(&deleted, 3, 0)
            .await
            .unwrap());
        repository.recover_superseded_future(0).await.unwrap();
        assert!(repository.load_schedules().await.unwrap().is_empty());
        assert!(repository
            .find_schedule(&deleted.schedule_id)
            .await
            .unwrap()
            .is_none());
        database.drop(None).await.unwrap();
    });
}

#[test]
fn local_mongo_recovers_after_outbox_write_before_state_writes() {
    let Ok(uri) = std::env::var("SCHEDULER_TEST_MONGODB_URI") else {
        return;
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let database_name = format!("recording_scheduler_test_{}", uuid::Uuid::new_v4());
        let client = Client::with_options(ClientOptions::parse(&uri).await.unwrap()).unwrap();
        let database = client.database(&database_name);
        let repository = MongoRepository::from_database(database.clone());
        repository.ensure_indexes().await.unwrap();

        let mut occurrence = occurrence();
        let mut group = RecordingGroup::new(
            &occurrence.entity_id,
            occurrence.planned_start_ms,
            occurrence.planned_end_ms,
            "scheduled".into(),
        )
        .unwrap();
        group.add_owner(&occurrence.occurrence_id);
        let intent_id = scheduled_intent_id(
            &occurrence.occurrence_id,
            group.generation,
            ScheduledRecordingIntentAction::Acquire,
        )
        .unwrap();
        group.begin_intent(intent_id.clone(), ScheduledRecordingIntentAction::Acquire);
        occurrence.state = RecordingOccurrenceState::StartPending;
        occurrence.group_id = Some(group.group_id.clone());
        let intent = ScheduledRecordingIntent {
            intent_id,
            occurrence_id: occurrence.occurrence_id.clone(),
            group_id: group.group_id.clone(),
            generation: group.generation,
            entity_id: occurrence.entity_id.clone(),
            start_request_id: group.start_request_id.clone(),
            planned_start_ms: occurrence.planned_start_ms,
            planned_end_ms: occurrence.planned_end_ms,
            relative_directory: group.relative_directory.clone(),
            reservation_id: None,
            action: ScheduledRecordingIntentAction::Acquire,
        };
        repository
            .persist_transition(&OutboxRecord {
                intent: intent.clone(),
                occurrence,
                group,
            })
            .await
            .unwrap();

        let mut recovered = SchedulerRuntime::from_persisted(
            FakeClock::new(0),
            repository.load_nonterminal_stored().await.unwrap(),
            repository.load_groups().await.unwrap(),
        )
        .unwrap();
        let replay = recovered.recover_outbox(&repository.pending_outbox().await.unwrap());
        assert_eq!(replay.replay, vec![intent]);
        assert!(replay.acknowledge.is_empty());
        assert_eq!(recovered.occurrences.len(), 1);
        assert_eq!(recovered.groups.len(), 1);
        database.drop(None).await.unwrap();
    });
}

fn occurrence() -> RecordingOccurrence {
    RecordingOccurrence {
        occurrence_id: "00000000-0000-0000-0000-000000000300".into(),
        schedule_id: "00000000-0000-0000-0000-000000000301".into(),
        schedule_revision: 1,
        entity_id: "kiwi-1".into(),
        planned_start_ms: 1_000_000,
        planned_end_ms: 1_060_000,
        dst_resolution: DstResolution::Exact,
        state: RecordingOccurrenceState::Planned,
        retry_count: 0,
        next_retry_at_ms: None,
        group_id: None,
        start_request_id: "00000000-0000-0000-0000-000000000302".into(),
        attempts: Vec::new(),
        last_error: None,
        suppressed_by_manual: false,
        created_at_ms: 999_000,
        updated_at_ms: 999_000,
        terminal_at_ms: None,
        expires_at_ms: None,
    }
}

fn schedule(revision: u64) -> RecordingSchedule {
    RecordingSchedule {
        schedule_id: "00000000-0000-0000-0000-000000000301".into(),
        revision,
        definition: RecordingScheduleDefinition {
            entity_id: "kiwi-1".into(),
            title: "test".into(),
            enabled: true,
            recurrence: RecordingScheduleRecurrence::OneTime {
                local_start: RecordingLocalStart {
                    date: "2026-01-01".into(),
                    time: "01:00".into(),
                    timezone: "UTC".into(),
                },
            },
            duration_ms: 60_000,
            relative_directory_template: "scheduled".into(),
        },
        created_at_ms: 0,
        created_by: "user".into(),
        updated_at_ms: 0,
        updated_by: "user".into(),
    }
}
