use bson::doc;
use mongodb::{options::ClientOptions, Client};
use recording_scheduler::{domain::RecordingGroup, mongo_repository::MongoRepository};
use robo_rover_lib::{
    scheduled_intent_id, DstResolution, RecordingOccurrence, RecordingOccurrenceState,
    ScheduledRecordingIntent, ScheduledRecordingIntentAction,
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
            group_id: group.group_id,
            generation: 1,
            entity_id: occurrence.entity_id.clone(),
            start_request_id: occurrence.start_request_id.clone(),
            planned_start_ms: occurrence.planned_start_ms,
            planned_end_ms: occurrence.planned_end_ms,
            relative_directory: "scheduled".into(),
            action: ScheduledRecordingIntentAction::Acquire,
        };
        repository.persist_intent(&intent).await.unwrap();
        repository.persist_intent(&intent).await.unwrap();
        assert_eq!(
            repository.pending_intents().await.unwrap(),
            vec![intent.clone()]
        );
        repository
            .acknowledge_intent(&intent.intent_id)
            .await
            .unwrap();
        assert!(repository.pending_intents().await.unwrap().is_empty());

        database
            .collection::<bson::Document>("recording_occurrences")
            .find_one(doc! {"occurrence_id": occurrence.occurrence_id}, None)
            .await
            .unwrap()
            .unwrap();
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
