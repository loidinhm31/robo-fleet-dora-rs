use mongodb::bson::{doc, DateTime};
use power_coordinator::{JournalIntent, JournalRecord};
use power_event_projector::mongo_repository::MongoRepository;
use robo_rover_lib::{
    LifecycleRole, PowerAuthority, PowerEvent, PowerEventType, PowerPolicy, PowerProfile,
    PowerState, PowerStatus, POWER_PROTOCOL_VERSION,
};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

fn record(epoch: u64, sequence: u64, at: u64) -> JournalRecord {
    let authority = PowerAuthority { epoch, sequence };
    JournalRecord {
        format_version: 1,
        sequence,
        intent: JournalIntent::TransitionApplied,
        event: PowerEvent {
            protocol_version: POWER_PROTOCOL_VERSION,
            event_id: Uuid::new_v4().hyphenated().to_string(),
            role: LifecycleRole::Rover,
            entity_id: "rover-kiwi".into(),
            authority,
            transition_id: None,
            event_type: PowerEventType::TransitionApplied,
            reason_code: None,
            detail: None,
            occurred_at_ms: at,
        },
        status: Some(PowerStatus {
            protocol_version: POWER_PROTOCOL_VERSION,
            role: LifecycleRole::Rover,
            entity_id: "rover-kiwi".into(),
            authority,
            policy: PowerPolicy::Awake,
            requested_profile: PowerProfile::NormalRover,
            effective_profile: PowerProfile::NormalRover,
            state: PowerState::Active,
            transition_id: None,
            reason_code: None,
            detail: None,
            updated_at_ms: at,
        }),
    }
}

#[test]
fn duplicate_and_reordered_events_do_not_regress_current_state() {
    let Ok(uri) = std::env::var("POWER_PROJECTOR_TEST_MONGODB_URI") else {
        return;
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let database = format!("power_projector_test_{}", Uuid::new_v4());
        let repository = MongoRepository::connect(&uri, &database).await.unwrap();
        repository.ensure_indexes().await.unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let newer = record(2, 5, now);
        let older = record(1, 99, now - 1);
        let same_epoch_stale = record(2, 4, now - 2);
        repository.project("test", &newer).await.unwrap();
        repository.project("test", &newer).await.unwrap();
        repository.project("test", &older).await.unwrap();
        repository.project("test", &same_epoch_stale).await.unwrap();
        let client = mongodb::Client::with_uri_str(&uri).await.unwrap();
        let current = client
            .database(&database)
            .collection::<mongodb::bson::Document>("power_current_state")
            .find_one(doc! {"deployment_id": "test"}, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.get_i64("authority_epoch").unwrap(), 2);
        assert_eq!(current.get_i64("sequence").unwrap(), 5);
        let history = repository
            .history("test", "rover-kiwi", Some(0), Some(now as i64), None, 10)
            .await
            .unwrap();
        assert_eq!(history.len(), 3);
        let following = repository
            .history(
                "test",
                "rover-kiwi",
                Some(0),
                Some(now as i64),
                Some((DateTime::from_millis(now as i64), newer.event.event_id)),
                10,
            )
            .await
            .unwrap();
        assert_eq!(following.len(), 2);
        client.database(&database).drop(None).await.unwrap();
    });
}
