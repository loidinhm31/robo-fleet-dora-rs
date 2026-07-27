use mongodb::bson::{doc, DateTime};
use power_coordinator::{JournalIntent, JournalRecord};
use power_event_projector::mongo_repository::{HistoryFilter, MongoRepository};
use power_event_projector::projector::PowerEventProjector;
use robo_rover_lib::{
    LifecycleRole, PowerAuthority, PowerDemandSource, PowerEvent, PowerEventContext,
    PowerEventType, PowerPolicy, PowerProfile, PowerState, PowerStatus, POWER_PROTOCOL_VERSION,
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
            context: PowerEventContext {
                demand_source: Some(PowerDemandSource::Scheduler),
                ..Default::default()
            },
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
            active_reservations: vec![],
            updated_at_ms: at,
        }),
    }
}

#[test]
fn projector_reports_a_bounded_failure_when_mongo_is_unavailable() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let repository = MongoRepository::connect(
            "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=10",
            "power_projector_unavailable",
        )
        .await
        .unwrap();
        let projector = PowerEventProjector::new("test".into(), repository);
        let health = projector.project_with_retry(&record(1, 1, 100), 2, 1).await;
        assert!(!health.healthy);
        assert_eq!(health.attempts, 2);
        assert!(health.reason.is_some());
        let startup = PowerEventProjector::open_with_retry(
            "test".into(),
            "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=10",
            "power_projector_unavailable",
            2,
            1,
        )
        .await;
        let Err(startup) = startup else {
            panic!("unavailable Mongo must fail startup after bounded retries");
        };
        assert_eq!(startup.attempts, 2);
    });
}

#[test]
#[ignore = "requires POWER_PROJECTOR_TEST_MONGODB_URI; run make test-power-projector-mongo"]
fn duplicate_and_reordered_events_do_not_regress_current_state() {
    let uri = std::env::var("POWER_PROJECTOR_TEST_MONGODB_URI")
        .expect("POWER_PROJECTOR_TEST_MONGODB_URI is required for the Mongo integration gate");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let database = format!("power_projector_test_{}", Uuid::new_v4());
        let projector = PowerEventProjector::open_with_retry("test".into(), &uri, &database, 2, 1)
            .await
            .unwrap();
        let repository = MongoRepository::connect(&uri, &database).await.unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let newer = record(2, 5, now);
        let mut command_intent = record(3, 1, now - 3);
        command_intent.intent = JournalIntent::Command;
        command_intent.event.event_type = PowerEventType::PolicyChanged;
        command_intent.status.as_mut().unwrap().policy = PowerPolicy::Sleep;
        let older = record(1, 99, now - 1);
        let same_epoch_stale = record(2, 4, now - 2);
        projector.project(&command_intent).await.unwrap();
        projector.project(&newer).await.unwrap();
        projector.project(&newer).await.unwrap();
        projector.project(&older).await.unwrap();
        projector.project(&same_epoch_stale).await.unwrap();
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
        assert_eq!(history.len(), 4);
        let scheduled = repository
            .history_filtered(
                "test",
                "rover-kiwi",
                Some(0),
                Some(now as i64),
                None,
                10,
                &HistoryFilter {
                    demand_source: Some(PowerDemandSource::Scheduler),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(scheduled.len(), 4);
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
        assert_eq!(following.len(), 3);
        let mut first_tie = record(4, 1, now - 10);
        first_tie.event.event_id = "11111111-1111-4111-8111-111111111111".into();
        let mut second_tie = record(4, 2, now - 10);
        second_tie.event.event_id = "22222222-2222-4222-8222-222222222222".into();
        projector.project(&first_tie).await.unwrap();
        projector.project(&second_tie).await.unwrap();
        let after_first_tie = repository
            .history(
                "test",
                "rover-kiwi",
                Some((now - 10) as i64),
                Some((now - 10) as i64),
                Some((
                    DateTime::from_millis((now - 10) as i64),
                    first_tie.event.event_id.clone(),
                )),
                10,
            )
            .await
            .unwrap();
        assert_eq!(after_first_tie.len(), 1);
        assert_eq!(
            after_first_tie[0].get_str("event_id").unwrap(),
            second_tie.event.event_id
        );
        client.database(&database).drop(None).await.unwrap();
    });
}
