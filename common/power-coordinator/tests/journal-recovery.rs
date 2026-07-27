use power_coordinator::{
    CoordinatorConfig, CoordinatorTime, DurablePowerCoordinator, EventJournal, JournalAppendClass,
    JournalConfig, JournalIntent, JournalRecord,
};
use robo_rover_lib::{
    LifecycleRole, PowerAuthority, PowerCommand, PowerCommandAction, PowerEvent, PowerEventType,
    PowerPolicy, PowerProfile, PowerState, PowerStatus, POWER_PROTOCOL_VERSION,
};
use std::{fs::OpenOptions, io::Write};
use tempfile::TempDir;
use uuid::Uuid;

fn record(epoch: u64) -> JournalRecord {
    let authority = PowerAuthority { epoch, sequence: 1 };
    JournalRecord {
        format_version: 1,
        sequence: 0,
        intent: JournalIntent::Transition,
        event: PowerEvent {
            protocol_version: POWER_PROTOCOL_VERSION,
            event_id: Uuid::new_v4().hyphenated().to_string(),
            role: LifecycleRole::Rover,
            entity_id: "rover-kiwi".into(),
            authority,
            transition_id: None,
            event_type: PowerEventType::TransitionRequested,
            reason_code: None,
            detail: None,
            context: Default::default(),
            occurred_at_ms: 100,
        },
        status: Some(PowerStatus {
            protocol_version: POWER_PROTOCOL_VERSION,
            role: LifecycleRole::Rover,
            entity_id: "rover-kiwi".into(),
            authority,
            policy: PowerPolicy::Awake,
            requested_profile: PowerProfile::NormalRover,
            effective_profile: PowerProfile::Dormant,
            state: PowerState::Waking,
            transition_id: None,
            reason_code: None,
            detail: None,
            updated_at_ms: 100,
        }),
    }
}

fn config(dir: &TempDir) -> JournalConfig {
    JournalConfig {
        directory: dir.path().into(),
        max_bytes: 8 * 1024,
        max_records: 10,
        wake_reserve_bytes: 2 * 1024,
        wake_reserve_records: 1,
    }
}

#[test]
fn recovers_a_torn_final_record_and_preserves_epoch_high_water() {
    let dir = TempDir::new().unwrap();
    let mut journal = EventJournal::open(config(&dir)).unwrap();
    journal
        .append(record(9), JournalAppendClass::Normal)
        .unwrap();
    drop(journal);
    let path = dir.path().join("power-events.log");
    let size = std::fs::metadata(&path).unwrap().len();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"RPCJ\x01")
        .unwrap();
    let recovered = EventJournal::open(config(&dir)).unwrap();
    assert_eq!(recovered.pending().count(), 1);
    assert_eq!(recovered.next_epoch(), 10);
    assert_eq!(std::fs::metadata(path).unwrap().len(), size);
}

#[test]
fn discards_a_corrupt_final_record_and_reports_it() {
    let dir = TempDir::new().unwrap();
    let mut journal = EventJournal::open(config(&dir)).unwrap();
    journal
        .append(record(1), JournalAppendClass::Normal)
        .unwrap();
    drop(journal);
    let path = dir.path().join("power-events.log");
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0xFF;
    std::fs::write(&path, bytes).unwrap();
    let recovered = EventJournal::open(config(&dir)).unwrap();
    assert!(recovered.health().recovered_torn_tail);
    assert_eq!(recovered.pending().count(), 0);
}

#[test]
fn reserved_capacity_keeps_a_wake_record_admissible() {
    let dir = TempDir::new().unwrap();
    let mut journal = EventJournal::open(config(&dir)).unwrap();
    while journal
        .append(record(1), JournalAppendClass::Normal)
        .is_ok()
    {}
    assert!(journal.capacity().unsafe_for_sleep);
    journal
        .append(record(1), JournalAppendClass::WakeToSafer)
        .unwrap();
}

#[test]
fn unacknowledged_outbox_records_survive_outage_and_interrupted_compaction() {
    let dir = TempDir::new().unwrap();
    let mut journal = EventJournal::open(config(&dir)).unwrap();
    let outbox_record = record(4);
    let event_id = outbox_record.event.event_id.clone();
    journal
        .append(outbox_record, JournalAppendClass::Normal)
        .unwrap();
    std::fs::write(dir.path().join("power-events.compacting"), b"interrupted").unwrap();
    drop(journal);
    let mut recovered = EventJournal::open(config(&dir)).unwrap();
    assert_eq!(recovered.pending().count(), 1);
    recovered.acknowledge(&event_id).unwrap();
    recovered.compact().unwrap();
    drop(recovered);
    let compacted = EventJournal::open(config(&dir)).unwrap();
    assert_eq!(compacted.pending().count(), 0);
    assert_eq!(compacted.next_epoch(), 5);
}

#[test]
fn transition_intent_is_durable_before_effects_are_returned() {
    let dir = TempDir::new().unwrap();
    let mut config = CoordinatorConfig::for_test(LifecycleRole::Rover, "rover-kiwi");
    config.journal_dir = dir.path().display().to_string();
    let mut coordinator = DurablePowerCoordinator::open(config, 100).unwrap();
    let effects = coordinator
        .tick(CoordinatorTime {
            wall_ms: 101,
            monotonic_ms: 1,
        })
        .unwrap();
    assert!(effects.transition.is_some());
    assert!(coordinator
        .pending_records()
        .iter()
        .any(|item| item.event.event_type == PowerEventType::TransitionRequested));
}

#[test]
fn restart_appends_a_fresh_awake_intent_with_a_new_epoch() {
    let dir = TempDir::new().unwrap();
    let mut config = CoordinatorConfig::for_test(LifecycleRole::Rover, "rover-kiwi");
    config.journal_dir = dir.path().display().to_string();
    drop(DurablePowerCoordinator::open(config.clone(), 100).unwrap());
    let restarted = DurablePowerCoordinator::open(config, 200).unwrap();
    let records = restarted.pending_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].event.authority.epoch, 2);
}

#[test]
fn durable_exact_command_replay_does_not_append_a_second_journal_intent() {
    let dir = TempDir::new().unwrap();
    let mut config = CoordinatorConfig::for_test(LifecycleRole::Rover, "rover-kiwi");
    config.journal_dir = dir.path().display().to_string();
    let mut coordinator = DurablePowerCoordinator::open(config, 100).unwrap();
    let command = PowerCommand {
        protocol_version: POWER_PROTOCOL_VERSION,
        command_id: "f4f3e2d1-c0b9-48a7-9615-141312111014".into(),
        role: LifecycleRole::Rover,
        entity_id: "rover-kiwi".into(),
        authority: PowerAuthority {
            epoch: 1,
            sequence: 1,
        },
        action: PowerCommandAction::SetPolicy {
            policy: PowerPolicy::Auto,
        },
        issued_at_ms: 100,
        not_before_ms: 100,
        expires_at_ms: 200,
        detail: None,
    };
    let time = CoordinatorTime {
        wall_ms: 101,
        monotonic_ms: 1,
    };
    let accepted = coordinator.apply_command(command.clone(), time);
    let records_after_first = coordinator.pending_records().len();
    assert!(accepted.accepted);
    assert_eq!(coordinator.apply_command(command, time), accepted);
    assert_eq!(coordinator.pending_records().len(), records_after_first);
}

#[test]
fn wake_policy_command_uses_reserved_capacity_and_keeps_auditable_context() {
    let dir = TempDir::new().unwrap();
    let mut config = CoordinatorConfig::for_test(LifecycleRole::Rover, "rover-kiwi");
    config.journal_dir = dir.path().display().to_string();
    config.journal_max_records = 4;
    config.journal_wake_reserve_records = 2;
    let mut coordinator = DurablePowerCoordinator::open(config, 100).unwrap();
    let time = CoordinatorTime {
        wall_ms: 101,
        monotonic_ms: 1,
    };
    let auto = PowerCommand {
        protocol_version: POWER_PROTOCOL_VERSION,
        command_id: "f4f3e2d1-c0b9-48a7-9615-141312111015".into(),
        role: LifecycleRole::Rover,
        entity_id: "rover-kiwi".into(),
        authority: PowerAuthority {
            epoch: 1,
            sequence: 1,
        },
        action: PowerCommandAction::SetPolicy {
            policy: PowerPolicy::Auto,
        },
        issued_at_ms: 100,
        not_before_ms: 100,
        expires_at_ms: 200,
        detail: None,
    };
    assert!(!coordinator.apply_command(auto.clone(), time).accepted);

    let mut wake = auto;
    wake.command_id = "f4f3e2d1-c0b9-48a7-9615-141312111016".into();
    wake.action = PowerCommandAction::SetPolicy {
        policy: PowerPolicy::Awake,
    };
    assert!(coordinator.apply_command(wake, time).accepted);
    let record = coordinator
        .pending_records()
        .into_iter()
        .find(|record| {
            record.intent == JournalIntent::CommandApplied
                && record.event.context.policy == Some(PowerPolicy::Awake)
        })
        .expect("wake command intent is journaled from reserved capacity");
    assert!(record.status.is_some());
    assert!(record.event.context.command_id.is_some());
}
