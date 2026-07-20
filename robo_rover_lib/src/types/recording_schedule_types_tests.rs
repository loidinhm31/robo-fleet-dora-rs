use super::*;
use serde_json::{json, Value};

const NOW_MS: i64 = 1_784_500_000_000;
const FIXTURE: &str = include_str!("../../tests/fixtures/recording-schedule-v1.json");

fn fixture(name: &str) -> Value {
    serde_json::from_str::<Value>(FIXTURE).unwrap()[name].clone()
}

#[test]
fn schedule_fixture_round_trips_all_public_responses() {
    let command: RecordingScheduleCommand =
        serde_json::from_value(fixture("accepted_command")).unwrap();
    command
        .validate_at(NOW_MS, RecordingScheduleValidationLimits::default())
        .unwrap();
    assert_eq!(
        serde_json::to_value(&command).unwrap(),
        fixture("accepted_command")
    );

    for name in ["accepted_result", "rejected_result", "conflict_result"] {
        let result: RecordingScheduleCommandResult = serde_json::from_value(fixture(name)).unwrap();
        result.validate().unwrap();
        assert_eq!(serde_json::to_value(result).unwrap(), fixture(name));
    }

    let snapshot: RecordingScheduleSnapshot = serde_json::from_value(fixture("snapshot")).unwrap();
    snapshot.validate().unwrap();
    assert_eq!(serde_json::to_value(snapshot).unwrap(), fixture("snapshot"));
}

#[test]
fn schedule_contract_rejects_protocol_identity_timezone_path_and_revision_errors() {
    let mut command: RecordingScheduleCommand =
        serde_json::from_value(fixture("accepted_command")).unwrap();
    command.protocol_version = 2;
    assert!(command.validate_at(NOW_MS, Default::default()).is_err());
    command.protocol_version = 1;
    command.request_id = "invalid".into();
    assert!(command.validate_at(NOW_MS, Default::default()).is_err());
    command.request_id = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11".into();
    if let RecordingScheduleAction::Create { schedule } = &mut command.action {
        match &mut schedule.recurrence {
            RecordingScheduleRecurrence::OneTime { local_start }
            | RecordingScheduleRecurrence::Daily { local_start }
            | RecordingScheduleRecurrence::Weekly { local_start, .. } => {
                local_start.timezone = "UTC+07".into();
            }
        }
    }
    assert!(command.validate_at(NOW_MS, Default::default()).is_err());
    if let RecordingScheduleAction::Create { schedule } = &mut command.action {
        match &mut schedule.recurrence {
            RecordingScheduleRecurrence::OneTime { local_start }
            | RecordingScheduleRecurrence::Daily { local_start }
            | RecordingScheduleRecurrence::Weekly { local_start, .. } => {
                local_start.timezone = "Asia/Ho_Chi_Minh".into();
            }
        }
        schedule.relative_directory_template = "../escape".into();
    }
    assert!(command.validate_at(NOW_MS, Default::default()).is_err());

    let update = RecordingScheduleCommand {
        protocol_version: 1,
        request_id: "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11".into(),
        action: RecordingScheduleAction::Delete {
            schedule_id: "550e8400-e29b-41d4-a716-446655440000".into(),
            expected_revision: 0,
        },
    };
    assert!(update.validate_at(NOW_MS, Default::default()).is_err());
}

#[test]
fn browser_command_cannot_carry_audit_or_recorder_fields() {
    let command = fixture("accepted_command");
    assert!(command.get("audit_actor").is_none());
    assert!(command.get("occurrence_id").is_none());
    assert!(command.get("recording_id").is_none());
    assert!(serde_json::from_value::<RecordingScheduleCommand>(json!({
        "protocol_version": 1,
        "request_id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "action": "create",
        "audit_actor": "spoofed",
        "schedule": command["schedule"].clone()
    }))
    .is_err());
}

#[test]
fn one_time_schedule_accepts_only_local_wall_clock_intent() {
    let command = json!({
        "protocol_version": 1,
        "request_id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "action": "create",
        "schedule": {
            "entity_id": "rover-a",
            "title": "One-time patrol",
            "enabled": true,
            "recurrence": {
                "kind": "one_time",
                "local_start": { "date": "2026-07-21", "time": "09:30", "timezone": "Asia/Ho_Chi_Minh" },
                "start_at_ms": 1
            },
            "duration_ms": 60000,
            "relative_directory_template": "schedules/one-time"
        }
    });
    assert!(serde_json::from_value::<RecordingScheduleCommand>(command).is_err());
}

#[test]
fn occurrence_identity_is_stable_and_terminal_ttl_is_enforced() {
    let id = occurrence_id("550e8400-e29b-41d4-a716-446655440000", 1, NOW_MS).unwrap();
    assert_eq!(
        id,
        occurrence_id("550e8400-e29b-41d4-a716-446655440000", 1, NOW_MS).unwrap()
    );
    assert_eq!(
        scheduled_group_id("rover-a", NOW_MS).unwrap(),
        scheduled_group_id("rover-a", NOW_MS).unwrap()
    );
    assert_ne!(
        scheduled_group_id("rover-a", NOW_MS).unwrap(),
        scheduled_group_id("rover-b", NOW_MS).unwrap()
    );
    let request_id = scheduled_start_request_id(&id).unwrap();
    let occurrence = RecordingOccurrence {
        occurrence_id: id,
        schedule_id: "550e8400-e29b-41d4-a716-446655440000".into(),
        schedule_revision: 1,
        entity_id: "rover-a".into(),
        planned_start_ms: NOW_MS,
        planned_end_ms: NOW_MS + 60_000,
        dst_resolution: DstResolution::Exact,
        state: RecordingOccurrenceState::Completed,
        retry_count: 0,
        next_retry_at_ms: None,
        group_id: None,
        start_request_id: request_id,
        attempts: Vec::new(),
        last_error: None,
        suppressed_by_manual: false,
        created_at_ms: NOW_MS,
        updated_at_ms: NOW_MS,
        terminal_at_ms: Some(NOW_MS),
        expires_at_ms: Some(NOW_MS + TERMINAL_OCCURRENCE_RETENTION_MS),
    };
    occurrence.validate().unwrap();
    assert!(RecordingOccurrence {
        expires_at_ms: None,
        ..occurrence.clone()
    }
    .validate()
    .is_err());
    assert!(RecordingOccurrence {
        expires_at_ms: Some(NOW_MS + 1),
        ..occurrence
    }
    .validate()
    .is_err());
}

#[test]
fn one_time_validation_rejects_a_same_day_past_wall_clock_start() {
    let command: RecordingScheduleCommand = serde_json::from_value(json!({
        "protocol_version": 1,
        "request_id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "action": "create",
        "schedule": {
            "entity_id": "rover-a",
            "title": "Past patrol",
            "enabled": true,
            "recurrence": { "kind": "one_time", "local_start": { "date": "2026-07-21", "time": "00:01", "timezone": "Asia/Ho_Chi_Minh" } },
            "duration_ms": 60000,
            "relative_directory_template": "schedules/past"
        }
    })).unwrap();
    let now_ms = chrono::DateTime::parse_from_rfc3339("2026-07-21T23:59:00+07:00")
        .unwrap()
        .timestamp_millis();
    assert!(command.validate_at(now_ms, Default::default()).is_err());
}
