use bson::Bson;
use recording_scheduler::{
    mongo_documents::occurrence_document,
    state_machine::{retry_at, transition},
};
use robo_rover_lib::{
    DstResolution, RecordingOccurrence, RecordingOccurrenceState, RecordingScheduleReasonCode,
    TERMINAL_OCCURRENCE_RETENTION_MS,
};

#[test]
fn terminal_document_has_ttl_date_and_signed_epoch_milliseconds() {
    let mut occurrence = occurrence(10_000);
    assert!(transition(
        &mut occurrence,
        RecordingOccurrenceState::Due,
        10_001,
        None
    ));
    assert!(transition(
        &mut occurrence,
        RecordingOccurrenceState::Missed,
        10_002,
        Some(RecordingScheduleReasonCode::Unavailable)
    ));
    let document = occurrence_document(&occurrence).unwrap();
    assert!(matches!(
        document.get("planned_start_ms"),
        Some(Bson::Int64(10_000))
    ));
    assert!(
        matches!(document.get("expire_at"), Some(Bson::DateTime(value)) if value.timestamp_millis() == 10_002 + TERMINAL_OCCURRENCE_RETENTION_MS)
    );
}

#[test]
fn retry_cadence_is_exact_and_stops_at_window_end() {
    let mut occurrence = occurrence(1_000_000);
    occurrence.planned_end_ms = 2_000_000;
    let cadence = (0..6)
        .map(|_| retry_at(&mut occurrence, 100_000).unwrap() - 100_000)
        .collect::<Vec<_>>();
    assert_eq!(cadence, vec![1_000, 2_000, 4_000, 8_000, 16_000, 30_000]);
    occurrence.planned_end_ms = 100_001;
    assert!(retry_at(&mut occurrence, 100_000).is_none());
}

fn occurrence(start_ms: i64) -> RecordingOccurrence {
    RecordingOccurrence {
        occurrence_id: "00000000-0000-0000-0000-000000000100".into(),
        schedule_id: "00000000-0000-0000-0000-000000000101".into(),
        schedule_revision: 1,
        entity_id: "kiwi-1".into(),
        planned_start_ms: start_ms,
        planned_end_ms: start_ms + 60_000,
        dst_resolution: DstResolution::Exact,
        state: RecordingOccurrenceState::Planned,
        retry_count: 0,
        next_retry_at_ms: None,
        group_id: None,
        start_request_id: "00000000-0000-0000-0000-000000000102".into(),
        attempts: Vec::new(),
        last_error: None,
        suppressed_by_manual: false,
        created_at_ms: start_ms - 1,
        updated_at_ms: start_ms - 1,
        terminal_at_ms: None,
        expires_at_ms: None,
    }
}
