use chrono::{TimeZone, Utc};
use recording_scheduler::{
    clock::{Clock, FakeClock},
    runtime::SchedulerRuntime,
};
use robo_rover_lib::{
    scheduled_intent_id, RecordingCoordinatorFeedback, RecordingLocalStart,
    RecordingOccurrenceState, RecordingSchedule, RecordingScheduleDefinition,
    RecordingScheduleReasonCode, RecordingScheduleRecurrence, ScheduledRecordingIntentAction,
};

#[test]
fn reconciliation_blocks_then_deduplicates_transient_feedback() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let mut runtime = SchedulerRuntime::new(clock.clone());
    runtime
        .materialize(&schedule(), now_ms + 2 * 60 * 60 * 1_000)
        .unwrap();
    clock.advance_ms(60 * 60 * 1_000);
    assert!(runtime.due().is_empty());

    runtime.complete_reconciliation();
    let due = runtime.due().pop().unwrap();
    let transition = runtime.begin_start(&due).unwrap();
    let generation = transition.generation;
    assert!(transition.intent_id.is_some());
    let feedback = RecordingCoordinatorFeedback {
        intent_id: scheduled_intent_id(&due, generation, ScheduledRecordingIntentAction::Acquire)
            .unwrap(),
        occurrence_id: due.clone(),
        generation,
        accepted: false,
        applied: false,
        retryable: true,
        group_id: runtime.occurrences[&due].group_id.clone(),
        recording_id: None,
        recorder_state: None,
        reason_code: Some(RecordingScheduleReasonCode::Unavailable),
        detail: None,
    };
    assert!(runtime.apply_feedback(feedback.clone()));
    assert_eq!(
        runtime.occurrences[&due].next_retry_at_ms,
        Some(clock.now_ms() + 1_000)
    );
    assert_eq!(runtime.occurrences[&due].retry_count, 1);
    assert!(!runtime.apply_feedback(feedback));
    assert_eq!(runtime.occurrences[&due].retry_count, 1);
    assert_eq!(
        runtime.occurrences[&due].state,
        RecordingOccurrenceState::StartPending
    );
    clock.advance_ms(1_000);
    let retried = runtime.due();
    assert_eq!(retried, vec![due.clone()]);
    let retried_transition = runtime.begin_start(&due).unwrap();
    let retry_generation = retried_transition.generation;
    assert!(
        retried_transition.intent_id.is_some(),
        "retry must re-emit acquire"
    );
    assert!(retry_generation > generation);
    assert_ne!(
        scheduled_intent_id(
            &due,
            retry_generation,
            ScheduledRecordingIntentAction::Acquire
        ),
        scheduled_intent_id(&due, generation, ScheduledRecordingIntentAction::Acquire)
    );
}

#[test]
fn overlap_uses_one_acquire_and_one_group_release_with_release_retry() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let mut runtime = SchedulerRuntime::new(clock.clone());
    let first = schedule();
    let mut second = schedule();
    second.schedule_id = "00000000-0000-0000-0000-000000000201".into();
    second.definition.recurrence = RecordingScheduleRecurrence::OneTime {
        local_start: RecordingLocalStart {
            date: "2026-01-01".into(),
            time: "01:30".into(),
            timezone: "UTC".into(),
        },
    };
    runtime
        .materialize(&first, now_ms + 4 * 60 * 60 * 1_000)
        .unwrap();
    runtime
        .materialize(&second, now_ms + 4 * 60 * 60 * 1_000)
        .unwrap();
    runtime.complete_reconciliation();

    clock.advance_ms(60 * 60 * 1_000);
    let first_id = runtime.due().pop().unwrap();
    let acquire = runtime.begin_start(&first_id).unwrap();
    let acquire_id = acquire.intent_id.clone().unwrap();

    clock.advance_ms(30 * 60 * 1_000);
    let second_id = runtime.due().pop().unwrap();
    assert!(runtime.begin_start(&second_id).unwrap().intent_id.is_none());
    assert_eq!(
        runtime
            .groups
            .values()
            .next()
            .unwrap()
            .pending_intent_id
            .as_deref(),
        Some(acquire_id.as_str())
    );

    assert!(runtime.apply_feedback(feedback(
        acquire_id,
        first_id.clone(),
        acquire.generation,
        true,
        true,
        false,
        Some("recording-1"),
    )));
    assert_eq!(
        runtime.occurrences[&first_id].state,
        RecordingOccurrenceState::Active
    );
    assert_eq!(
        runtime.occurrences[&second_id].state,
        RecordingOccurrenceState::Active
    );

    clock.advance_ms(30 * 60 * 1_000);
    let first_stop = runtime.due_stops().pop().unwrap();
    assert_eq!(first_stop, first_id);
    assert!(runtime.begin_stop(&first_stop).unwrap().intent_id.is_none());

    clock.advance_ms(30 * 60 * 1_000);
    let second_stop = runtime.due_stops().pop().unwrap();
    let release = runtime.begin_stop(&second_stop).unwrap();
    let release_id = release.intent_id.clone().unwrap();
    let group = runtime.groups.values().next().unwrap();
    assert_ne!(
        group.start_request_id,
        runtime.occurrences[&second_stop].start_request_id
    );

    assert!(runtime.apply_feedback(feedback(
        release_id,
        second_stop.clone(),
        release.generation,
        false,
        false,
        true,
        None,
    )));
    clock.advance_ms(1_000);
    assert_eq!(runtime.due_stops(), vec![second_stop.clone()]);
    assert!(runtime
        .begin_stop(&second_stop)
        .unwrap()
        .intent_id
        .is_some());
}

fn feedback(
    intent_id: String,
    occurrence_id: String,
    generation: u64,
    accepted: bool,
    applied: bool,
    retryable: bool,
    recording_id: Option<&str>,
) -> RecordingCoordinatorFeedback {
    RecordingCoordinatorFeedback {
        intent_id,
        occurrence_id,
        generation,
        accepted,
        applied,
        retryable,
        group_id: None,
        recording_id: recording_id.map(str::to_owned),
        recorder_state: None,
        reason_code: retryable.then_some(RecordingScheduleReasonCode::Unavailable),
        detail: None,
    }
}

fn schedule() -> RecordingSchedule {
    RecordingSchedule {
        schedule_id: "00000000-0000-0000-0000-000000000200".into(),
        revision: 1,
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
            duration_ms: 60 * 60 * 1_000,
            relative_directory_template: "scheduled".into(),
        },
        created_at_ms: 0,
        created_by: "user".into(),
        updated_at_ms: 0,
        updated_by: "user".into(),
    }
}

fn epoch(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> i64 {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
        .unwrap()
        .timestamp_millis()
}
