use chrono::{TimeZone, Utc};
use recording_scheduler::{
    clock::{Clock, FakeClock},
    mongo_repository::OutboxRecord,
    mongo_repository::Stored,
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

#[test]
fn bridging_overlap_merges_every_group_and_keeps_earliest_directory() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let mut runtime = SchedulerRuntime::new(clock);
    let first = schedule_at("00000000-0000-0000-0000-000000000210", "01:00", "first");
    let second = schedule_at("00000000-0000-0000-0000-000000000211", "03:00", "second");
    let mut bridge = schedule_at("00000000-0000-0000-0000-000000000212", "01:30", "bridge");
    bridge.definition.duration_ms = 2 * 60 * 60 * 1_000;
    runtime
        .materialize(&first, now_ms + 5 * 60 * 60 * 1_000)
        .unwrap();
    runtime
        .materialize(&second, now_ms + 5 * 60 * 60 * 1_000)
        .unwrap();
    assert_eq!(runtime.groups.len(), 2);

    runtime
        .materialize(&bridge, now_ms + 5 * 60 * 60 * 1_000)
        .unwrap();

    assert_eq!(runtime.groups.len(), 1);
    let group = runtime.groups.values().next().unwrap();
    assert_eq!(group.start_ms, now_ms + 60 * 60 * 1_000);
    assert_eq!(group.relative_directory, "first");
    assert!(runtime
        .occurrences
        .values()
        .all(|occurrence| occurrence.group_id.as_deref() == Some(&group.group_id)));
}

#[test]
fn bridge_preserves_the_live_group_session_and_one_eventual_release() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let mut runtime = SchedulerRuntime::new(clock.clone());
    let mut first = schedule_at("00000000-0000-0000-0000-000000000213", "01:00", "first");
    first.definition.duration_ms = 2 * 60 * 60 * 1_000;
    runtime
        .materialize(&first, now_ms + 6 * 60 * 60 * 1_000)
        .unwrap();
    runtime.complete_reconciliation();
    clock.advance_ms(60 * 60 * 1_000);
    let first_id = runtime.due().pop().unwrap();
    let acquire = runtime.begin_start(&first_id).unwrap();
    assert!(runtime.apply_feedback(feedback(
        acquire.intent_id.unwrap(),
        first_id,
        acquire.generation,
        true,
        true,
        false,
        Some("recording-1"),
    )));
    let before = runtime.groups.values().next().unwrap().clone();

    clock.advance_ms(30 * 60 * 1_000);
    let second = schedule_at("00000000-0000-0000-0000-000000000214", "04:00", "second");
    let mut bridge = schedule_at("00000000-0000-0000-0000-000000000215", "02:00", "bridge");
    bridge.definition.duration_ms = 3 * 60 * 60 * 1_000;
    runtime
        .materialize(&second, now_ms + 6 * 60 * 60 * 1_000)
        .unwrap();
    runtime
        .materialize(&bridge, now_ms + 6 * 60 * 60 * 1_000)
        .unwrap();

    assert_eq!(runtime.groups.len(), 1);
    let merged = runtime.groups.values().next().unwrap();
    assert_eq!(merged.group_id, before.group_id);
    assert_eq!(merged.start_request_id, before.start_request_id);
    assert_eq!(merged.recording_id, before.recording_id);
    assert_eq!(merged.relative_directory, before.relative_directory);
    assert_eq!(merged.owner_ids, before.owner_ids);

    clock.advance_ms(30 * 60 * 1_000);
    let bridge_id = runtime.due().pop().unwrap();
    assert!(runtime.begin_start(&bridge_id).unwrap().intent_id.is_none());
    clock.advance_ms(60 * 60 * 1_000);
    let first_stop = runtime.due_stops().pop().unwrap();
    assert!(runtime.begin_stop(&first_stop).unwrap().intent_id.is_none());
    clock.advance_ms(60 * 60 * 1_000);
    let second_id = runtime.due().pop().unwrap();
    assert!(runtime.begin_start(&second_id).unwrap().intent_id.is_none());
    clock.advance_ms(60 * 60 * 1_000);
    let release_count = runtime
        .due_stops()
        .into_iter()
        .filter_map(|id| runtime.begin_stop(&id))
        .filter(|transition| transition.intent_id.is_some())
        .count();
    assert_eq!(release_count, 1);
}

#[test]
fn outbox_recovers_a_transition_written_before_occurrence_and_group() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let schedule = schedule();
    let mut before_crash = SchedulerRuntime::new(clock.clone());
    before_crash
        .materialize(&schedule, now_ms + 2 * 60 * 60 * 1_000)
        .unwrap();
    before_crash.complete_reconciliation();
    clock.advance_ms(60 * 60 * 1_000);
    let occurrence_id = before_crash.due().pop().unwrap();
    let transition = before_crash.begin_start(&occurrence_id).unwrap();
    let occurrence = before_crash.occurrences[&occurrence_id].clone();
    let group = before_crash.groups[occurrence.group_id.as_ref().unwrap()].clone();
    let intent = intent(&occurrence, &group, transition.intent_id.unwrap());

    let mut recovered = SchedulerRuntime::new(clock);
    recovered
        .materialize(&schedule, now_ms + 2 * 60 * 60 * 1_000)
        .unwrap();
    let recovery = recovered.recover_outbox(&[OutboxRecord {
        intent: intent.clone(),
        occurrence,
        group,
    }]);

    assert_eq!(recovery.replay, vec![intent]);
    assert!(recovery.acknowledge.is_empty());
    assert_eq!(
        recovered.occurrences[&occurrence_id].state,
        RecordingOccurrenceState::StartPending
    );
}

#[test]
fn outbox_acknowledges_an_intent_already_confirmed_by_reconciliation() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let schedule = schedule();
    let mut runtime = SchedulerRuntime::new(clock.clone());
    runtime
        .materialize(&schedule, now_ms + 2 * 60 * 60 * 1_000)
        .unwrap();
    runtime.complete_reconciliation();
    clock.advance_ms(60 * 60 * 1_000);
    let occurrence_id = runtime.due().pop().unwrap();
    let transition = runtime.begin_start(&occurrence_id).unwrap();
    let occurrence = runtime.occurrences[&occurrence_id].clone();
    let group = runtime.groups[occurrence.group_id.as_ref().unwrap()].clone();
    let intent = intent(&occurrence, &group, transition.intent_id.unwrap());
    let record = OutboxRecord {
        intent: intent.clone(),
        occurrence,
        group,
    };
    assert!(runtime.apply_feedback(feedback(
        intent.intent_id.clone(),
        occurrence_id,
        intent.generation,
        true,
        true,
        false,
        Some("recording-1")
    )));

    let recovery = runtime.recover_outbox(&[record]);
    assert!(recovery.replay.is_empty());
    assert_eq!(recovery.acknowledge, vec![intent.intent_id]);
}

#[test]
fn outbox_replays_a_release_when_active_feedback_has_no_recording_id() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let mut runtime = SchedulerRuntime::new(clock.clone());
    runtime
        .materialize(&schedule(), now_ms + 2 * 60 * 60 * 1_000)
        .unwrap();
    runtime.complete_reconciliation();
    clock.advance_ms(60 * 60 * 1_000);
    let occurrence_id = runtime.due().pop().unwrap();
    let acquire = runtime.begin_start(&occurrence_id).unwrap();
    assert!(runtime.apply_feedback(feedback(
        acquire.intent_id.unwrap(),
        occurrence_id.clone(),
        acquire.generation,
        true,
        true,
        false,
        None,
    )));
    clock.advance_ms(60 * 60 * 1_000);
    let stop_id = runtime.due_stops().pop().unwrap();
    let release = runtime.begin_stop(&stop_id).unwrap();
    let occurrence = runtime.occurrences[&stop_id].clone();
    let group = runtime.groups[occurrence.group_id.as_ref().unwrap()].clone();
    let intent = intent_with_action(
        &occurrence,
        &group,
        release.intent_id.unwrap(),
        ScheduledRecordingIntentAction::Release,
    );

    let recovery = runtime.recover_outbox(&[OutboxRecord {
        intent: intent.clone(),
        occurrence,
        group,
    }]);
    assert_eq!(recovery.replay, vec![intent]);
    assert!(recovery.acknowledge.is_empty());
}

#[test]
fn outbox_acknowledges_retry_feedback_persisted_before_acknowledgement() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let mut runtime = SchedulerRuntime::new(clock.clone());
    runtime
        .materialize(&schedule(), now_ms + 2 * 60 * 60 * 1_000)
        .unwrap();
    runtime.complete_reconciliation();
    clock.advance_ms(60 * 60 * 1_000);
    let occurrence_id = runtime.due().pop().unwrap();
    let transition = runtime.begin_start(&occurrence_id).unwrap();
    let occurrence = runtime.occurrences[&occurrence_id].clone();
    let group = runtime.groups[occurrence.group_id.as_ref().unwrap()].clone();
    let intent = intent(&occurrence, &group, transition.intent_id.unwrap());
    assert!(runtime.apply_feedback(feedback(
        intent.intent_id.clone(),
        occurrence_id,
        intent.generation,
        false,
        false,
        true,
        None,
    )));

    let recovery = runtime.recover_outbox(&[OutboxRecord {
        intent: intent.clone(),
        occurrence,
        group,
    }]);
    assert!(recovery.replay.is_empty());
    assert_eq!(recovery.acknowledge, vec![intent.intent_id]);
}

#[test]
fn restart_rebuilds_missing_group_with_a_persistable_revision() {
    let now_ms = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now_ms);
    let mut before_crash = SchedulerRuntime::new(clock.clone());
    before_crash
        .materialize(&schedule(), now_ms + 2 * 60 * 60 * 1_000)
        .unwrap();
    let occurrence = before_crash.occurrences.values().next().unwrap().clone();
    let recovered = SchedulerRuntime::from_persisted(
        clock,
        vec![Stored {
            value: occurrence,
            revision: 3,
        }],
        Vec::<Stored<recording_scheduler::domain::RecordingGroup>>::new(),
    )
    .unwrap();

    assert_eq!(recovered.groups.len(), 1);
    assert_eq!(
        recovered
            .group_revisions
            .values()
            .copied()
            .collect::<Vec<_>>(),
        vec![0]
    );
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

fn intent(
    occurrence: &robo_rover_lib::RecordingOccurrence,
    group: &recording_scheduler::domain::RecordingGroup,
    intent_id: String,
) -> robo_rover_lib::ScheduledRecordingIntent {
    intent_with_action(
        occurrence,
        group,
        intent_id,
        ScheduledRecordingIntentAction::Acquire,
    )
}

fn intent_with_action(
    occurrence: &robo_rover_lib::RecordingOccurrence,
    group: &recording_scheduler::domain::RecordingGroup,
    intent_id: String,
    action: ScheduledRecordingIntentAction,
) -> robo_rover_lib::ScheduledRecordingIntent {
    robo_rover_lib::ScheduledRecordingIntent {
        intent_id,
        occurrence_id: occurrence.occurrence_id.clone(),
        group_id: group.group_id.clone(),
        generation: group.generation,
        entity_id: occurrence.entity_id.clone(),
        start_request_id: group.start_request_id.clone(),
        planned_start_ms: occurrence.planned_start_ms,
        planned_end_ms: occurrence.planned_end_ms,
        relative_directory: group.relative_directory.clone(),
        action,
    }
}

fn schedule() -> RecordingSchedule {
    schedule_at("00000000-0000-0000-0000-000000000200", "01:00", "scheduled")
}

fn schedule_at(id: &str, time: &str, directory: &str) -> RecordingSchedule {
    RecordingSchedule {
        schedule_id: id.into(),
        revision: 1,
        definition: RecordingScheduleDefinition {
            entity_id: "kiwi-1".into(),
            title: "test".into(),
            enabled: true,
            recurrence: RecordingScheduleRecurrence::OneTime {
                local_start: RecordingLocalStart {
                    date: "2026-01-01".into(),
                    time: time.into(),
                    timezone: "UTC".into(),
                },
            },
            duration_ms: 60 * 60 * 1_000,
            relative_directory_template: directory.into(),
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
