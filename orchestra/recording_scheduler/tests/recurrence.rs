use chrono::{TimeZone, Timelike, Utc};
use chrono_tz::America::New_York;
use recording_scheduler::{
    clock::FakeClock,
    mongo_repository::Stored,
    recurrence::{candidates, resolve_local},
    runtime::SchedulerRuntime,
};
use robo_rover_lib::{
    occurrence_id, scheduled_start_request_id, DstResolution, IsoWeekday, RecordingLocalStart,
    RecordingSchedule, RecordingScheduleDefinition, RecordingScheduleRecurrence,
};

const HOUR_MS: i64 = 60 * 60 * 1_000;

#[test]
fn daily_and_weekly_candidates_are_ordered_and_unique() {
    let daily_schedule = schedule(
        "00000000-0000-0000-0000-000000000001",
        daily("2026-01-30", "09:30"),
    );
    let starts = candidates(
        &daily_schedule,
        epoch(2026, 1, 30, 0, 0),
        epoch(2026, 2, 2, 23, 0),
    )
    .unwrap();
    assert_eq!(starts.len(), 4);
    assert!(starts
        .windows(2)
        .all(|pair| pair[0].start_ms < pair[1].start_ms));

    let weekly = schedule(
        "00000000-0000-0000-0000-000000000002",
        weekly("2026-02-01", "09:30"),
    );
    let starts = candidates(&weekly, epoch(2026, 2, 1, 0, 0), epoch(2026, 2, 10, 0, 0)).unwrap();
    assert_eq!(starts.len(), 3);
}

#[test]
fn recurrence_crosses_leap_day_and_year_boundary_without_extra_occurrences() {
    let leap_daily = schedule(
        "00000000-0000-0000-0000-000000000007",
        daily("2028-02-28", "09:30"),
    );
    let leap_starts = candidates(
        &leap_daily,
        epoch(2028, 2, 28, 0, 0),
        epoch(2028, 3, 1, 23, 0),
    )
    .unwrap();
    assert_eq!(
        leap_starts
            .iter()
            .map(|start| start.start_ms)
            .collect::<Vec<_>>(),
        vec![
            epoch(2028, 2, 28, 9, 30),
            epoch(2028, 2, 29, 9, 30),
            epoch(2028, 3, 1, 9, 30),
        ],
        "daily UTC recurrence must materialize each leap-day boundary exactly once"
    );

    let year_weekly = schedule(
        "00000000-0000-0000-0000-000000000008",
        weekly("2026-12-30", "09:30"),
    );
    let year_starts = candidates(
        &year_weekly,
        epoch(2026, 12, 30, 0, 0),
        epoch(2027, 1, 7, 23, 0),
    )
    .unwrap();
    assert_eq!(
        year_starts
            .iter()
            .map(|start| start.start_ms)
            .collect::<Vec<_>>(),
        vec![
            epoch(2026, 12, 30, 9, 30),
            epoch(2027, 1, 4, 9, 30),
            epoch(2027, 1, 6, 9, 30),
        ],
        "Monday/Wednesday UTC recurrence must retain its exact year-boundary dates"
    );
}

#[test]
fn dst_gap_shifts_forward_and_fold_uses_earlier_instant() {
    let gap = resolve_local(
        New_York,
        chrono::NaiveDate::from_ymd_opt(2026, 3, 8)
            .unwrap()
            .and_hms_opt(2, 30, 0)
            .unwrap(),
    )
    .unwrap();
    let gap_local = chrono::DateTime::from_timestamp_millis(gap.start_ms)
        .unwrap()
        .with_timezone(&New_York);
    assert_eq!(gap.resolution, DstResolution::GapShifted);
    assert_eq!((gap_local.hour(), gap_local.minute()), (3, 0));

    let fold = resolve_local(
        New_York,
        chrono::NaiveDate::from_ymd_opt(2026, 11, 1)
            .unwrap()
            .and_hms_opt(1, 30, 0)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(fold.resolution, DstResolution::FoldEarlier);
    assert_eq!(
        fold.start_ms,
        New_York
            .with_ymd_and_hms(2026, 11, 1, 1, 30, 0)
            .earliest()
            .unwrap()
            .with_timezone(&Utc)
            .timestamp_millis()
    );
}

#[test]
fn occurrence_and_start_ids_survive_restart() {
    let first = occurrence_id(
        "00000000-0000-0000-0000-000000000003",
        4,
        epoch(2026, 2, 28, 9, 30),
    )
    .unwrap();
    assert_eq!(
        first,
        occurrence_id(
            "00000000-0000-0000-0000-000000000003",
            4,
            epoch(2026, 2, 28, 9, 30)
        )
        .unwrap()
    );
    assert_eq!(
        scheduled_start_request_id(&first).unwrap(),
        scheduled_start_request_id(&first).unwrap()
    );
}

#[test]
fn overlapping_occurrences_share_one_group_after_reconciliation() {
    let now = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now);
    let mut runtime = SchedulerRuntime::new(clock.clone());
    let first = schedule(
        "00000000-0000-0000-0000-000000000004",
        one_time("2026-01-01", "01:00"),
    );
    let second = schedule(
        "00000000-0000-0000-0000-000000000005",
        one_time("2026-01-01", "01:30"),
    );
    runtime.materialize(&first, now + 3 * HOUR_MS).unwrap();
    runtime.materialize(&second, now + 3 * HOUR_MS).unwrap();
    assert_eq!(runtime.groups.len(), 1);
    let group = runtime.groups.values().next().unwrap();
    assert!(group.owner_ids.is_empty());
    assert_eq!(group.end_ms, now + (150 * 60 * 1_000));

    clock.advance_ms(HOUR_MS);
    assert!(runtime.due().is_empty());
    runtime.complete_reconciliation();
    let due = runtime.due();
    assert_eq!(due.len(), 1);
    assert!(runtime.begin_start(&due[0]).unwrap().intent_id.is_some());
}

#[test]
fn persisted_group_keeps_its_directory_and_owner_lifecycle_after_restart() {
    let now = epoch(2026, 1, 1, 0, 0);
    let clock = FakeClock::new(now);
    let schedule = schedule(
        "00000000-0000-0000-0000-000000000006",
        one_time("2026-01-01", "01:00"),
    );
    let mut before = SchedulerRuntime::new(clock.clone());
    before.materialize(&schedule, now + 3 * HOUR_MS).unwrap();
    let occurrences = before
        .occurrences
        .values()
        .cloned()
        .map(|value| Stored { value, revision: 1 });
    let groups = before
        .groups
        .values()
        .cloned()
        .map(|value| Stored { value, revision: 1 });
    let mut after = SchedulerRuntime::from_persisted(clock.clone(), occurrences, groups).unwrap();
    let group = after.groups.values().next().unwrap();
    assert_eq!(group.relative_directory, "scheduled");
    assert!(group.owner_ids.is_empty());

    clock.advance_ms(HOUR_MS);
    after.complete_reconciliation();
    let due = after.due().pop().unwrap();
    assert!(after.begin_start(&due).unwrap().intent_id.is_some());
    let group = after.groups.values().next().unwrap();
    assert_eq!(group.owner_ids.len(), 1);
}

fn schedule(id: &str, recurrence: RecordingScheduleRecurrence) -> RecordingSchedule {
    RecordingSchedule {
        schedule_id: id.into(),
        revision: 1,
        definition: RecordingScheduleDefinition {
            entity_id: "kiwi-1".into(),
            title: "test".into(),
            enabled: true,
            recurrence,
            duration_ms: HOUR_MS,
            relative_directory_template: "scheduled".into(),
        },
        created_at_ms: 0,
        created_by: "user".into(),
        updated_at_ms: 0,
        updated_by: "user".into(),
    }
}

fn one_time(date: &str, time: &str) -> RecordingScheduleRecurrence {
    let mut recurrence = daily(date, time);
    if let RecordingScheduleRecurrence::Daily { local_start } = recurrence {
        recurrence = RecordingScheduleRecurrence::OneTime { local_start };
    }
    recurrence
}

fn daily(date: &str, time: &str) -> RecordingScheduleRecurrence {
    RecordingScheduleRecurrence::Daily {
        local_start: local(date, time),
    }
}

fn weekly(date: &str, time: &str) -> RecordingScheduleRecurrence {
    RecordingScheduleRecurrence::Weekly {
        local_start: local(date, time),
        weekdays: vec![IsoWeekday::Monday, IsoWeekday::Wednesday],
    }
}

fn local(date: &str, time: &str) -> RecordingLocalStart {
    RecordingLocalStart {
        date: date.into(),
        time: time.into(),
        timezone: "UTC".into(),
    }
}

fn epoch(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> i64 {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
        .unwrap()
        .timestamp_millis()
}
