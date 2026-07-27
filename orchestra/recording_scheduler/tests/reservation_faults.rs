use recording_scheduler::{
    clock::{Clock, FakeClock},
    domain::{is_transient_recorder_storage_failure, ReservationState},
    runtime::SchedulerRuntime,
};
use robo_rover_lib::{
    DstResolution, PowerAuthority, PowerCommandResult, PowerPolicy, PowerProfile, PowerState,
    PowerStatus, RecordingOccurrence, RecordingOccurrenceState, RecordingScheduleReasonCode,
    POWER_PROTOCOL_VERSION,
};

const START_MS: i64 = 100_000;
const END_MS: i64 = 120_000;

fn occurrence() -> RecordingOccurrence {
    RecordingOccurrence {
        occurrence_id: "00000000-0000-0000-0000-000000000101".into(),
        schedule_id: "00000000-0000-0000-0000-000000000102".into(),
        schedule_revision: 7,
        entity_id: "rover-kiwi".into(),
        planned_start_ms: START_MS,
        planned_end_ms: END_MS,
        dst_resolution: DstResolution::Exact,
        state: RecordingOccurrenceState::Planned,
        retry_count: 0,
        next_retry_at_ms: None,
        group_id: None,
        start_request_id: "00000000-0000-0000-0000-000000000103".into(),
        attempts: Vec::new(),
        last_error: None,
        suppressed_by_manual: false,
        created_at_ms: 0,
        updated_at_ms: 0,
        terminal_at_ms: None,
        expires_at_ms: None,
    }
}

fn status(now_ms: i64, reservation_id: Option<&str>) -> PowerStatus {
    let mut status = PowerStatus {
        protocol_version: POWER_PROTOCOL_VERSION,
        role: robo_rover_lib::LifecycleRole::Rover,
        entity_id: "rover-kiwi".into(),
        authority: PowerAuthority {
            epoch: 4,
            sequence: 8,
        },
        policy: PowerPolicy::Auto,
        requested_profile: PowerProfile::ScheduledCapture,
        effective_profile: PowerProfile::ScheduledCapture,
        state: PowerState::Active,
        transition_id: Some("00000000-0000-0000-0000-000000000104".into()),
        reason_code: None,
        detail: None,
        active_reservations: vec![],
        updated_at_ms: now_ms as u64,
    };
    if let Some(reservation_id) = reservation_id {
        status
            .active_reservations
            .push(robo_rover_lib::PowerReservationReadiness {
                reservation_id: reservation_id.into(),
                activation_started_at_ms: now_ms.saturating_sub(35_001) as u64,
            });
    }
    status
}

fn accepted(command_id: &str) -> PowerCommandResult {
    PowerCommandResult {
        protocol_version: POWER_PROTOCOL_VERSION,
        command_id: command_id.into(),
        accepted: true,
        authority: PowerAuthority {
            epoch: 4,
            sequence: 9,
        },
        reason_code: None,
        detail: None,
    }
}

fn prepared() -> (SchedulerRuntime<FakeClock>, FakeClock, String, String) {
    let clock = FakeClock::new(0);
    let mut runtime = SchedulerRuntime::from_occurrences(clock.clone(), [occurrence()]).unwrap();
    runtime.prepare_future_reservations(200_000);
    let group_id = runtime.groups.keys().next().unwrap().clone();
    let reservation_id = runtime.groups[&group_id]
        .power_reservation
        .as_ref()
        .unwrap()
        .reservation_id
        .clone();
    (runtime, clock, group_id, reservation_id)
}

#[test]
fn status_cannot_acknowledge_register_or_admit_recording() {
    let (mut runtime, clock, group_id, _) = prepared();
    let occurrence_id = occurrence().occurrence_id;
    runtime.mark_reservation_registering(&group_id, "00000000-0000-0000-0000-000000000105".into());
    runtime.observe_power_status(&status(
        clock.now_ms(),
        Some(
            &runtime.groups[&group_id]
                .power_reservation
                .as_ref()
                .unwrap()
                .reservation_id,
        ),
    ));
    assert!(!runtime.reservation_ready_for(&occurrence_id));

    assert_eq!(
        runtime.apply_power_command_result(&accepted("00000000-0000-0000-0000-000000000105")),
        Some("00000000-0000-0000-0000-000000000105".into())
    );
    clock.advance_ms(65_000);
    runtime.observe_power_status(&status(
        clock.now_ms(),
        Some(
            &runtime.groups[&group_id]
                .power_reservation
                .as_ref()
                .unwrap()
                .reservation_id,
        ),
    ));
    assert!(runtime.reservation_ready_for(&occurrence_id));
}

#[test]
fn fresh_status_that_omits_a_ready_reservation_revokes_admission() {
    let (mut runtime, clock, group_id, _) = prepared();
    let occurrence_id = occurrence().occurrence_id;
    let command_id = "00000000-0000-0000-0000-000000000120";
    runtime.mark_reservation_registering(&group_id, command_id.into());
    runtime.apply_power_command_result(&accepted(command_id));
    clock.advance_ms(65_000);
    runtime.observe_power_status(&status(
        clock.now_ms(),
        Some(
            &runtime.groups[&group_id]
                .power_reservation
                .as_ref()
                .unwrap()
                .reservation_id,
        ),
    ));
    assert!(runtime.reservation_ready_for(&occurrence_id));

    clock.advance_ms(1);
    runtime.observe_power_status(&status(clock.now_ms(), None));
    assert_eq!(
        runtime.groups[&group_id]
            .power_reservation
            .as_ref()
            .unwrap()
            .state,
        ReservationState::Blocked
    );
    assert!(!runtime.reservation_ready_for(&occurrence_id));
}

#[test]
fn send_crash_boundary_repairs_or_replays_only_the_exact_command() {
    let (mut runtime, _, group_id, _) = prepared();
    runtime.mark_reservation_registering(&group_id, "00000000-0000-0000-0000-000000000106".into());
    assert!(runtime.repair_reservation_outbox(&[]));
    assert_eq!(
        runtime.groups[&group_id]
            .power_reservation
            .as_ref()
            .unwrap()
            .state,
        ReservationState::Pending
    );

    runtime.mark_reservation_registering(&group_id, "00000000-0000-0000-0000-000000000107".into());
    assert!(!runtime.repair_reservation_outbox(&["00000000-0000-0000-0000-000000000107".into()]));
    assert_eq!(
        runtime.groups[&group_id]
            .power_reservation
            .as_ref()
            .unwrap()
            .state,
        ReservationState::Registering
    );
}

#[test]
fn delete_after_ready_keeps_release_tombstone_until_exact_acknowledgement() {
    let (mut runtime, clock, group_id, reservation_id) = prepared();
    let command_id = "00000000-0000-0000-0000-000000000108";
    runtime.mark_reservation_registering(&group_id, command_id.into());
    runtime.apply_power_command_result(&accepted(command_id));
    clock.advance_ms(65_000);
    runtime.observe_power_status(&status(
        clock.now_ms(),
        Some(
            &runtime.groups[&group_id]
                .power_reservation
                .as_ref()
                .unwrap()
                .reservation_id,
        ),
    ));
    runtime.occurrences.values_mut().next().unwrap().state = RecordingOccurrenceState::Cancelled;
    runtime.rebuild_groups().unwrap();

    assert!(runtime.groups.contains_key(&reservation_id));
    assert_eq!(
        runtime.reservations_to_release(),
        vec![reservation_id.clone()]
    );
    let release_id = "00000000-0000-0000-0000-000000000109";
    assert!(runtime.mark_reservation_releasing(&reservation_id, release_id.into()));
    runtime.apply_power_command_result(&accepted(release_id));
    runtime.prune_released_reservation_tombstones();
    assert!(!runtime.groups.contains_key(&reservation_id));
}

#[test]
fn schedule_revision_edit_after_ready_retires_the_old_reservation() {
    let (mut runtime, clock, group_id, reservation_id) = prepared();
    let command_id = "00000000-0000-0000-0000-000000000118";
    runtime.mark_reservation_registering(&group_id, command_id.into());
    runtime.apply_power_command_result(&accepted(command_id));
    clock.advance_ms(65_000);
    runtime.observe_power_status(&status(
        clock.now_ms(),
        Some(
            &runtime.groups[&group_id]
                .power_reservation
                .as_ref()
                .unwrap()
                .reservation_id,
        ),
    ));

    let old_id = occurrence().occurrence_id;
    let mut replacement = runtime.occurrences[&old_id].clone();
    replacement.occurrence_id = "00000000-0000-0000-0000-000000000119".into();
    replacement.schedule_revision = replacement.schedule_revision.saturating_add(1);
    replacement.group_id = None;
    runtime.occurrences.get_mut(&old_id).unwrap().state = RecordingOccurrenceState::Cancelled;
    runtime
        .occurrences
        .insert(replacement.occurrence_id.clone(), replacement);
    runtime.rebuild_groups().unwrap();

    assert!(runtime.groups.contains_key(&reservation_id));
    assert!(
        runtime.groups[&reservation_id]
            .power_reservation
            .as_ref()
            .unwrap()
            .retired
    );
    assert_eq!(runtime.reservations_to_release(), vec![reservation_id]);
}

#[test]
fn late_ready_records_activation_to_ready_latency_and_miss() {
    let (mut runtime, clock, group_id, _) = prepared();
    let command_id = "00000000-0000-0000-0000-000000000110";
    runtime.mark_reservation_registering(&group_id, command_id.into());
    runtime.apply_power_command_result(&accepted(command_id));
    clock.advance_ms(START_MS + 1);
    runtime.observe_power_status(&status(
        clock.now_ms(),
        Some(
            &runtime.groups[&group_id]
                .power_reservation
                .as_ref()
                .unwrap()
                .reservation_id,
        ),
    ));
    let reservation = runtime.groups[&group_id]
        .power_reservation
        .as_ref()
        .unwrap();
    assert_eq!(reservation.actual_ready_ms, Some(35_001));
    assert!(reservation.prewarm_missed);
    assert_eq!(
        reservation.sample_count, 1,
        "late Ready samples are retained so the rolling p95 learns misses"
    );
}

#[test]
fn invalid_after_ready_retries_release_with_a_new_exact_command() {
    let (mut runtime, clock, group_id, _) = prepared();
    let register_id = "00000000-0000-0000-0000-000000000111";
    runtime.mark_reservation_registering(&group_id, register_id.into());
    runtime.apply_power_command_result(&accepted(register_id));
    clock.advance_ms(65_000);
    runtime.observe_power_status(&status(
        clock.now_ms(),
        Some(
            &runtime.groups[&group_id]
                .power_reservation
                .as_ref()
                .unwrap()
                .reservation_id,
        ),
    ));
    runtime.occurrences.values_mut().next().unwrap().state = RecordingOccurrenceState::Cancelled;
    let release_id = "00000000-0000-0000-0000-000000000112";
    assert!(runtime.mark_reservation_releasing(&group_id, release_id.into()));
    assert_eq!(
        runtime.apply_power_command_result(&PowerCommandResult {
            protocol_version: POWER_PROTOCOL_VERSION,
            command_id: release_id.into(),
            accepted: false,
            authority: PowerAuthority {
                epoch: 4,
                sequence: 9
            },
            reason_code: Some(robo_rover_lib::PowerReasonCode::StaleAuthority),
            detail: Some("retry with fresh snapshot".into()),
        }),
        Some(release_id.into())
    );
    assert_eq!(
        runtime.groups[&group_id]
            .power_reservation
            .as_ref()
            .unwrap()
            .state,
        ReservationState::ReleasePending
    );
    assert_eq!(runtime.reservations_to_release(), vec![group_id.clone()]);
    assert!(runtime
        .mark_reservation_releasing(&group_id, "00000000-0000-0000-0000-000000000113".into()));
}

#[test]
fn multiple_reservations_keep_snapshot_and_ready_evidence_entity_scoped() {
    let clock = FakeClock::new(0);
    let mut other = occurrence();
    other.occurrence_id = "00000000-0000-0000-0000-000000000114".into();
    other.schedule_id = "00000000-0000-0000-0000-000000000115".into();
    other.entity_id = "rover-other".into();
    let mut runtime =
        SchedulerRuntime::from_occurrences(clock.clone(), [occurrence(), other]).unwrap();
    runtime.prepare_future_reservations(200_000);
    let kiwi_group = runtime
        .groups
        .values()
        .find(|group| group.entity_id == "rover-kiwi")
        .unwrap()
        .group_id
        .clone();
    let other_group = runtime
        .groups
        .values()
        .find(|group| group.entity_id == "rover-other")
        .unwrap()
        .group_id
        .clone();
    let kiwi_command = "00000000-0000-0000-0000-000000000116";
    let other_command = "00000000-0000-0000-0000-000000000117";
    runtime.mark_reservation_registering(&kiwi_group, kiwi_command.into());
    runtime.mark_reservation_registering(&other_group, other_command.into());
    runtime.apply_power_command_result(&accepted(kiwi_command));
    runtime.apply_power_command_result(&accepted(other_command));
    clock.advance_ms(65_000);
    runtime.observe_power_status(&status(
        clock.now_ms(),
        Some(
            &runtime.groups[&kiwi_group]
                .power_reservation
                .as_ref()
                .unwrap()
                .reservation_id,
        ),
    ));
    assert_eq!(
        runtime.groups[&kiwi_group]
            .power_reservation
            .as_ref()
            .unwrap()
            .state,
        ReservationState::Ready
    );
    assert_eq!(
        runtime.groups[&other_group]
            .power_reservation
            .as_ref()
            .unwrap()
            .state,
        ReservationState::Accepted
    );
}

#[test]
fn only_the_recorder_storage_allowlist_retries() {
    assert!(is_transient_recorder_storage_failure(
        Some(RecordingScheduleReasonCode::Unavailable),
        Some("storage temporarily unavailable"),
    ));
    assert!(!is_transient_recorder_storage_failure(
        Some(RecordingScheduleReasonCode::Internal),
        Some("storage temporarily unavailable"),
    ));
    assert!(!is_transient_recorder_storage_failure(
        Some(RecordingScheduleReasonCode::Unavailable),
        Some("arbitrary unavailable"),
    ));
}
