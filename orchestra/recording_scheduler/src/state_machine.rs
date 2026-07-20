use robo_rover_lib::{
    RecordingOccurrence, RecordingOccurrenceError, RecordingOccurrenceState,
    RecordingScheduleReasonCode, TERMINAL_OCCURRENCE_RETENTION_MS,
};

pub fn transition(
    occurrence: &mut RecordingOccurrence,
    next: RecordingOccurrenceState,
    now_ms: i64,
    reason: Option<RecordingScheduleReasonCode>,
) -> bool {
    if !legal(occurrence.state, next) || occurrence.state == next {
        return false;
    }
    occurrence.state = next;
    occurrence.updated_at_ms = now_ms;
    occurrence.next_retry_at_ms = None;
    occurrence.last_error = reason.map(|reason_code| RecordingOccurrenceError {
        reason_code,
        detail: "coordinator rejected scheduled transition".into(),
    });
    if next.is_terminal() {
        occurrence.terminal_at_ms = Some(now_ms);
        occurrence.expires_at_ms = Some(now_ms.saturating_add(TERMINAL_OCCURRENCE_RETENTION_MS));
    }
    true
}

pub fn retry_at(occurrence: &mut RecordingOccurrence, now_ms: i64) -> Option<i64> {
    let delay_seconds = [1_i64, 2, 4, 8, 16]
        .get(occurrence.retry_count as usize)
        .copied()
        .unwrap_or(30);
    occurrence.retry_count = occurrence.retry_count.saturating_add(1);
    let retry_at = now_ms.saturating_add(delay_seconds * 1_000);
    (occurrence.state == RecordingOccurrenceState::StopPending
        || retry_at < occurrence.planned_end_ms)
        .then_some(retry_at)
}

fn legal(current: RecordingOccurrenceState, next: RecordingOccurrenceState) -> bool {
    use RecordingOccurrenceState::*;
    matches!(
        (current, next),
        (Planned, Due | Cancelled | Missed)
            | (Due, StartPending | Active | Suppressed | Missed | Failed)
            | (StartPending, Active | Due | Failed | Missed | Suppressed)
            | (Active, StopPending | Completed | Suppressed | Failed)
            | (StopPending, Completed | Active | Failed)
    )
}
