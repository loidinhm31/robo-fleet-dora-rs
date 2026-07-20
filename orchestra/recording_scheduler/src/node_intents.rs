use eyre::Result;
use robo_rover_lib::{
    RecordingOccurrence, ScheduledRecordingIntent, ScheduledRecordingIntentAction,
};

use crate::domain::RecordingGroup;

pub(crate) fn build_intent(
    occurrence: &RecordingOccurrence,
    group: &RecordingGroup,
    intent_id: String,
    action: ScheduledRecordingIntentAction,
) -> ScheduledRecordingIntent {
    ScheduledRecordingIntent {
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

pub(crate) fn pending_intents(
    scheduler: &crate::runtime::SchedulerRuntime<crate::clock::SystemClock>,
) -> Result<Vec<ScheduledRecordingIntent>> {
    Ok(scheduler
        .groups
        .values()
        .filter_map(|group| {
            let intent_id = group.pending_intent_id.clone()?;
            let action = group.pending_action?;
            scheduler
                .occurrences
                .values()
                .find(|occurrence| {
                    occurrence.group_id.as_deref() == Some(&group.group_id)
                        && matches!(
                            (action, occurrence.state),
                            (
                                ScheduledRecordingIntentAction::Acquire,
                                robo_rover_lib::RecordingOccurrenceState::StartPending
                            ) | (
                                ScheduledRecordingIntentAction::Release,
                                robo_rover_lib::RecordingOccurrenceState::StopPending
                            )
                        )
                })
                .map(|occurrence| build_intent(occurrence, group, intent_id, action))
        })
        .collect())
}
