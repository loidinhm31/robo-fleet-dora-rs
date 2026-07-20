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
