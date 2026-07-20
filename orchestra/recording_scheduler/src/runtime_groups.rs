use robo_rover_lib::{
    scheduled_start_request_id, DstResolution, RecordingOccurrence, RecordingOccurrenceState,
    RecordingSchedule,
};

use crate::{clock::Clock, domain::RecordingGroup, runtime::SchedulerRuntime};

impl<C: Clock> SchedulerRuntime<C> {
    pub fn rebuild_groups(&mut self) -> Result<(), String> {
        let mut occurrences = self
            .occurrences
            .values()
            .filter(|item| !item.state.is_terminal())
            .cloned()
            .collect::<Vec<_>>();
        occurrences.sort_by(|left, right| {
            (&left.entity_id, left.planned_start_ms, &left.occurrence_id).cmp(&(
                &right.entity_id,
                right.planned_start_ms,
                &right.occurrence_id,
            ))
        });
        self.groups.clear();
        let mut assignments = Vec::with_capacity(occurrences.len());
        for occurrence in occurrences {
            let group_id = self.group_id_for(&occurrence, String::new())?;
            let group = self.groups.get_mut(&group_id).expect("group inserted");
            group.end_ms = group.end_ms.max(occurrence.planned_end_ms);
            assignments.push((occurrence.occurrence_id, group_id));
        }
        for (occurrence_id, group_id) in assignments {
            self.occurrences
                .get_mut(&occurrence_id)
                .expect("occurrence exists")
                .group_id = Some(group_id);
        }
        Ok(())
    }

    pub(crate) fn assign_new_groups(&mut self, schedule: &RecordingSchedule) -> Result<(), String> {
        let mut ids = self
            .occurrences
            .values()
            .filter(|occurrence| {
                occurrence.schedule_id == schedule.schedule_id
                    && occurrence.schedule_revision == schedule.revision
                    && occurrence.group_id.is_none()
            })
            .map(|occurrence| occurrence.occurrence_id.clone())
            .collect::<Vec<_>>();
        ids.sort_by(|left, right| {
            let left = &self.occurrences[left];
            let right = &self.occurrences[right];
            (&left.entity_id, left.planned_start_ms, &left.occurrence_id).cmp(&(
                &right.entity_id,
                right.planned_start_ms,
                &right.occurrence_id,
            ))
        });
        for id in ids {
            let occurrence = self.occurrences[&id].clone();
            let group_id = self.group_id_for(
                &occurrence,
                schedule.definition.relative_directory_template.clone(),
            )?;
            let group = self.groups.get_mut(&group_id).expect("group exists");
            group.end_ms = group.end_ms.max(occurrence.planned_end_ms);
            self.occurrences
                .get_mut(&id)
                .expect("occurrence exists")
                .group_id = Some(group_id);
        }
        Ok(())
    }

    fn group_id_for(
        &mut self,
        occurrence: &RecordingOccurrence,
        relative_directory: String,
    ) -> Result<String, String> {
        if let Some(group) = self.groups.values().find(|group| {
            group.entity_id == occurrence.entity_id
                && group.start_ms < occurrence.planned_end_ms
                && occurrence.planned_start_ms < group.end_ms
        }) {
            return Ok(group.group_id.clone());
        }
        let group = RecordingGroup::new(
            &occurrence.entity_id,
            occurrence.planned_start_ms,
            occurrence.planned_end_ms,
            relative_directory,
        )?;
        let group_id = group.group_id.clone();
        self.groups.insert(group_id.clone(), group);
        Ok(group_id)
    }
}

pub(crate) fn occurrence(
    schedule: &RecordingSchedule,
    occurrence_id: String,
    start_ms: i64,
    resolution: DstResolution,
    now_ms: i64,
) -> Result<RecordingOccurrence, String> {
    Ok(RecordingOccurrence {
        start_request_id: scheduled_start_request_id(&occurrence_id)?,
        occurrence_id,
        schedule_id: schedule.schedule_id.clone(),
        schedule_revision: schedule.revision,
        entity_id: schedule.definition.entity_id.clone(),
        planned_start_ms: start_ms,
        planned_end_ms: start_ms.saturating_add(schedule.definition.duration_ms),
        dst_resolution: resolution,
        state: RecordingOccurrenceState::Planned,
        retry_count: 0,
        next_retry_at_ms: None,
        group_id: None,
        attempts: Vec::new(),
        last_error: None,
        suppressed_by_manual: false,
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
        terminal_at_ms: None,
        expires_at_ms: None,
    })
}
