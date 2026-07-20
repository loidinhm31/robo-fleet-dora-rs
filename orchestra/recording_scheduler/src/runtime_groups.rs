use std::collections::BTreeMap;

use robo_rover_lib::{
    scheduled_group_id, scheduled_start_request_id, DstResolution, RecordingOccurrence,
    RecordingOccurrenceState, RecordingSchedule,
};

use crate::{clock::Clock, domain::RecordingGroup, runtime::SchedulerRuntime};

impl<C: Clock> SchedulerRuntime<C> {
    pub fn rebuild_groups(&mut self) -> Result<(), String> {
        self.rebuild_groups_with_directories(&BTreeMap::new())
    }

    pub(crate) fn assign_new_groups(&mut self, schedule: &RecordingSchedule) -> Result<(), String> {
        self.rebuild_groups_with_directories(&BTreeMap::from([(
            schedule.schedule_id.as_str(),
            schedule.definition.relative_directory_template.as_str(),
        )]))
    }

    fn rebuild_groups_with_directories(
        &mut self,
        directories: &BTreeMap<&str, &str>,
    ) -> Result<(), String> {
        let previous = std::mem::take(&mut self.groups);
        let mut items = self
            .occurrences
            .values()
            .filter(|item| !item.state.is_terminal())
            .cloned()
            .collect::<Vec<_>>();
        items.sort_by_key(|item| {
            (
                item.entity_id.clone(),
                item.planned_start_ms,
                item.occurrence_id.clone(),
            )
        });
        let mut rebuilt = BTreeMap::new();
        let mut cursor = 0;
        while cursor < items.len() {
            let first = cursor;
            let entity_id = items[cursor].entity_id.clone();
            let mut end_ms = items[cursor].planned_end_ms;
            cursor += 1;
            while cursor < items.len()
                && items[cursor].entity_id == entity_id
                && items[cursor].planned_start_ms < end_ms
            {
                end_ms = end_ms.max(items[cursor].planned_end_ms);
                cursor += 1;
            }
            let members = &items[first..cursor];
            let earliest = &members[0];
            let locked_group_ids = members
                .iter()
                .filter_map(|member| member.group_id.as_ref())
                .filter(|group_id| {
                    previous.get(*group_id).is_some_and(|group| {
                        group.recording_id.is_some()
                            || group.pending_intent_id.is_some()
                            || !group.owner_ids.is_empty()
                    })
                })
                .collect::<std::collections::BTreeSet<_>>();
            if locked_group_ids.len() > 1 {
                return Err("cannot merge multiple live recording groups".into());
            }
            let group_id = locked_group_ids
                .iter()
                .next()
                .map(|group_id| (*group_id).clone())
                .unwrap_or(scheduled_group_id(&entity_id, earliest.planned_start_ms)?);
            let old_directory = earliest
                .group_id
                .as_ref()
                .and_then(|id| previous.get(id))
                .map(|group| group.relative_directory.clone());
            let directory = directories
                .get(earliest.schedule_id.as_str())
                .map(|value| (*value).to_owned())
                .or(old_directory)
                .unwrap_or_default();
            let locked = locked_group_ids.contains(&group_id);
            let mut group = previous
                .get(&group_id)
                .cloned()
                .unwrap_or(RecordingGroup::new(
                    &entity_id,
                    earliest.planned_start_ms,
                    end_ms,
                    directory.clone(),
                )?);
            if !locked {
                group.start_ms = earliest.planned_start_ms;
            }
            group.end_ms = end_ms;
            if group.relative_directory.is_empty() {
                group.relative_directory = directory;
            }
            for member in members {
                self.occurrences
                    .get_mut(&member.occurrence_id)
                    .expect("occurrence exists")
                    .group_id = Some(group_id.clone());
            }
            rebuilt.insert(group_id, group);
        }
        for (group_id, _) in previous
            .into_iter()
            .filter(|(id, _)| !rebuilt.contains_key(id))
        {
            if let Some(revision) = self.group_revisions.remove(&group_id) {
                self.removed_group_revisions.insert(group_id, revision);
            }
        }
        self.groups = rebuilt;
        for group_id in self.groups.keys() {
            self.group_revisions.entry(group_id.clone()).or_insert(0);
        }
        Ok(())
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
