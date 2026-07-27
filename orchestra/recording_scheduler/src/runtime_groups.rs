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
            let member_ids = members
                .iter()
                .map(|member| member.occurrence_id.clone())
                .collect::<std::collections::BTreeSet<_>>();
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
            let previous_group = previous.get(&group_id);
            let reservation_changed = previous_group
                .and_then(|group| group.power_reservation.as_ref())
                .is_some_and(|reservation| {
                    !matches!(reservation.state, crate::domain::ReservationState::Released)
                        && (previous_group.is_some_and(|group| group.end_ms != end_ms)
                            || previous_group
                                .is_some_and(|group| group.start_ms != earliest.planned_start_ms)
                            // A schedule edit can preserve the same wall-clock
                            // window while replacing its revision-scoped
                            // occurrence ID.  A registered reservation's
                            // immutable payload is tied to the old owner set,
                            // so it must be retired and released rather than
                            // silently adopted by the replacement group.
                            || previous_group
                                .is_some_and(|group| group.owner_ids != member_ids))
                });
            if reservation_changed {
                if let Some(tombstone) = previous_group.and_then(reservation_tombstone) {
                    rebuilt.insert(tombstone.group_id.clone(), tombstone);
                    self.group_revisions
                        .entry(
                            previous_group
                                .and_then(|group| group.power_reservation.as_ref())
                                .expect("reservation checked")
                                .reservation_id
                                .clone(),
                        )
                        .or_insert(0);
                }
            }
            let mut group = if reservation_changed {
                let mut fresh = RecordingGroup::new(
                    &entity_id,
                    earliest.planned_start_ms,
                    end_ms,
                    directory.clone(),
                )?;
                // The group identity remains the overlap key; only the old
                // reservation becomes a separately persisted tombstone.
                fresh.group_id = group_id.clone();
                fresh.start_request_id = scheduled_start_request_id(&group_id)?;
                fresh
            } else {
                previous
                    .get(&group_id)
                    .cloned()
                    .unwrap_or(RecordingGroup::new(
                        &entity_id,
                        earliest.planned_start_ms,
                        end_ms,
                        directory.clone(),
                    )?)
            };
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
        let removed = previous
            .into_iter()
            .filter(|(id, _)| !rebuilt.contains_key(id))
            .collect::<Vec<_>>();
        for (group_id, group) in removed {
            if let Some(tombstone) = reservation_tombstone(&group) {
                self.group_revisions
                    .entry(tombstone.group_id.clone())
                    .or_insert(0);
                rebuilt.insert(tombstone.group_id.clone(), tombstone);
            }
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

/// Keep a releaseable reservation after its source schedule/group disappears.
/// It has no recorder ownership and is persisted under the reservation UUID so
/// an edit that reuses the overlap group id cannot overwrite the release work.
fn reservation_tombstone(group: &RecordingGroup) -> Option<RecordingGroup> {
    let reservation = group.power_reservation.as_ref()?;
    if matches!(
        reservation.state,
        crate::domain::ReservationState::Pending | crate::domain::ReservationState::Released
    ) {
        return None;
    }
    let mut tombstone = group.clone();
    tombstone.group_id = reservation.reservation_id.clone();
    tombstone.owner_ids.clear();
    tombstone.recording_id = None;
    tombstone.pending_intent_id = None;
    tombstone.pending_action = None;
    tombstone.power_reservation.as_mut()?.retired = true;
    Some(tombstone)
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
