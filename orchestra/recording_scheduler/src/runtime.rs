use std::collections::BTreeMap;

use robo_rover_lib::{
    occurrence_id, scheduled_intent_id, RecordingAttemptState, RecordingClipAttempt,
    RecordingCoordinatorFeedback, RecordingOccurrence, RecordingOccurrenceState, RecordingSchedule,
    ScheduledRecordingIntent, ScheduledRecordingIntentAction,
};

use crate::mongo_repository::{OutboxRecord, Stored};
use crate::{
    clock::Clock,
    domain::{is_transient_reason, RecordingGroup},
    recurrence::candidates,
    runtime_groups::occurrence,
    state_machine::{retry_at, transition},
};

pub struct SchedulerRuntime<C> {
    pub(crate) clock: C,
    pub occurrences: BTreeMap<String, RecordingOccurrence>,
    pub groups: BTreeMap<String, RecordingGroup>,
    pub occurrence_revisions: BTreeMap<String, u64>,
    pub group_revisions: BTreeMap<String, u64>,
    pub removed_group_revisions: BTreeMap<String, u64>,
    pub(crate) reconciliation_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupTransition {
    pub intent_id: Option<String>,
    pub generation: u64,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct OutboxRecovery {
    pub replay: Vec<ScheduledRecordingIntent>,
    pub acknowledge: Vec<String>,
}

impl<C: Clock> SchedulerRuntime<C> {
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            occurrences: BTreeMap::new(),
            groups: BTreeMap::new(),
            occurrence_revisions: BTreeMap::new(),
            group_revisions: BTreeMap::new(),
            removed_group_revisions: BTreeMap::new(),
            reconciliation_complete: false,
        }
    }

    pub fn from_occurrences(
        clock: C,
        occurrences: impl IntoIterator<Item = RecordingOccurrence>,
    ) -> Result<Self, String> {
        let mut runtime = Self::new(clock);
        for occurrence in occurrences {
            runtime
                .occurrence_revisions
                .insert(occurrence.occurrence_id.clone(), 0);
            runtime
                .occurrences
                .insert(occurrence.occurrence_id.clone(), occurrence);
        }
        runtime.rebuild_groups()?;
        Ok(runtime)
    }

    pub fn from_persisted(
        clock: C,
        occurrences: impl IntoIterator<Item = Stored<RecordingOccurrence>>,
        groups: impl IntoIterator<Item = Stored<RecordingGroup>>,
    ) -> Result<Self, String> {
        let mut runtime = Self::new(clock);
        for stored in occurrences {
            runtime
                .occurrence_revisions
                .insert(stored.value.occurrence_id.clone(), stored.revision);
            runtime
                .occurrences
                .insert(stored.value.occurrence_id.clone(), stored.value);
        }
        for stored in groups {
            runtime
                .group_revisions
                .insert(stored.value.group_id.clone(), stored.revision);
            runtime
                .groups
                .insert(stored.value.group_id.clone(), stored.value);
        }
        runtime.rebuild_groups()?;
        Ok(runtime)
    }

    pub fn complete_reconciliation(&mut self) {
        self.reconciliation_complete = true;
    }

    pub fn has_handled_intent(&self, intent_id: &str) -> bool {
        self.groups
            .values()
            .any(|group| group.handled_feedback_generations.contains_key(intent_id))
    }

    /// The outbox is authoritative at process boundaries. A record either
    /// proves the transition is already terminal/applied or restores its
    /// snapshots before the coordinator can see a replay.
    pub fn recover_outbox(&mut self, records: &[OutboxRecord]) -> OutboxRecovery {
        let mut recovery = OutboxRecovery::default();
        for record in records {
            let handled = self.groups.values().any(|group| {
                group
                    .handled_feedback_generations
                    .contains_key(&record.intent.intent_id)
            });
            let terminal = self
                .occurrences
                .get(&record.intent.occurrence_id)
                .is_some_and(|occurrence| occurrence.state.is_terminal());
            let applied = record.intent.action == ScheduledRecordingIntentAction::Acquire
                && self
                    .occurrences
                    .get(&record.intent.occurrence_id)
                    .is_some_and(|item| item.state == RecordingOccurrenceState::Active);
            if handled || terminal || applied {
                recovery.acknowledge.push(record.intent.intent_id.clone());
                continue;
            }
            self.occurrence_revisions
                .entry(record.occurrence.occurrence_id.clone())
                .or_insert(0);
            self.group_revisions
                .entry(record.group.group_id.clone())
                .or_insert(0);
            self.occurrences.insert(
                record.occurrence.occurrence_id.clone(),
                record.occurrence.clone(),
            );
            self.groups
                .insert(record.group.group_id.clone(), record.group.clone());
            recovery.replay.push(record.intent.clone());
        }
        recovery
    }

    pub fn hydrate_group_directories(&mut self, schedules: &[RecordingSchedule]) {
        let directories = schedules
            .iter()
            .map(|schedule| {
                (
                    schedule.schedule_id.as_str(),
                    schedule.definition.relative_directory_template.as_str(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for group in self
            .groups
            .values_mut()
            .filter(|group| group.relative_directory.is_empty())
        {
            let selected = self
                .occurrences
                .values()
                .filter(|occurrence| occurrence.group_id.as_deref() == Some(&group.group_id))
                .min_by_key(|occurrence| (occurrence.planned_start_ms, &occurrence.occurrence_id))
                .and_then(|occurrence| directories.get(occurrence.schedule_id.as_str()))
                .copied();
            if let Some(directory) = selected {
                group.relative_directory = directory.to_owned();
            }
        }
    }

    pub fn materialize(
        &mut self,
        schedule: &RecordingSchedule,
        through_ms: i64,
    ) -> Result<usize, String> {
        let now_ms = self.clock.now_ms();
        let mut inserted = 0;
        for resolved in candidates(schedule, now_ms, through_ms)? {
            let id = occurrence_id(&schedule.schedule_id, schedule.revision, resolved.start_ms)?;
            if self.occurrences.contains_key(&id) {
                continue;
            }
            let occurrence = occurrence(
                schedule,
                id.clone(),
                resolved.start_ms,
                resolved.resolution,
                now_ms,
            )?;
            self.occurrences.insert(id, occurrence);
            inserted += 1;
        }
        self.assign_new_groups(schedule)?;
        for occurrence_id in self.occurrences.keys() {
            self.occurrence_revisions
                .entry(occurrence_id.clone())
                .or_insert(0);
        }
        for group_id in self.groups.keys() {
            self.group_revisions.entry(group_id.clone()).or_insert(0);
        }
        Ok(inserted)
    }

    pub fn due(&mut self) -> Vec<String> {
        if !self.reconciliation_complete {
            return Vec::new();
        }
        let now_ms = self.clock.now_ms();
        self.occurrences
            .values_mut()
            .filter_map(|occurrence| {
                ((occurrence.state == RecordingOccurrenceState::Planned
                    && occurrence.planned_start_ms <= now_ms)
                    || (occurrence.state == RecordingOccurrenceState::StartPending
                        && occurrence
                            .next_retry_at_ms
                            .is_some_and(|retry| retry <= now_ms)))
                .then(|| {
                    let next = if now_ms >= occurrence.planned_end_ms {
                        RecordingOccurrenceState::Missed
                    } else {
                        RecordingOccurrenceState::Due
                    };
                    transition(occurrence, next, now_ms, None)
                })
                .filter(|changed| *changed)
                .and_then(|_| {
                    (occurrence.state == RecordingOccurrenceState::Due)
                        .then(|| occurrence.occurrence_id.clone())
                })
            })
            .collect()
    }

    pub fn apply_feedback(&mut self, feedback: RecordingCoordinatorFeedback) -> bool {
        let Some((group_id, state)) =
            self.occurrences
                .get(&feedback.occurrence_id)
                .and_then(|occurrence| {
                    occurrence
                        .group_id
                        .clone()
                        .map(|group_id| (group_id, occurrence.state))
                })
        else {
            return false;
        };
        if !matches!(
            state,
            RecordingOccurrenceState::StartPending | RecordingOccurrenceState::StopPending
        ) {
            return false;
        }
        let action = if state == RecordingOccurrenceState::StartPending {
            ScheduledRecordingIntentAction::Acquire
        } else {
            ScheduledRecordingIntentAction::Release
        };
        if scheduled_intent_id(&feedback.occurrence_id, feedback.generation, action)
            .ok()
            .as_deref()
            != Some(&feedback.intent_id)
        {
            return false;
        }
        let Some(group) = self.groups.get_mut(&group_id) else {
            return false;
        };
        if group.pending_intent_id.as_deref() != Some(&feedback.intent_id)
            || group.pending_action != Some(action)
        {
            return false;
        }
        if !group.accept_feedback(&feedback.intent_id, feedback.generation) {
            return false;
        }
        let Some(occurrence) = self.occurrences.get_mut(&feedback.occurrence_id) else {
            return false;
        };
        let now_ms = self.clock.now_ms();
        if feedback.accepted && feedback.applied {
            if state == RecordingOccurrenceState::StopPending {
                if let Some(attempt) = occurrence.attempts.last_mut() {
                    attempt.state = RecordingAttemptState::Completed;
                    attempt.ended_at_ms = Some(now_ms);
                }
                group.recording_id = None;
                group.finish_intent();
                return transition(
                    occurrence,
                    RecordingOccurrenceState::Completed,
                    now_ms,
                    None,
                );
            }
            group.recording_id = feedback.recording_id;
            group.finish_intent();
            return self.activate_pending_group_owners(&group_id, now_ms);
        }
        if feedback.retryable && is_transient_reason(feedback.reason_code) {
            if action == ScheduledRecordingIntentAction::Acquire {
                group.retry_acquire();
            } else {
                group.retry_release();
            }
            group.finish_intent();
            occurrence.next_retry_at_ms = retry_at(occurrence, now_ms);
            return occurrence.next_retry_at_ms.is_some()
                || (action == ScheduledRecordingIntentAction::Acquire
                    && transition(occurrence, RecordingOccurrenceState::Missed, now_ms, None));
        }
        group.finish_intent();
        if let Some(recording_id) = feedback.recording_id {
            occurrence.attempts.push(RecordingClipAttempt {
                recording_id,
                state: RecordingAttemptState::Failed,
                started_at_ms: now_ms,
                ended_at_ms: Some(now_ms),
            });
        }
        let failed = transition(
            occurrence,
            RecordingOccurrenceState::Failed,
            now_ms,
            feedback.reason_code,
        );
        if failed && action == ScheduledRecordingIntentAction::Acquire {
            group.remove_owner(&feedback.occurrence_id);
            self.return_pending_group_owners_to_due(&group_id, now_ms);
        }
        failed
    }

    pub fn begin_start(&mut self, occurrence_id: &str) -> Option<GroupTransition> {
        let group_id = self.occurrences.get(occurrence_id)?.group_id.clone()?;
        if self.occurrences.get(occurrence_id)?.state != RecordingOccurrenceState::Due {
            return None;
        }
        let group = self.groups.get_mut(&group_id)?;
        let was_empty = group.owner_ids.is_empty();
        group.add_owner(occurrence_id);
        let group_is_active = self.occurrences.values().any(|occurrence| {
            occurrence.group_id.as_deref() == Some(&group_id)
                && occurrence.state == RecordingOccurrenceState::Active
        });
        let occurrence = self.occurrences.get_mut(occurrence_id)?;
        let next = if was_empty || !group_is_active {
            RecordingOccurrenceState::StartPending
        } else {
            RecordingOccurrenceState::Active
        };
        if !transition(occurrence, next, self.clock.now_ms(), None) {
            return None;
        }
        if group_is_active || group.pending_intent_id.is_some() {
            return Some(GroupTransition {
                intent_id: None,
                generation: group.generation,
            });
        }
        let intent_id = scheduled_intent_id(
            occurrence_id,
            group.generation,
            ScheduledRecordingIntentAction::Acquire,
        )
        .ok()?;
        group.begin_intent(intent_id.clone(), ScheduledRecordingIntentAction::Acquire);
        Some(GroupTransition {
            intent_id: Some(intent_id),
            generation: group.generation,
        })
    }

    pub fn begin_stop(&mut self, occurrence_id: &str) -> Option<GroupTransition> {
        let group_id = self.occurrences.get(occurrence_id)?.group_id.clone()?;
        let state = self.occurrences.get(occurrence_id)?.state;
        if !matches!(
            state,
            RecordingOccurrenceState::Active | RecordingOccurrenceState::StopPending
        ) {
            return None;
        }
        let group = self.groups.get_mut(&group_id)?;
        if state == RecordingOccurrenceState::Active {
            group.remove_owner(occurrence_id);
        }
        let is_last_owner = group.owner_ids.is_empty();
        let occurrence = self.occurrences.get_mut(occurrence_id)?;
        let next = if is_last_owner {
            RecordingOccurrenceState::StopPending
        } else {
            RecordingOccurrenceState::Completed
        };
        if state == RecordingOccurrenceState::Active
            && !transition(occurrence, next, self.clock.now_ms(), None)
        {
            return None;
        }
        if next == RecordingOccurrenceState::Completed {
            if let Some(attempt) = occurrence.attempts.last_mut() {
                attempt.state = RecordingAttemptState::Completed;
                attempt.ended_at_ms = Some(self.clock.now_ms());
            }
        }
        if !is_last_owner || group.pending_intent_id.is_some() {
            return Some(GroupTransition {
                intent_id: None,
                generation: group.generation,
            });
        }
        let intent_id = scheduled_intent_id(
            occurrence_id,
            group.generation,
            ScheduledRecordingIntentAction::Release,
        )
        .ok()?;
        group.begin_intent(intent_id.clone(), ScheduledRecordingIntentAction::Release);
        Some(GroupTransition {
            intent_id: Some(intent_id),
            generation: group.generation,
        })
    }

    pub fn due_stops(&mut self) -> Vec<String> {
        if !self.reconciliation_complete {
            return Vec::new();
        }
        let now_ms = self.clock.now_ms();
        self.occurrences
            .values()
            .filter(|occurrence| {
                (occurrence.state == RecordingOccurrenceState::Active
                    && occurrence.planned_end_ms <= now_ms)
                    || (occurrence.state == RecordingOccurrenceState::StopPending
                        && occurrence
                            .next_retry_at_ms
                            .is_some_and(|retry| retry <= now_ms))
            })
            .map(|occurrence| occurrence.occurrence_id.clone())
            .collect()
    }

    fn activate_pending_group_owners(&mut self, group_id: &str, now_ms: i64) -> bool {
        let recording_id = self.groups[group_id].recording_id.clone();
        let mut changed = false;
        for occurrence in self.occurrences.values_mut().filter(|occurrence| {
            occurrence.group_id.as_deref() == Some(group_id)
                && occurrence.state == RecordingOccurrenceState::StartPending
        }) {
            changed |= transition(occurrence, RecordingOccurrenceState::Active, now_ms, None);
            if let Some(recording_id) = &recording_id {
                occurrence.attempts.push(RecordingClipAttempt {
                    recording_id: recording_id.clone(),
                    state: RecordingAttemptState::Started,
                    started_at_ms: now_ms,
                    ended_at_ms: None,
                });
            }
        }
        changed
    }

    fn return_pending_group_owners_to_due(&mut self, group_id: &str, now_ms: i64) {
        for occurrence in self.occurrences.values_mut().filter(|occurrence| {
            occurrence.group_id.as_deref() == Some(group_id)
                && occurrence.state == RecordingOccurrenceState::StartPending
        }) {
            transition(occurrence, RecordingOccurrenceState::Due, now_ms, None);
        }
    }
}
