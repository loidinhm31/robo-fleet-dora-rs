use std::collections::BTreeMap;

use robo_rover_lib::{
    occurrence_id, scheduled_intent_id, PowerCommandResult, PowerProfile, PowerState, PowerStatus,
    RecordingAttemptState, RecordingClipAttempt, RecordingCoordinatorFeedback, RecordingOccurrence,
    RecordingOccurrenceState, RecordingSchedule, ScheduledRecordingIntent,
    ScheduledRecordingIntentAction,
};

use crate::mongo_repository::{OutboxRecord, Stored};
use crate::{
    clock::Clock,
    domain::{is_transient_recorder_storage_failure, RecordingGroup, ReservationState},
    prewarm::{PersistedPrewarmEstimator, PrewarmEstimator, PrewarmMetrics},
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
    prewarm_estimators: BTreeMap<String, PrewarmEstimator>,
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
            prewarm_estimators: BTreeMap::new(),
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

    pub fn restore_prewarm_estimators(
        &mut self,
        estimators: impl IntoIterator<Item = PersistedPrewarmEstimator>,
    ) {
        self.prewarm_estimators = estimators
            .into_iter()
            .map(|item| {
                (
                    item.entity_id,
                    PrewarmEstimator::from_samples(item.samples, item.miss_count),
                )
            })
            .collect();
    }

    pub fn prewarm_estimators(&self) -> Vec<PersistedPrewarmEstimator> {
        self.prewarm_estimators
            .iter()
            .map(|(entity_id, estimator)| PersistedPrewarmEstimator {
                entity_id: entity_id.clone(),
                samples: estimator.samples(),
                miss_count: estimator.miss_count(),
            })
            .collect()
    }

    pub fn prewarm_metrics(&self, group_id: &str) -> Option<PrewarmMetrics> {
        let group = self.groups.get(group_id)?;
        let reservation = group.power_reservation.as_ref()?;
        let estimator = self.prewarm_estimators.get(&group.entity_id);
        Some(PrewarmMetrics {
            entity_id: group.entity_id.clone(),
            reservation_id: reservation.reservation_id.clone(),
            sample_count: reservation.sample_count,
            estimate_ms: reservation.prewarm_estimate_ms,
            actual_ready_ms: reservation.actual_ready_ms,
            bootstrap_active: reservation.bootstrap_active,
            missed: reservation.prewarm_missed,
            miss_count: estimator
                .map(PrewarmEstimator::miss_count)
                .unwrap_or_default(),
        })
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
                let due = occurrence.state == RecordingOccurrenceState::Due
                    || (occurrence.state == RecordingOccurrenceState::Planned
                        && occurrence.planned_start_ms <= now_ms)
                    || (occurrence.state == RecordingOccurrenceState::StartPending
                        && occurrence
                            .next_retry_at_ms
                            .is_some_and(|retry| retry <= now_ms));
                if !due {
                    return None;
                }
                let missed = now_ms >= occurrence.planned_end_ms;
                let next = if missed {
                    RecordingOccurrenceState::Missed
                } else {
                    RecordingOccurrenceState::Due
                };
                if occurrence.state != RecordingOccurrenceState::Due
                    && !transition(occurrence, next, now_ms, None)
                {
                    return None;
                }
                if missed {
                    tracing::warn!(
                        event = "recording_scheduler_missed",
                        occurrence_id = %occurrence.occurrence_id,
                        planned_end_ms = occurrence.planned_end_ms,
                        "scheduled recording missed its entire planned window"
                    );
                    None
                } else {
                    Some(occurrence.occurrence_id.clone())
                }
            })
            .collect()
    }

    /// Materialize group-scoped reservations only inside the power contract's
    /// bounded seven-day horizon. Persisting this state before command output
    /// makes a restart replay the same deterministic reservation identity.
    pub fn prepare_future_reservations(&mut self, horizon_ms: i64) -> Vec<String> {
        let now_ms = self.clock.now_ms();
        let group_ids = self
            .groups
            .values()
            .filter(|group| {
                group.start_ms > now_ms
                    && group.start_ms <= horizon_ms
                    && group.end_ms <= horizon_ms
            })
            .map(|group| group.group_id.clone())
            .collect::<Vec<_>>();
        let mut changed = Vec::new();
        for group_id in group_ids {
            let Some(group) = self.groups.get_mut(&group_id) else {
                continue;
            };
            let estimate = self
                .prewarm_estimators
                .entry(group.entity_id.clone())
                .or_default()
                .estimate();
            if group
                .ensure_power_reservation(
                    group.start_ms.saturating_sub(estimate.estimate_ms),
                    now_ms,
                )
                .unwrap_or(false)
            {
                changed.push(group_id);
            }
            if let Some(reservation) = group.power_reservation.as_mut() {
                reservation.sample_count = estimate.sample_count;
                reservation.prewarm_estimate_ms = estimate.estimate_ms;
                reservation.bootstrap_active = estimate.bootstrap_active;
            }
        }
        changed
    }

    pub fn pending_reservations(&self) -> Vec<String> {
        self.groups
            .values()
            .filter(|group| {
                group.power_reservation.as_ref().is_some_and(|reservation| {
                    reservation.state == ReservationState::Pending && !reservation.retired
                })
            })
            .map(|group| group.group_id.clone())
            .collect()
    }

    /// A local send is not a registration acknowledgement. Only the exact,
    /// signed Rover command result advances a reservation out of Registering.
    pub fn mark_reservation_registering(&mut self, group_id: &str, command_id: String) -> bool {
        let Some(reservation) = self
            .groups
            .get_mut(group_id)
            .and_then(|group| group.power_reservation.as_mut())
        else {
            return false;
        };
        if reservation.state != ReservationState::Pending || reservation.retired {
            return false;
        }
        reservation.state = ReservationState::Registering;
        reservation.command_id = Some(command_id);
        reservation.command_attempt = reservation.command_attempt.saturating_add(1);
        true
    }

    /// Correlate a remote acknowledgement by command id. Aggregate status may
    /// inform readiness after this point but can never acknowledge delivery.
    pub fn apply_power_command_result(&mut self, result: &PowerCommandResult) -> Option<String> {
        let now_ms = self.clock.now_ms();
        for group in self.groups.values_mut() {
            let Some(reservation) = group.power_reservation.as_mut() else {
                continue;
            };
            if reservation.command_id.as_deref() != Some(&result.command_id) {
                continue;
            }
            match reservation.state {
                ReservationState::Registering => {
                    reservation.command_id = None;
                    if result.accepted {
                        reservation.state = ReservationState::Accepted;
                        reservation.registered_at_ms = Some(now_ms);
                        reservation.register_ack_command_id = Some(result.command_id.clone());
                    } else {
                        // A deleted/superseded owner still sends a release
                        // fence even when register was rejected: a delayed
                        // duplicate must not revive this reservation later.
                        reservation.state = if reservation.retired {
                            ReservationState::ReleasePending
                        } else {
                            ReservationState::Failed
                        };
                    }
                    return Some(result.command_id.clone());
                }
                ReservationState::Releasing => {
                    reservation.command_id = None;
                    reservation.state = if result.accepted {
                        ReservationState::Released
                    } else {
                        ReservationState::ReleasePending
                    };
                    return Some(result.command_id.clone());
                }
                _ => return None,
            }
        }
        None
    }

    /// A tombstone outlives deletion/edit/manual suppression only until the
    /// exact release acknowledgement is durable. Then normal CAS persistence
    /// removes its now-unneeded synthetic group record.
    pub fn prune_released_reservation_tombstones(&mut self) {
        let released = self
            .groups
            .iter()
            .filter_map(|(group_id, group)| {
                group
                    .power_reservation
                    .as_ref()
                    .is_some_and(|reservation| {
                        reservation.retired && reservation.state == ReservationState::Released
                    })
                    .then(|| group_id.clone())
            })
            .collect::<Vec<_>>();
        for group_id in released {
            self.groups.remove(&group_id);
            if let Some(revision) = self.group_revisions.remove(&group_id) {
                self.removed_group_revisions.insert(group_id, revision);
            }
        }
    }

    /// Repair a process boundary between group persistence and outbox insert.
    /// No command was sent without an outbox row, so an in-flight state lacking
    /// its exact command can safely return to a pending durable intent.
    pub fn repair_reservation_outbox(&mut self, command_ids: &[String]) -> bool {
        let mut changed = false;
        for group in self.groups.values_mut() {
            let Some(reservation) = group.power_reservation.as_mut() else {
                continue;
            };
            if reservation
                .command_id
                .as_ref()
                .is_some_and(|command_id| !command_ids.contains(command_id))
            {
                reservation.state = match reservation.state {
                    ReservationState::Registering if reservation.retired => {
                        ReservationState::ReleasePending
                    }
                    ReservationState::Registering => ReservationState::Pending,
                    ReservationState::Releasing => ReservationState::ReleasePending,
                    state => state,
                };
                reservation.command_id = None;
                changed = true;
            }
        }
        changed
    }

    pub fn expire_reservation_command(&mut self, command_id: &str) -> bool {
        for group in self.groups.values_mut() {
            let Some(reservation) = group.power_reservation.as_mut() else {
                continue;
            };
            if reservation.command_id.as_deref() != Some(command_id) {
                continue;
            }
            reservation.state = match reservation.state {
                ReservationState::Registering if reservation.retired => {
                    ReservationState::ReleasePending
                }
                ReservationState::Registering => ReservationState::Failed,
                ReservationState::Releasing => ReservationState::ReleasePending,
                state => state,
            };
            reservation.command_id = None;
            return true;
        }
        false
    }

    /// Status is aggregate evidence, not an admission grant. It unlocks only
    /// reservations for the exact rover after their prewarm boundary and only
    /// once ScheduledCapture is effective and active.
    pub fn observe_power_status(&mut self, status: &PowerStatus) -> Vec<String> {
        let now_ms = self.clock.now_ms();
        let mut changed = Vec::new();
        for group in self
            .groups
            .values_mut()
            .filter(|group| group.entity_id == status.entity_id)
        {
            let Some(reservation) = group.power_reservation.as_mut() else {
                continue;
            };
            let readiness = status
                .active_reservations
                .iter()
                .find(|item| item.reservation_id == reservation.reservation_id);
            if !matches!(
                reservation.state,
                ReservationState::Accepted
                    | ReservationState::Prewarming
                    | ReservationState::Blocked
                    | ReservationState::Failed
                    | ReservationState::Ready
            ) {
                continue;
            }
            let Ok(status_updated_at_ms) = i64::try_from(status.updated_at_ms) else {
                continue;
            };
            if reservation
                .status_updated_at_ms
                .is_some_and(|previous| status_updated_at_ms < previous)
            {
                continue;
            }
            reservation.status_updated_at_ms = Some(status_updated_at_ms);
            // A newer, signed aggregate status that no longer names a
            // reservation is negative readiness evidence. In particular, it
            // must revoke a previously Ready admission immediately instead of
            // leaving old evidence fresh by timestamp alone.
            if readiness.is_none() && reservation.state == ReservationState::Ready {
                reservation.state = ReservationState::Blocked;
                reservation.ready_authority = None;
                changed.push(group.group_id.clone());
                continue;
            }
            let next = if matches!(status.state, PowerState::Failed | PowerState::Degraded) {
                ReservationState::Failed
            } else if status.reason_code.is_some() {
                ReservationState::Blocked
            } else if let Some(readiness) = readiness {
                let Ok(activation_started_at_ms) =
                    i64::try_from(readiness.activation_started_at_ms)
                else {
                    continue;
                };
                reservation.activation_started_at_ms = Some(activation_started_at_ms);
                if now_ms >= reservation.prewarm_at_ms
                    && status.effective_profile == PowerProfile::ScheduledCapture
                    && status.state == PowerState::Active
                {
                    ReservationState::Ready
                } else if now_ms >= reservation.prewarm_at_ms {
                    ReservationState::Prewarming
                } else {
                    continue;
                }
            } else {
                continue;
            };
            if reservation.state != next {
                reservation.state = next;
                reservation.transition_id = status.transition_id.clone();
                if next == ReservationState::Ready {
                    reservation.ready_at_ms = Some(now_ms);
                    let activation_started_at_ms =
                        reservation.activation_started_at_ms.get_or_insert(now_ms);
                    let actual_ready_ms = now_ms.saturating_sub(*activation_started_at_ms);
                    reservation.actual_ready_ms = Some(actual_ready_ms);
                    reservation.prewarm_missed = now_ms > group.start_ms;
                    let estimate = self
                        .prewarm_estimators
                        .entry(group.entity_id.clone())
                        .or_default();
                    estimate.observe(actual_ready_ms, reservation.prewarm_missed);
                    let estimate = estimate.estimate();
                    reservation.sample_count = estimate.sample_count;
                    reservation.prewarm_estimate_ms = estimate.estimate_ms;
                    reservation.bootstrap_active = estimate.bootstrap_active;
                    reservation.ready_authority = Some(status.authority);
                }
                changed.push(group.group_id.clone());
            }
        }
        changed
    }

    pub fn reservation_ready_for(&self, occurrence_id: &str) -> bool {
        const STATUS_FRESHNESS_MS: i64 = 30_000;
        let now_ms = self.clock.now_ms();
        self.occurrences
            .get(occurrence_id)
            .and_then(|occurrence| occurrence.group_id.as_ref())
            .and_then(|group_id| self.groups.get(group_id))
            .is_some_and(|group| {
                group.reservation_is_ready()
                    && group.power_reservation.as_ref().is_some_and(|reservation| {
                        reservation.register_ack_command_id.is_some()
                            && reservation.ready_authority.is_some()
                            && reservation
                                .status_updated_at_ms
                                .is_some_and(|updated_at_ms| {
                                    updated_at_ms >= now_ms.saturating_sub(STATUS_FRESHNESS_MS)
                                })
                    })
            })
    }

    pub fn reservations_to_release(&self) -> Vec<String> {
        self.groups
            .values()
            .filter(|group| {
                group.reservation_needs_release()
                    && group.power_reservation.as_ref().is_some_and(|reservation| {
                        reservation.state != ReservationState::Registering
                    })
                    && self
                        .occurrences
                        .values()
                        .filter(|occurrence| {
                            occurrence.group_id.as_deref() == Some(&group.group_id)
                        })
                        .all(|occurrence| occurrence.state.is_terminal())
            })
            .map(|group| group.group_id.clone())
            .collect()
    }

    pub fn mark_reservation_releasing(&mut self, group_id: &str, command_id: String) -> bool {
        let Some(reservation) = self
            .groups
            .get_mut(group_id)
            .and_then(|group| group.power_reservation.as_mut())
        else {
            return false;
        };
        if matches!(
            reservation.state,
            ReservationState::Accepted
                | ReservationState::Prewarming
                | ReservationState::Ready
                | ReservationState::Blocked
                | ReservationState::Failed
                | ReservationState::ReleasePending
        ) {
            reservation.state = ReservationState::Releasing;
            reservation.command_id = Some(command_id);
            reservation.command_attempt = reservation.command_attempt.saturating_add(1);
            return true;
        }
        false
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
        // Manual recording commands suppress the *current deterministic group*,
        // not merely the bridge's in-memory copy. Persist the terminal owner
        // transition before acknowledging the feedback so a scheduler restart
        // cannot replay the old scheduled demand.
        if feedback.manual_suppression {
            let action = match state {
                RecordingOccurrenceState::StartPending => ScheduledRecordingIntentAction::Acquire,
                RecordingOccurrenceState::StopPending => ScheduledRecordingIntentAction::Release,
                RecordingOccurrenceState::Active => {
                    // The start outbox was already acknowledged. Terminal recorder
                    // evidence is still authoritative for this live generation.
                    ScheduledRecordingIntentAction::Acquire
                }
                _ => return false,
            };
            let Some(group) = self.groups.get_mut(&group_id) else {
                return false;
            };
            if group.generation != feedback.generation {
                return false;
            }
            if state != RecordingOccurrenceState::Active {
                if scheduled_intent_id(&feedback.occurrence_id, feedback.generation, action)
                    .ok()
                    .as_deref()
                    != Some(&feedback.intent_id)
                    || group.pending_intent_id.as_deref() != Some(&feedback.intent_id)
                    || group.pending_action != Some(action)
                    || !group.accept_feedback(&feedback.intent_id, feedback.generation)
                {
                    return false;
                }
                group.finish_intent();
            }
            group.recording_id = None;
            let now_ms = self.clock.now_ms();
            // The bridge receives only the group first-owner intent, whereas the
            // scheduler owns every overlap member. Suppress all owners atomically
            // so an unobserved later occurrence cannot replay this group.
            let owner_ids = group.owner_ids.iter().cloned().collect::<Vec<_>>();
            for owner_id in &owner_ids {
                group.remove_owner(owner_id);
            }
            let mut changed = false;
            for owner_id in owner_ids {
                let Some(occurrence) = self.occurrences.get_mut(&owner_id) else {
                    continue;
                };
                if occurrence.state.is_terminal() {
                    continue;
                }
                occurrence.suppressed_by_manual = true;
                if let Some(attempt) = occurrence.attempts.last_mut() {
                    attempt.state = RecordingAttemptState::Partial;
                    attempt.ended_at_ms = Some(now_ms);
                }
                let next = if occurrence.state == RecordingOccurrenceState::StopPending {
                    RecordingOccurrenceState::Completed
                } else {
                    RecordingOccurrenceState::Suppressed
                };
                changed |= transition(occurrence, next, now_ms, None);
            }
            return changed;
        }
        // A recorder can fail after its accepted start feedback has already made
        // the occurrence Active. This is not an old start reply: it is terminal
        // recorder evidence for the same live group and must be retained as a
        // partial clip attempt instead of being silently dropped.
        if state == RecordingOccurrenceState::Active
            && feedback.recorder_state == Some(robo_rover_lib::RecordingSessionState::Failed)
            && feedback.group_id.as_deref() == Some(&group_id)
        {
            let Some(recording_id) = feedback.recording_id.as_deref() else {
                return false;
            };
            let Some(group) = self.groups.get_mut(&group_id) else {
                return false;
            };
            if group.generation != feedback.generation {
                return false;
            }
            let Some(occurrence) = self.occurrences.get_mut(&feedback.occurrence_id) else {
                return false;
            };
            let now_ms = self.clock.now_ms();
            if let Some(attempt) = occurrence
                .attempts
                .iter_mut()
                .rev()
                .find(|attempt| attempt.recording_id == recording_id)
            {
                attempt.state = RecordingAttemptState::Partial;
                attempt.ended_at_ms = Some(now_ms);
            } else {
                occurrence.attempts.push(RecordingClipAttempt {
                    recording_id: recording_id.to_owned(),
                    state: RecordingAttemptState::Failed,
                    started_at_ms: now_ms,
                    ended_at_ms: Some(now_ms),
                });
            }
            group.recording_id = None;
            group.finish_intent();
            return transition(
                occurrence,
                RecordingOccurrenceState::Failed,
                now_ms,
                feedback.reason_code,
            );
        }
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
        if feedback.retryable
            && is_transient_recorder_storage_failure(
                feedback.reason_code,
                feedback.detail.as_deref(),
            )
        {
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
