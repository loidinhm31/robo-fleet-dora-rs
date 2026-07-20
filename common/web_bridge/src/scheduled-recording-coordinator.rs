use std::collections::{BTreeMap, BTreeSet};

use robo_rover_lib::{
    scheduled_start_request_id, RecordingCoordinatorFeedback, RecordingReconciliationSession,
    RecordingReconciliationSnapshot, RecordingScheduleReasonCode, RecordingSessionAction,
    RecordingSessionCommand, RecordingSessionCommandResult, RecordingSessionState,
    ScheduledRecordingIntent, ScheduledRecordingIntentAction, RECORDING_PROTOCOL_VERSION,
};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorEffect {
    StartScheduled {
        group_id: String,
        generation: u64,
        entity_id: String,
        consumer_id: String,
        planned_end_ms: i64,
        command: RecordingSessionCommand,
    },
    AcquireMedia {
        entity_id: String,
        consumer_id: String,
    },
    ReleaseMedia {
        consumer_id: String,
    },
    RecorderCommand(RecordingSessionCommand),
    ManualRecorderCommand(RecordingSessionCommand),
    Feedback(RecordingCoordinatorFeedback),
    InvariantViolation {
        entity_id: String,
    },
    ManualStartDeferred {
        entity_id: String,
    },
}

#[derive(Debug, Default)]
pub struct ScheduledRecordingCoordinator {
    groups: BTreeMap<String, Group>,
    handled_intents: BTreeSet<String>,
    /// Intents whose recovery feedback could not yet be delivered. Replays must
    /// remain retryable rather than being mistaken for an accepted start.
    retryable_intents: BTreeSet<String>,
    snapshot_barrier_complete: bool,
    snapshot_sessions: Vec<RecordingReconciliationSession>,
    recovered_suppression_stops: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct Group {
    entity_id: String,
    generation: u64,
    start_request_id: String,
    relative_directory: String,
    owners: BTreeMap<String, IntentRef>,
    recording_id: Option<String>,
    demand_acquired: bool,
    suppressed: bool,
    suppressed_until_ms: i64,
    manual_start: Option<RecordingSessionCommand>,
    manual_stop_request_id: Option<String>,
    deferred_manual_stop: Option<RecordingSessionCommand>,
    pending_suppression_acks: BTreeSet<String>,
    suppression_persisted: bool,
    pending_stop: Option<IntentRef>,
}

#[derive(Debug, Clone)]
struct IntentRef {
    intent_id: String,
    occurrence_id: String,
    generation: u64,
    planned_end_ms: i64,
}

impl ScheduledRecordingCoordinator {
    pub fn apply(&mut self, intent: ScheduledRecordingIntent) -> Vec<CoordinatorEffect> {
        if self.retryable_intents.contains(&intent.intent_id) {
            return vec![retryable_feedback(
                &intent_ref(&intent),
                &intent.group_id,
                "previous scheduled start could not be enqueued",
            )];
        }
        if !self.handled_intents.insert(intent.intent_id.clone()) {
            return vec![feedback(&intent, true, true, false, None, None)];
        }
        match intent.action {
            ScheduledRecordingIntentAction::Acquire => self.acquire(intent),
            ScheduledRecordingIntentAction::Release => self.release(intent),
        }
    }

    /// The snapshot is the startup barrier. Desired groups are retained while waiting;
    /// only this method may adopt an existing recorder session or restart a missing one.
    pub fn reconcile_snapshot(
        &mut self,
        snapshot: &RecordingReconciliationSnapshot,
    ) -> Vec<CoordinatorEffect> {
        self.snapshot_barrier_complete = true;
        self.snapshot_sessions = snapshot.sessions.clone();
        let mut effects = Vec::new();
        let active = self
            .snapshot_sessions
            .iter()
            .filter(|session| is_active(session.state))
            .collect::<Vec<_>>();
        let duplicate_entities = active
            .iter()
            .fold(BTreeSet::new(), |mut duplicates, session| {
                if active
                    .iter()
                    .filter(|other| other.entity_id == session.entity_id)
                    .count()
                    > 1
                {
                    duplicates.insert(session.entity_id.clone());
                }
                duplicates
            });
        for entity_id in &duplicate_entities {
            effects.push(CoordinatorEffect::InvariantViolation {
                entity_id: entity_id.clone(),
            });
        }
        for group_id in self.groups.keys().cloned().collect::<Vec<_>>() {
            if self
                .groups
                .get(&group_id)
                .is_some_and(|group| duplicate_entities.contains(&group.entity_id))
            {
                continue;
            }
            effects.extend(self.reconcile_group(&group_id));
        }
        effects
    }

    pub fn recorder_result(
        &mut self,
        result: &RecordingSessionCommandResult,
    ) -> Vec<CoordinatorEffect> {
        let Some((group_id, is_start_result)) = self.groups.iter().find_map(|(id, group)| {
            (group.start_request_id == result.request_id)
                .then(|| (id.clone(), true))
                .or_else(|| {
                    (group.manual_stop_request_id.as_deref() == Some(&result.request_id))
                        .then(|| (id.clone(), false))
                })
        }) else {
            return Vec::new();
        };
        let Some(group) = self.groups.get_mut(&group_id) else {
            return Vec::new();
        };
        if !is_start_result {
            group.manual_stop_request_id = None;
            if !result.accepted && group.suppressed && group.suppression_persisted {
                // The scheduler has already made suppression durable. Do not
                // clear it locally: retry the exact scheduled session until a
                // terminal status releases its generation-scoped demand.
                let Some(recording_id) = group.recording_id.clone() else {
                    return Vec::new();
                };
                let request_id = Uuid::new_v4().to_string();
                group.manual_stop_request_id = Some(request_id.clone());
                return vec![CoordinatorEffect::RecorderCommand(stop_command(
                    &request_id,
                    &recording_id,
                ))];
            }
            return Vec::new();
        }
        if !result.accepted {
            let group = self.groups.remove(&group_id).expect("known group");
            let mut effects = vec![CoordinatorEffect::ReleaseMedia {
                consumer_id: consumer_id(&group_id, group.generation),
            }];
            effects.extend(
                group
                    .owners
                    .values()
                    .map(|owner| failed_feedback(owner, result)),
            );
            if let Some(manual) = group.manual_start {
                effects.push(CoordinatorEffect::ManualRecorderCommand(manual));
            }
            return effects;
        }
        group.recording_id = result.recording_id.clone();
        group.demand_acquired = true;
        let mut effects = group
            .owners
            .values()
            .map(|owner| accepted_feedback(owner, result.recording_id.clone()))
            .collect::<Vec<_>>();
        if group.suppressed && group.suppression_persisted {
            if let Some(recording_id) = result.recording_id.as_deref() {
                let request_id = Uuid::new_v4().to_string();
                group.manual_stop_request_id = Some(request_id.clone());
                effects.push(CoordinatorEffect::RecorderCommand(stop_command(
                    &request_id,
                    recording_id,
                )));
            }
        }
        effects
    }

    pub fn recorder_status(
        &mut self,
        recording_id: &str,
        state: RecordingSessionState,
    ) -> Vec<CoordinatorEffect> {
        if !matches!(
            state,
            RecordingSessionState::Completed | RecordingSessionState::Failed
        ) {
            return Vec::new();
        }
        // A status can race a delayed desired intent after the snapshot barrier.
        // Do not let that later intent adopt a session already known terminal.
        self.snapshot_sessions
            .retain(|session| session.recording_id != recording_id);
        let Some(group_id) = self.groups.iter().find_map(|(id, group)| {
            (group.recording_id.as_deref() == Some(recording_id)).then(|| id.clone())
        }) else {
            return Vec::new();
        };
        let Some(group) = self.groups.get_mut(&group_id) else {
            return Vec::new();
        };
        let mut effects = Vec::new();
        if group.demand_acquired {
            group.demand_acquired = false;
            effects.push(CoordinatorEffect::ReleaseMedia {
                consumer_id: consumer_id(&group_id, group.generation),
            });
        }
        if state == RecordingSessionState::Failed {
            effects.extend(
                group
                    .owners
                    .values()
                    .map(|owner| terminal_failure_feedback(owner, &group_id, recording_id)),
            );
        }
        if group.suppressed && !group.suppression_persisted {
            // If a natural scheduled release was already pending, use that intent
            // so the scheduler can complete its outbox record. Otherwise each
            // active owner is durably transitioned to Suppressed.
            if let Some(stop) = group.pending_stop.as_ref() {
                effects.push(manual_suppression_feedback(stop, &group_id));
            } else {
                effects.extend(
                    group
                        .owners
                        .values()
                        .map(|owner| manual_suppression_feedback(owner, &group_id)),
                );
            }
        }
        group.recording_id = None;
        if let Some(stop) = group.pending_stop.take() {
            if !group.suppressed {
                effects.push(accepted_feedback(&stop, Some(recording_id.to_owned())));
            }
        }
        if group.suppressed {
            if !group.suppression_persisted {
                // The scheduler must durably suppress this deterministic group
                // before a manual transition can be completed or forgotten.
                return effects;
            }
            if let Some(manual) = group.manual_start.take() {
                effects.push(CoordinatorEffect::ManualRecorderCommand(manual));
            }
            // Suppression applies to this deterministic merged group only. The
            // scheduler has made all of its owners terminal, so a later group
            // begins from a clean generation after its own boundary.
            self.groups.remove(&group_id);
            return effects;
        }
        self.groups.remove(&group_id);
        effects
    }

    pub fn scheduled_start_enqueue_failed(
        &mut self,
        group_id: &str,
        generation: u64,
        detail: &str,
    ) -> Vec<CoordinatorEffect> {
        let Some(group) = self.groups.get(group_id) else {
            return Vec::new();
        };
        if group.generation != generation {
            return Vec::new();
        }
        let group = self.groups.remove(group_id).expect("group was checked");
        let mut effects = group
            .owners
            .values()
            .map(|owner| {
                self.retryable_intents.insert(owner.intent_id.clone());
                retryable_feedback(owner, group_id, detail)
            })
            .collect::<Vec<_>>();
        if let Some(manual) = group.manual_start {
            effects.push(CoordinatorEffect::ManualRecorderCommand(manual));
        }
        effects
    }

    pub fn scheduled_start_enqueued(&mut self, group_id: &str, generation: u64) {
        if let Some(group) = self
            .groups
            .get_mut(group_id)
            .filter(|group| group.generation == generation)
        {
            group.demand_acquired = true;
        }
    }

    /// Once suppression is durable, a failed enqueue must keep retrying the
    /// exact scheduled session. Clearing local state would diverge from the
    /// scheduler's already-persisted suppression and strand the recording.
    pub fn recorder_command_enqueue_failed(
        &mut self,
        command: &RecordingSessionCommand,
    ) -> Vec<CoordinatorEffect> {
        let RecordingSessionAction::Stop { recording_id } = &command.action else {
            return Vec::new();
        };
        let Some((group_id, group)) = self
            .groups
            .iter_mut()
            .find(|(_, group)| group.recording_id.as_deref() == Some(recording_id))
        else {
            return Vec::new();
        };
        if group.suppressed && group.suppression_persisted {
            let request_id = Uuid::new_v4().to_string();
            group.manual_stop_request_id = Some(request_id.clone());
            return vec![CoordinatorEffect::RecorderCommand(stop_command(
                &request_id,
                recording_id,
            ))];
        }
        group.suppressed = false;
        group.suppressed_until_ms = 0;
        group.manual_start = None;
        group.manual_stop_request_id = None;
        group.deferred_manual_stop = None;
        group.pending_suppression_acks.clear();
        group.suppression_persisted = false;
        group
            .pending_stop
            .take()
            .into_iter()
            .map(|stop| retryable_feedback(&stop, group_id, "recorder stop queue unavailable"))
            .collect()
    }

    pub fn has_scheduled_entity(&self, entity_id: &str) -> bool {
        self.groups
            .values()
            .any(|group| group.entity_id == entity_id && !group.owners.is_empty())
    }

    pub fn has_current_scheduled_start(
        &self,
        group_id: &str,
        generation: u64,
        request_id: &str,
    ) -> bool {
        self.groups.get(group_id).is_some_and(|group| {
            group.generation == generation
                && group.start_request_id == request_id
                && !group.owners.is_empty()
        })
    }

    pub fn is_durable_manual_stop(&self, command: &RecordingSessionCommand) -> bool {
        matches!(
            &command.action,
            RecordingSessionAction::Stop { recording_id }
                if self.groups.values().any(|group| {
                    group.suppressed
                        && group.suppression_persisted
                        && group.manual_stop_request_id.as_deref() == Some(&command.request_id)
                        && group.recording_id.as_deref() == Some(recording_id)
                })
        )
    }

    /// Call only after the manual request has been admitted. This keeps a rejected
    /// manual request from changing scheduler ownership or suppression state.
    pub fn defer_manual_start(
        &mut self,
        command: RecordingSessionCommand,
    ) -> Option<Vec<CoordinatorEffect>> {
        let RecordingSessionAction::Start { entity_id, .. } = &command.action else {
            return None;
        };
        let (group_id, group) = self
            .groups
            .iter_mut()
            .find(|(_, group)| group.entity_id == *entity_id && !group.owners.is_empty())?;
        group.suppressed = true;
        group.suppressed_until_ms = group
            .owners
            .values()
            .map(|owner| owner.planned_end_ms)
            .max()
            .unwrap_or(group.suppressed_until_ms);
        group.manual_start = Some(command.clone());
        let mut effects = begin_manual_suppression(group, group_id);
        let Some(recording_id) = group.recording_id.clone() else {
            if group.demand_acquired {
                effects.push(CoordinatorEffect::ManualStartDeferred {
                    entity_id: entity_id.clone(),
                });
            } else {
                // There is no recorder session to finalize. The persisted
                // suppression acknowledgement will release this manual start.
            }
            return Some(effects);
        };
        let request_id = Uuid::new_v4().to_string();
        group.manual_stop_request_id = Some(request_id.clone());
        group.deferred_manual_stop = Some(stop_command(&request_id, &recording_id));
        effects.push(CoordinatorEffect::ManualStartDeferred {
            entity_id: entity_id.clone(),
        });
        Some(effects)
    }

    pub fn suppress_manual_stop(
        &mut self,
        command: RecordingSessionCommand,
    ) -> Option<Vec<CoordinatorEffect>> {
        let RecordingSessionAction::Stop { recording_id } = &command.action else {
            return None;
        };
        let (group_id, group) = self.groups.iter_mut().find(|(_, group)| {
            group.recording_id.as_deref() == Some(recording_id) && !group.owners.is_empty()
        })?;
        group.suppressed = true;
        group.suppressed_until_ms = group
            .owners
            .values()
            .map(|owner| owner.planned_end_ms)
            .max()
            .unwrap_or(group.suppressed_until_ms);
        group.manual_stop_request_id = Some(command.request_id.clone());
        group.deferred_manual_stop = Some(command);
        Some(begin_manual_suppression(group, group_id))
    }

    /// Called only after the scheduler has persisted a manual suppression
    /// feedback record and echoed it over Dora. Until then no recorder Stop is
    /// released, closing the bridge-crash replay race.
    pub fn manual_suppression_persisted(
        &mut self,
        feedback: &RecordingCoordinatorFeedback,
    ) -> Vec<CoordinatorEffect> {
        if !feedback.manual_suppression {
            return Vec::new();
        }
        let Some(group_id) = self.groups.iter().find_map(|(group_id, group)| {
            group
                .pending_suppression_acks
                .contains(&feedback.intent_id)
                .then(|| group_id.clone())
        }) else {
            let Some(group_id) = feedback.group_id.as_deref() else {
                return Vec::new();
            };
            let Ok(start_request_id) = scheduled_start_request_id(group_id) else {
                return Vec::new();
            };
            let Some(session) = self.snapshot_sessions.iter().find(|session| {
                is_active(session.state) && session.start_request_id == start_request_id
            }) else {
                return Vec::new();
            };
            if !self
                .recovered_suppression_stops
                .insert(session.recording_id.clone())
            {
                return Vec::new();
            }
            return vec![CoordinatorEffect::RecorderCommand(stop_command(
                &Uuid::new_v4().to_string(),
                &session.recording_id,
            ))];
        };
        let Some(group) = self.groups.get_mut(&group_id) else {
            return Vec::new();
        };
        group.pending_suppression_acks.remove(&feedback.intent_id);
        if !group.pending_suppression_acks.is_empty() {
            return Vec::new();
        }
        group.suppression_persisted = true;
        if let Some(stop) = group.deferred_manual_stop.take() {
            return vec![CoordinatorEffect::RecorderCommand(stop)];
        }
        if group.recording_id.is_none() && !group.demand_acquired {
            if let Some(manual) = group.manual_start.take() {
                self.groups.remove(&group_id);
                return vec![CoordinatorEffect::ManualRecorderCommand(manual)];
            }
        }
        Vec::new()
    }

    fn acquire(&mut self, intent: ScheduledRecordingIntent) -> Vec<CoordinatorEffect> {
        let owner = intent_ref(&intent);
        if let Some(group) = self.groups.get_mut(&intent.group_id) {
            if group.suppressed
                && intent.generation > group.generation
                && intent.planned_start_ms < group.suppressed_until_ms
            {
                group.generation = intent.generation;
                group.owners.clear();
                group.owners.insert(owner.occurrence_id.clone(), owner);
                return vec![manual_suppression_feedback(
                    group.owners.values().next().expect("inserted owner"),
                    &intent.group_id,
                )];
            }
            if group.generation != intent.generation || group.entity_id != intent.entity_id {
                return vec![feedback(
                    &intent,
                    false,
                    false,
                    false,
                    Some(RecordingScheduleReasonCode::Conflict),
                    Some("stale group generation"),
                )];
            }
            if group
                .owners
                .insert(owner.occurrence_id.clone(), owner)
                .is_some()
            {
                return vec![feedback(&intent, true, true, false, None, None)];
            }
            group.suppressed_until_ms = group.suppressed_until_ms.max(intent.planned_end_ms);
            return Vec::new();
        }
        let group = Group {
            entity_id: intent.entity_id,
            generation: intent.generation,
            start_request_id: intent.start_request_id,
            relative_directory: intent.relative_directory,
            owners: BTreeMap::from([(owner.occurrence_id.clone(), owner)]),
            recording_id: None,
            demand_acquired: false,
            suppressed: false,
            suppressed_until_ms: intent.planned_end_ms,
            manual_start: None,
            manual_stop_request_id: None,
            deferred_manual_stop: None,
            pending_suppression_acks: BTreeSet::new(),
            suppression_persisted: false,
            pending_stop: None,
        };
        let group_id = intent.group_id;
        self.groups.insert(group_id.clone(), group);
        if self.snapshot_barrier_complete {
            self.reconcile_group(&group_id)
        } else {
            Vec::new()
        }
    }

    fn release(&mut self, intent: ScheduledRecordingIntent) -> Vec<CoordinatorEffect> {
        let Some(group) = self.groups.get_mut(&intent.group_id) else {
            return vec![feedback(&intent, true, true, false, None, None)];
        };
        if group.generation != intent.generation {
            return vec![feedback(
                &intent,
                false,
                false,
                false,
                Some(RecordingScheduleReasonCode::Conflict),
                Some("stale group generation"),
            )];
        }
        group.owners.remove(&intent.occurrence_id);
        if !group.owners.is_empty() {
            return vec![feedback(&intent, true, true, false, None, None)];
        }
        if group.suppressed && group.recording_id.is_some() {
            // A manual replacement already issued an exact stop. Keep the group
            // through terminal status so the deferred manual start cannot race a
            // natural scheduled boundary.
            group.pending_stop = Some(intent_ref(&intent));
            return Vec::new();
        }
        if group.suppressed || group.recording_id.is_none() {
            let group = self.groups.remove(&intent.group_id).expect("known group");
            let mut effects = Vec::new();
            if group.demand_acquired {
                effects.push(CoordinatorEffect::ReleaseMedia {
                    consumer_id: consumer_id(&intent.group_id, group.generation),
                });
            }
            effects.push(accepted_feedback(&intent_ref(&intent), None));
            return effects;
        }
        group.pending_stop = Some(intent_ref(&intent));
        vec![CoordinatorEffect::RecorderCommand(stop_command(
            &intent.intent_id,
            group.recording_id.as_deref().expect("recording checked"),
        ))]
    }

    fn reconcile_group(&mut self, group_id: &str) -> Vec<CoordinatorEffect> {
        let Some(group) = self.groups.get_mut(group_id) else {
            return Vec::new();
        };
        if group.owners.is_empty() || group.suppressed {
            return Vec::new();
        }
        if let Some(session) = self.snapshot_sessions.iter().find(|session| {
            is_active(session.state)
                && session.entity_id == group.entity_id
                && session.start_request_id == group.start_request_id
        }) {
            group.recording_id = Some(session.recording_id.clone());
            if !group.demand_acquired {
                group.demand_acquired = true;
                return vec![CoordinatorEffect::AcquireMedia {
                    entity_id: group.entity_id.clone(),
                    consumer_id: consumer_id(group_id, group.generation),
                }];
            }
            return Vec::new();
        }
        if group.recording_id.is_none() && !group.demand_acquired {
            return vec![start_effect(group_id, group)];
        }
        Vec::new()
    }
}

fn start_effect(group_id: &str, group: &Group) -> CoordinatorEffect {
    CoordinatorEffect::StartScheduled {
        group_id: group_id.to_owned(),
        generation: group.generation,
        entity_id: group.entity_id.clone(),
        consumer_id: consumer_id(group_id, group.generation),
        planned_end_ms: group
            .owners
            .values()
            .map(|owner| owner.planned_end_ms)
            .max()
            .unwrap_or_default(),
        command: RecordingSessionCommand {
            protocol_version: RECORDING_PROTOCOL_VERSION,
            request_id: group.start_request_id.clone(),
            action: RecordingSessionAction::Start {
                entity_id: group.entity_id.clone(),
                relative_directory: group.relative_directory.clone(),
            },
        },
    }
}

fn consumer_id(group_id: &str, generation: u64) -> String {
    format!("scheduled:{group_id}:{generation}")
}

fn intent_ref(intent: &ScheduledRecordingIntent) -> IntentRef {
    IntentRef {
        intent_id: intent.intent_id.clone(),
        occurrence_id: intent.occurrence_id.clone(),
        generation: intent.generation,
        planned_end_ms: intent.planned_end_ms,
    }
}

fn begin_manual_suppression(group: &mut Group, group_id: &str) -> Vec<CoordinatorEffect> {
    group.suppression_persisted = false;
    group.pending_suppression_acks = group
        .owners
        .values()
        .map(|owner| owner.intent_id.clone())
        .collect();
    group
        .owners
        .values()
        .map(|owner| manual_suppression_feedback(owner, group_id))
        .collect()
}

fn stop_command(request_id: &str, recording_id: &str) -> RecordingSessionCommand {
    RecordingSessionCommand {
        protocol_version: RECORDING_PROTOCOL_VERSION,
        request_id: request_id.to_owned(),
        action: RecordingSessionAction::Stop {
            recording_id: recording_id.to_owned(),
        },
    }
}

fn accepted_feedback(intent: &IntentRef, recording_id: Option<String>) -> CoordinatorEffect {
    CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
        intent_id: intent.intent_id.clone(),
        occurrence_id: intent.occurrence_id.clone(),
        generation: intent.generation,
        accepted: true,
        applied: true,
        retryable: false,
        group_id: None,
        recording_id,
        recorder_state: Some(RecordingSessionState::Recording),
        manual_suppression: false,
        reason_code: None,
        detail: None,
    })
}

fn failed_feedback(
    intent: &IntentRef,
    result: &RecordingSessionCommandResult,
) -> CoordinatorEffect {
    CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
        intent_id: intent.intent_id.clone(),
        occurrence_id: intent.occurrence_id.clone(),
        generation: intent.generation,
        accepted: false,
        applied: false,
        retryable: true,
        group_id: None,
        recording_id: None,
        recorder_state: Some(RecordingSessionState::Failed),
        manual_suppression: false,
        reason_code: Some(RecordingScheduleReasonCode::Unavailable),
        detail: result.detail.clone(),
    })
}

fn retryable_feedback(intent: &IntentRef, group_id: &str, detail: &str) -> CoordinatorEffect {
    CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
        intent_id: intent.intent_id.clone(),
        occurrence_id: intent.occurrence_id.clone(),
        generation: intent.generation,
        accepted: false,
        applied: false,
        retryable: true,
        group_id: Some(group_id.to_owned()),
        recording_id: None,
        recorder_state: Some(RecordingSessionState::Failed),
        manual_suppression: false,
        reason_code: Some(RecordingScheduleReasonCode::Unavailable),
        detail: Some(detail.to_owned()),
    })
}

fn terminal_failure_feedback(
    intent: &IntentRef,
    group_id: &str,
    recording_id: &str,
) -> CoordinatorEffect {
    CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
        intent_id: intent.intent_id.clone(),
        occurrence_id: intent.occurrence_id.clone(),
        generation: intent.generation,
        accepted: false,
        applied: false,
        retryable: false,
        group_id: Some(group_id.to_owned()),
        recording_id: Some(recording_id.to_owned()),
        recorder_state: Some(RecordingSessionState::Failed),
        manual_suppression: false,
        reason_code: Some(RecordingScheduleReasonCode::Unavailable),
        detail: Some("recorder terminated scheduled session".into()),
    })
}

fn manual_suppression_feedback(intent: &IntentRef, group_id: &str) -> CoordinatorEffect {
    CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
        intent_id: intent.intent_id.clone(),
        occurrence_id: intent.occurrence_id.clone(),
        generation: intent.generation,
        accepted: true,
        applied: true,
        retryable: false,
        group_id: Some(group_id.to_owned()),
        recording_id: None,
        recorder_state: Some(RecordingSessionState::Completed),
        manual_suppression: true,
        reason_code: None,
        detail: Some("scheduled group suppressed by manual recording command".into()),
    })
}

fn feedback(
    intent: &ScheduledRecordingIntent,
    accepted: bool,
    applied: bool,
    retryable: bool,
    reason: Option<RecordingScheduleReasonCode>,
    detail: Option<&str>,
) -> CoordinatorEffect {
    CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
        intent_id: intent.intent_id.clone(),
        occurrence_id: intent.occurrence_id.clone(),
        generation: intent.generation,
        accepted,
        applied,
        retryable,
        group_id: Some(intent.group_id.clone()),
        recording_id: None,
        recorder_state: None,
        manual_suppression: false,
        reason_code: reason,
        detail: detail.map(str::to_owned),
    })
}

fn is_active(state: RecordingSessionState) -> bool {
    matches!(
        state,
        RecordingSessionState::Starting
            | RecordingSessionState::Recording
            | RecordingSessionState::Stopping
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn id() -> String {
        Uuid::new_v4().to_string()
    }
    fn intent(
        action: ScheduledRecordingIntentAction,
        group_id: &str,
        generation: u64,
        entity: &str,
    ) -> ScheduledRecordingIntent {
        ScheduledRecordingIntent {
            intent_id: id(),
            occurrence_id: id(),
            group_id: group_id.into(),
            generation,
            entity_id: entity.into(),
            start_request_id: id(),
            planned_start_ms: 1,
            planned_end_ms: 2,
            relative_directory: "scheduled/a".into(),
            action,
        }
    }
    fn snapshot(
        sessions: Vec<robo_rover_lib::RecordingReconciliationSession>,
    ) -> RecordingReconciliationSnapshot {
        RecordingReconciliationSnapshot {
            request_id: id(),
            sessions,
        }
    }

    fn suppression_feedback(effects: &[CoordinatorEffect]) -> RecordingCoordinatorFeedback {
        effects
            .iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::Feedback(feedback) if feedback.manual_suppression => {
                    Some(feedback.clone())
                }
                _ => None,
            })
            .expect("manual suppression feedback")
    }

    #[test]
    fn waits_for_snapshot_then_adopts_without_duplicate_start() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        let mut scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        scheduled.start_request_id = scheduled_start_request_id(&scheduled.group_id).unwrap();
        assert!(coordinator.apply(scheduled.clone()).is_empty());
        let effects = coordinator.reconcile_snapshot(&snapshot(vec![
            robo_rover_lib::RecordingReconciliationSession {
                entity_id: "rover-a".into(),
                start_request_id: scheduled.start_request_id,
                recording_id: id(),
                state: RecordingSessionState::Recording,
            },
        ]));
        assert!(matches!(
            effects.as_slice(),
            [CoordinatorEffect::AcquireMedia { .. }]
        ));
    }

    #[test]
    fn restart_after_snapshot_and_stale_generation_are_safe() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 3, "rover-a");
        assert!(matches!(
            coordinator.apply(scheduled.clone()).as_slice(),
            [CoordinatorEffect::StartScheduled { .. }]
        ));
        let stale = ScheduledRecordingIntent {
            generation: 2,
            intent_id: id(),
            ..scheduled
        };
        assert!(matches!(
            coordinator.apply(stale).as_slice(),
            [CoordinatorEffect::Feedback(_)]
        ));
    }

    #[test]
    fn terminal_status_after_snapshot_prevents_delayed_intent_from_adopting_a_dead_session() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        let recording_id = id();
        coordinator.reconcile_snapshot(&snapshot(vec![
            robo_rover_lib::RecordingReconciliationSession {
                entity_id: scheduled.entity_id.clone(),
                start_request_id: scheduled.start_request_id.clone(),
                recording_id: recording_id.clone(),
                state: RecordingSessionState::Recording,
            },
        ]));
        assert!(coordinator
            .recorder_status(&recording_id, RecordingSessionState::Completed)
            .is_empty());
        assert!(matches!(
            coordinator.apply(scheduled).as_slice(),
            [CoordinatorEffect::StartScheduled { .. }]
        ));
    }

    #[test]
    fn replayed_manual_suppression_ack_stops_only_the_matching_recovered_session() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        let mut scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        scheduled.start_request_id = scheduled_start_request_id(&scheduled.group_id).unwrap();
        let recording_id = id();
        coordinator.reconcile_snapshot(&snapshot(vec![
            robo_rover_lib::RecordingReconciliationSession {
                entity_id: scheduled.entity_id.clone(),
                start_request_id: scheduled.start_request_id.clone(),
                recording_id: recording_id.clone(),
                state: RecordingSessionState::Recording,
            },
        ]));
        let feedback = RecordingCoordinatorFeedback {
            intent_id: id(),
            occurrence_id: scheduled.occurrence_id,
            generation: scheduled.generation,
            accepted: true,
            applied: true,
            retryable: false,
            group_id: Some(scheduled.group_id),
            recording_id: None,
            recorder_state: None,
            manual_suppression: true,
            reason_code: None,
            detail: None,
        };
        assert!(matches!(
            coordinator.manual_suppression_persisted(&feedback).as_slice(),
            [CoordinatorEffect::RecorderCommand(RecordingSessionCommand {
                action: RecordingSessionAction::Stop { recording_id: stopped },
                ..
            })] if stopped == &recording_id
        ));
        assert!(coordinator
            .manual_suppression_persisted(&feedback)
            .is_empty());
    }

    #[test]
    fn duplicate_and_reordered_intents_never_start_a_second_session() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 4, "rover-a");
        assert!(matches!(
            coordinator.apply(scheduled.clone()).as_slice(),
            [CoordinatorEffect::StartScheduled { .. }]
        ));
        assert!(matches!(
            coordinator.apply(scheduled.clone()).as_slice(),
            [CoordinatorEffect::Feedback(_)]
        ));
        let release = ScheduledRecordingIntent {
            intent_id: id(),
            action: ScheduledRecordingIntentAction::Release,
            ..scheduled.clone()
        };
        assert!(matches!(
            coordinator.apply(release).as_slice(),
            [CoordinatorEffect::Feedback(_)]
        ));
        assert!(matches!(
            coordinator.apply(scheduled).as_slice(),
            [CoordinatorEffect::Feedback(_)]
        ));
    }

    #[test]
    fn inactive_rover_failure_is_bounded_and_retryable() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(
            ScheduledRecordingIntentAction::Acquire,
            &id(),
            1,
            "rover-offline",
        );
        coordinator.apply(scheduled.clone());
        let effects = coordinator.scheduled_start_enqueue_failed(
            &scheduled.group_id,
            scheduled.generation,
            "target rover is not active",
        );
        assert!(matches!(
            effects.as_slice(),
            [CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
                retryable: true,
                reason_code: Some(RecordingScheduleReasonCode::Unavailable),
                ..
            })]
        ));
    }

    #[test]
    fn enqueue_failure_resets_group_for_retry_generation() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        coordinator.apply(scheduled.clone());
        assert!(matches!(
            coordinator
                .scheduled_start_enqueue_failed(&scheduled.group_id, 1, "queue full")
                .as_slice(),
            [CoordinatorEffect::Feedback(_)]
        ));
        assert!(matches!(
            coordinator.apply(scheduled.clone()).as_slice(),
            [CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
                retryable: true,
                ..
            })]
        ));
        let retry = ScheduledRecordingIntent {
            intent_id: id(),
            generation: 2,
            ..scheduled
        };
        assert!(matches!(
            coordinator.apply(retry).as_slice(),
            [CoordinatorEffect::StartScheduled { .. }]
        ));
    }

    #[test]
    fn manual_start_waits_for_terminal_scheduled_status() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        coordinator.apply(scheduled.clone());
        let recording_id = id();
        coordinator.recorder_result(&RecordingSessionCommandResult {
            protocol_version: 1,
            request_id: scheduled.start_request_id.clone(),
            accepted: true,
            recording_id: Some(recording_id.clone()),
            reason_code: None,
            detail: None,
        });
        let manual = RecordingSessionCommand {
            protocol_version: 1,
            request_id: id(),
            action: RecordingSessionAction::Start {
                entity_id: "rover-a".into(),
                relative_directory: "manual/a".into(),
            },
        };
        let suppression = coordinator.defer_manual_start(manual.clone()).unwrap();
        assert!(matches!(
            suppression.as_slice(),
            [
                CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
                    manual_suppression: true,
                    ..
                }),
                CoordinatorEffect::ManualStartDeferred { .. }
            ]
        ));
        assert!(matches!(
            coordinator
                .manual_suppression_persisted(&suppression_feedback(&suppression))
                .as_slice(),
            [CoordinatorEffect::RecorderCommand(_)]
        ));
        let release = ScheduledRecordingIntent {
            intent_id: id(),
            action: ScheduledRecordingIntentAction::Release,
            ..scheduled.clone()
        };
        assert!(coordinator.apply(release).is_empty());
        let effects = coordinator.recorder_status(&recording_id, RecordingSessionState::Completed);
        assert!(
            matches!(effects.as_slice(), [CoordinatorEffect::ReleaseMedia { .. }, CoordinatorEffect::ManualRecorderCommand(command)] if command == &manual)
        );
    }

    #[test]
    fn manual_stop_result_uses_a_distinct_request_and_cannot_reopen_scheduled_start() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        coordinator.apply(scheduled.clone());
        let recording_id = id();
        coordinator.recorder_result(&RecordingSessionCommandResult {
            protocol_version: 1,
            request_id: scheduled.start_request_id.clone(),
            accepted: true,
            recording_id: Some(recording_id.clone()),
            reason_code: None,
            detail: None,
        });
        let effects = coordinator
            .defer_manual_start(RecordingSessionCommand {
                protocol_version: 1,
                request_id: id(),
                action: RecordingSessionAction::Start {
                    entity_id: "rover-a".into(),
                    relative_directory: "manual/a".into(),
                },
            })
            .unwrap();
        let stop = coordinator
            .manual_suppression_persisted(&suppression_feedback(&effects))
            .into_iter()
            .into_iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::RecorderCommand(command) => Some(command),
                _ => None,
            })
            .unwrap();
        assert_ne!(stop.request_id, scheduled.start_request_id);
        let retry = coordinator
            .recorder_result(&RecordingSessionCommandResult {
                protocol_version: 1,
                request_id: stop.request_id.clone(),
                accepted: false,
                recording_id: None,
                reason_code: None,
                detail: Some("queue unavailable".into()),
            })
            .into_iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::RecorderCommand(command) => Some(command),
                _ => None,
            })
            .expect("durably suppressed exact stop is retried");
        assert_ne!(retry.request_id, stop.request_id);
        assert!(
            matches!(&retry.action, RecordingSessionAction::Stop { recording_id: id } if id == &recording_id)
        );
        assert!(coordinator
            .recorder_result(&RecordingSessionCommandResult {
                protocol_version: 1,
                request_id: retry.request_id,
                accepted: true,
                recording_id: Some(recording_id),
                reason_code: None,
                detail: None,
            })
            .is_empty());
    }

    #[test]
    fn rejected_manual_stop_retries_exact_scheduled_session_after_durable_suppression() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        coordinator.apply(scheduled.clone());
        let recording_id = id();
        coordinator.recorder_result(&RecordingSessionCommandResult {
            protocol_version: 1,
            request_id: scheduled.start_request_id.clone(),
            accepted: true,
            recording_id: Some(recording_id.clone()),
            reason_code: None,
            detail: None,
        });
        let manual = RecordingSessionCommand {
            protocol_version: 1,
            request_id: id(),
            action: RecordingSessionAction::Start {
                entity_id: "rover-a".into(),
                relative_directory: "manual/a".into(),
            },
        };
        let suppression = coordinator.defer_manual_start(manual.clone()).unwrap();
        let stop = coordinator
            .manual_suppression_persisted(&suppression_feedback(&suppression))
            .into_iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::RecorderCommand(command) => Some(command),
                _ => None,
            })
            .unwrap();
        let retry = coordinator.recorder_command_enqueue_failed(&stop);
        assert!(
            matches!(retry.as_slice(), [CoordinatorEffect::RecorderCommand(command)] if command.request_id != stop.request_id && matches!(&command.action, RecordingSessionAction::Stop { recording_id: id } if id == &recording_id))
        );
        assert!(matches!(
            coordinator
                .recorder_status(&recording_id, RecordingSessionState::Completed)
                .as_slice(),
            [CoordinatorEffect::ReleaseMedia { .. }, CoordinatorEffect::ManualRecorderCommand(command)] if command == &manual
        ));
    }

    #[test]
    fn release_invalidates_a_queued_scheduled_start_generation() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        coordinator.apply(scheduled.clone());
        assert!(coordinator.has_current_scheduled_start(
            &scheduled.group_id,
            scheduled.generation,
            &scheduled.start_request_id
        ));
        coordinator.apply(ScheduledRecordingIntent {
            intent_id: id(),
            action: ScheduledRecordingIntentAction::Release,
            ..scheduled.clone()
        });
        assert!(!coordinator.has_current_scheduled_start(
            &scheduled.group_id,
            scheduled.generation,
            &scheduled.start_request_id
        ));
    }

    #[test]
    fn manual_stop_suppresses_the_exact_scheduled_recording_until_release_boundary() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        coordinator.apply(scheduled.clone());
        let recording_id = id();
        coordinator.recorder_result(&RecordingSessionCommandResult {
            protocol_version: 1,
            request_id: scheduled.start_request_id.clone(),
            accepted: true,
            recording_id: Some(recording_id.clone()),
            reason_code: None,
            detail: None,
        });
        let manual_stop = RecordingSessionCommand {
            protocol_version: 1,
            request_id: id(),
            action: RecordingSessionAction::Stop {
                recording_id: recording_id.clone(),
            },
        };
        let suppression = coordinator
            .suppress_manual_stop(manual_stop.clone())
            .unwrap();
        assert!(matches!(
            suppression.as_slice(),
            [CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
                manual_suppression: true,
                ..
            })]
        ));
        assert!(matches!(
            coordinator
                .manual_suppression_persisted(&suppression_feedback(&suppression))
                .as_slice(),
            [CoordinatorEffect::RecorderCommand(command)] if command == &manual_stop
        ));
        assert!(matches!(
            coordinator
                .recorder_status(&recording_id, RecordingSessionState::Completed)
                .as_slice(),
            [CoordinatorEffect::ReleaseMedia { .. }]
        ));
        let release = ScheduledRecordingIntent {
            intent_id: id(),
            action: ScheduledRecordingIntentAction::Release,
            ..scheduled
        };
        assert!(matches!(
            coordinator.apply(release).as_slice(),
            [CoordinatorEffect::Feedback(_)]
        ));
    }

    #[test]
    fn rejected_scheduled_start_releases_deferred_manual_start() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        coordinator.reconcile_snapshot(&snapshot(vec![]));
        let scheduled = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        coordinator.apply(scheduled.clone());
        coordinator.scheduled_start_enqueued(&scheduled.group_id, scheduled.generation);
        let manual = RecordingSessionCommand {
            protocol_version: 1,
            request_id: id(),
            action: RecordingSessionAction::Start {
                entity_id: "rover-a".into(),
                relative_directory: "manual/a".into(),
            },
        };
        let suppression = coordinator.defer_manual_start(manual.clone()).unwrap();
        assert!(matches!(
            suppression.as_slice(),
            [
                CoordinatorEffect::Feedback(RecordingCoordinatorFeedback {
                    manual_suppression: true,
                    ..
                }),
                CoordinatorEffect::ManualStartDeferred { .. }
            ]
        ));
        assert!(coordinator
            .manual_suppression_persisted(&suppression_feedback(&suppression))
            .is_empty());
        let effects = coordinator.recorder_result(&RecordingSessionCommandResult {
            protocol_version: 1,
            request_id: scheduled.start_request_id,
            accepted: false,
            recording_id: None,
            reason_code: None,
            detail: Some("recorder unavailable".into()),
        });
        assert!(matches!(
            effects.first(),
            Some(CoordinatorEffect::ReleaseMedia { .. })
        ));
        assert!(effects
            .iter()
            .any(|effect| matches!(effect, CoordinatorEffect::ManualRecorderCommand(command) if command == &manual)));
    }

    #[test]
    fn separate_rovers_reconcile_independently() {
        let mut coordinator = ScheduledRecordingCoordinator::default();
        let a = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-a");
        let b = intent(ScheduledRecordingIntentAction::Acquire, &id(), 1, "rover-b");
        coordinator.apply(a.clone());
        coordinator.apply(b.clone());
        let effects = coordinator.reconcile_snapshot(&snapshot(vec![
            robo_rover_lib::RecordingReconciliationSession {
                entity_id: "rover-a".into(),
                start_request_id: a.start_request_id,
                recording_id: id(),
                state: RecordingSessionState::Recording,
            },
        ]));
        assert_eq!(
            effects
                .iter()
                .filter(|effect| matches!(effect, CoordinatorEffect::AcquireMedia { .. }))
                .count(),
            1
        );
        assert_eq!(
            effects
                .iter()
                .filter(|effect| matches!(effect, CoordinatorEffect::StartScheduled { .. }))
                .count(),
            1
        );
    }
}
