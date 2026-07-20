use std::collections::{BTreeMap, BTreeSet};

use robo_rover_lib::{
    scheduled_group_id, scheduled_start_request_id, RecordingOccurrence, RecordingOccurrenceState,
    RecordingScheduleReasonCode, ScheduledRecordingIntentAction,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingGroup {
    pub group_id: String,
    pub entity_id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    /// Chosen once from the earliest occurrence in the union window. It must
    /// remain available even after that schedule has been deleted.
    pub relative_directory: String,
    /// Stable coordinator key for the one recorder session shared by this group.
    pub start_request_id: String,
    pub recording_id: Option<String>,
    pub pending_intent_id: Option<String>,
    pub pending_action: Option<ScheduledRecordingIntentAction>,
    pub owner_ids: BTreeSet<String>,
    pub generation: u64,
    pub handled_feedback_generations: BTreeMap<String, u64>,
}

impl RecordingGroup {
    pub fn new(
        entity_id: &str,
        start_ms: i64,
        end_ms: i64,
        relative_directory: String,
    ) -> Result<Self, String> {
        Self {
            group_id: scheduled_group_id(entity_id, start_ms)?,
            entity_id: entity_id.to_owned(),
            start_ms,
            end_ms,
            relative_directory,
            start_request_id: String::new(),
            recording_id: None,
            pending_intent_id: None,
            pending_action: None,
            owner_ids: BTreeSet::new(),
            generation: 0,
            handled_feedback_generations: BTreeMap::new(),
        }
        .with_start_request_id()
    }

    fn with_start_request_id(mut self) -> Result<Self, String> {
        self.start_request_id = scheduled_start_request_id(&self.group_id)?;
        Ok(self)
    }

    pub fn add_owner(&mut self, occurrence_id: &str) -> bool {
        let was_empty = self.owner_ids.is_empty();
        let added = self.owner_ids.insert(occurrence_id.to_owned());
        if added && was_empty {
            self.generation = self.generation.saturating_add(1);
        }
        added
    }

    pub fn remove_owner(&mut self, occurrence_id: &str) -> bool {
        let removed = self.owner_ids.remove(occurrence_id);
        if removed && self.owner_ids.is_empty() {
            self.generation = self.generation.saturating_add(1);
        }
        removed
    }

    pub fn retry_acquire(&mut self) {
        self.generation = self.generation.saturating_add(1);
    }

    pub fn retry_release(&mut self) {
        self.generation = self.generation.saturating_add(1);
    }

    pub fn begin_intent(&mut self, intent_id: String, action: ScheduledRecordingIntentAction) {
        self.pending_intent_id = Some(intent_id);
        self.pending_action = Some(action);
    }

    pub fn finish_intent(&mut self) {
        self.pending_intent_id = None;
        self.pending_action = None;
    }

    pub fn overlaps(&self, entity_id: &str, start_ms: i64) -> bool {
        self.entity_id == entity_id && start_ms < self.end_ms
    }

    pub fn accept_feedback(&mut self, intent_id: &str, generation: u64) -> bool {
        if generation != self.generation
            || self.handled_feedback_generations.contains_key(intent_id)
        {
            return false;
        }
        self.handled_feedback_generations
            .insert(intent_id.to_owned(), generation);
        true
    }
}

pub fn is_transient_reason(reason: Option<RecordingScheduleReasonCode>) -> bool {
    matches!(
        reason,
        Some(RecordingScheduleReasonCode::Unavailable | RecordingScheduleReasonCode::Internal)
    )
}

pub fn due_occurrences(occurrences: &[RecordingOccurrence], now_ms: i64) -> Vec<String> {
    occurrences
        .iter()
        .filter(|occurrence| {
            occurrence.state == RecordingOccurrenceState::Planned
                && occurrence.planned_start_ms <= now_ms
        })
        .map(|occurrence| occurrence.occurrence_id.clone())
        .collect()
}
