use std::collections::{BTreeMap, BTreeSet};

use robo_rover_lib::{
    scheduled_group_id, scheduled_reservation_id, scheduled_start_request_id, RecordingOccurrence,
    RecordingOccurrenceState, RecordingScheduleReasonCode, ScheduledRecordingIntentAction,
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
    /// Scheduler-owned state for a group-scoped future power reservation.
    /// It is stored with the group so recovery can replay admission safely
    /// without turning the scheduler into a lifecycle owner.
    #[serde(default)]
    pub power_reservation: Option<ScheduledPowerReservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReservationState {
    #[default]
    Pending,
    Registering,
    Accepted,
    Prewarming,
    Ready,
    Blocked,
    Failed,
    ReleasePending,
    Releasing,
    Released,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledPowerReservation {
    pub reservation_id: String,
    pub generation: u64,
    pub prewarm_at_ms: i64,
    pub expires_at_ms: i64,
    pub state: ReservationState,
    /// The exact remote command currently awaiting a signed Rover result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_id: Option<String>,
    /// Register acknowledgement retained as the correlation proof used by the
    /// recorder admission gate after the in-flight command id is cleared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub register_ack_command_id: Option<String>,
    /// Monotonic attempt number makes a post-expiry retry a new command while
    /// preserving exact correlation for the command currently in flight.
    #[serde(default)]
    pub command_attempt: u32,
    pub transition_id: Option<String>,
    pub registered_at_ms: Option<i64>,
    /// Prewarm latency starts when the reservation becomes active, never when
    /// it was merely registered ahead of its not-before boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_started_at_ms: Option<i64>,
    pub ready_at_ms: Option<i64>,
    #[serde(default)]
    pub sample_count: usize,
    #[serde(default)]
    pub prewarm_estimate_ms: i64,
    #[serde(default)]
    pub bootstrap_active: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_ready_ms: Option<i64>,
    #[serde(default)]
    pub prewarm_missed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_updated_at_ms: Option<i64>,
    /// Exact coordinator authority observed with reservation-scoped Ready
    /// evidence; aggregate status alone never unlocks recording.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready_authority: Option<robo_rover_lib::PowerAuthority>,
    #[serde(default)]
    pub retired: bool,
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
            power_reservation: None,
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

    pub fn ensure_power_reservation(
        &mut self,
        prewarm_at_ms: i64,
        now_ms: i64,
    ) -> Result<bool, String> {
        let generation = self.generation.max(1);
        let expires_at_ms = self.end_ms;
        // A registered reservation is immutable at the coordinator. Do not
        // "improve" its lead time later from a newer p95 sample: that would
        // turn an idempotent replay into a duplicate-payload mismatch.
        let changed = self.power_reservation.is_none();
        if changed {
            self.power_reservation = Some(ScheduledPowerReservation {
                reservation_id: scheduled_reservation_id(&self.group_id, generation)?,
                generation,
                prewarm_at_ms,
                expires_at_ms,
                state: ReservationState::Pending,
                command_id: None,
                register_ack_command_id: None,
                command_attempt: 0,
                transition_id: None,
                registered_at_ms: None,
                activation_started_at_ms: None,
                ready_at_ms: None,
                sample_count: 0,
                prewarm_estimate_ms: 0,
                bootstrap_active: true,
                actual_ready_ms: None,
                prewarm_missed: false,
                status_updated_at_ms: None,
                ready_authority: None,
                retired: false,
            });
        }
        // A reservation must remain a future demand. A late materializer is
        // allowed to register it, but never extends its fixed expiry window.
        Ok(changed && expires_at_ms > now_ms)
    }

    pub fn reservation_is_ready(&self) -> bool {
        self.power_reservation
            .as_ref()
            .is_some_and(|reservation| reservation.state == ReservationState::Ready)
    }

    pub fn reservation_needs_release(&self) -> bool {
        self.power_reservation.as_ref().is_some_and(|reservation| {
            !matches!(
                reservation.state,
                ReservationState::Pending
                    | ReservationState::Registering
                    | ReservationState::Released
            )
        })
    }
}

/// Only explicitly named recorder/storage faults can retry inside the current
/// occurrence window. Broad `Internal`/`Unavailable` retries hide permanent
/// invalidation and can churn a power reservation after an edit or deletion.
pub fn is_transient_recorder_storage_failure(
    reason: Option<RecordingScheduleReasonCode>,
    detail: Option<&str>,
) -> bool {
    matches!(reason, Some(RecordingScheduleReasonCode::Unavailable))
        && matches!(
            detail,
            Some(
                "recording command queue is full"
                    | "scheduled media demand unavailable"
                    | "recorder stop queue unavailable"
                    | "recorder temporarily unavailable"
                    | "storage temporarily unavailable"
            )
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
