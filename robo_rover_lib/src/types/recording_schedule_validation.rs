use std::str::FromStr;

use chrono::{
    DateTime, Duration, LocalResult, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Timelike, Utc,
};
use chrono_tz::Tz;

use super::{
    validate_id, validate_relative_directory, validate_uuid, AuthenticatedRecordingScheduleCommand,
    RecordingLocalStart, RecordingOccurrence, RecordingSchedule, RecordingScheduleAction,
    RecordingScheduleCommand, RecordingScheduleCommandResult, RecordingScheduleDefinition,
    RecordingScheduleQuery, RecordingScheduleRecurrence, RecordingScheduleSnapshot,
    RECORDING_SCHEDULE_PROTOCOL_VERSION,
};

pub const DEFAULT_SCHEDULE_MAX_FUTURE_MS: i64 = 365 * 24 * 60 * 60 * 1_000;
pub const DEFAULT_MAX_ENABLED_SCHEDULES_PER_ROVER: usize = 50;
const MIN_DURATION_MS: i64 = 60_000;
const MAX_DURATION_MS: i64 = 24 * 60 * 60 * 1_000;
const MAX_TITLE_LEN: usize = 128;
const MAX_DETAIL_LEN: usize = 256;
pub const TERMINAL_OCCURRENCE_RETENTION_MS: i64 = 90 * 24 * 60 * 60 * 1_000;

/// Runtime-configurable policy. Defaults are provisional until Phase 2 configuration is approved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordingScheduleValidationLimits {
    pub max_future_ms: i64,
    pub max_enabled_schedules_per_rover: usize,
}

impl Default for RecordingScheduleValidationLimits {
    fn default() -> Self {
        Self {
            max_future_ms: DEFAULT_SCHEDULE_MAX_FUTURE_MS,
            max_enabled_schedules_per_rover: DEFAULT_MAX_ENABLED_SCHEDULES_PER_ROVER,
        }
    }
}

impl RecordingScheduleCommand {
    pub fn validate_at(
        &self,
        now_ms: i64,
        limits: RecordingScheduleValidationLimits,
    ) -> Result<(), String> {
        validate_protocol(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        match &self.action {
            RecordingScheduleAction::Create { schedule } => {
                validate_definition(schedule, now_ms, limits)
            }
            RecordingScheduleAction::Update {
                schedule_id,
                expected_revision,
                schedule,
            } => {
                validate_schedule_id_and_revision(schedule_id, *expected_revision)?;
                validate_definition(schedule, now_ms, limits)
            }
            RecordingScheduleAction::SetEnabled {
                schedule_id,
                expected_revision,
                ..
            }
            | RecordingScheduleAction::Delete {
                schedule_id,
                expected_revision,
            } => validate_schedule_id_and_revision(schedule_id, *expected_revision),
        }
    }
}

impl AuthenticatedRecordingScheduleCommand {
    pub fn validate_at(
        &self,
        now_ms: i64,
        limits: RecordingScheduleValidationLimits,
    ) -> Result<(), String> {
        self.command.validate_at(now_ms, limits)?;
        validate_id("audit_actor", &self.audit_actor)
    }
}

impl RecordingScheduleQuery {
    pub fn validate(&self) -> Result<(), String> {
        validate_protocol(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        validate_id("entity_id", &self.entity_id)
    }
}

impl RecordingSchedule {
    pub fn validate(&self) -> Result<(), String> {
        validate_uuid("schedule_id", &self.schedule_id)?;
        if self.revision == 0 || self.created_at_ms < 0 || self.updated_at_ms < self.created_at_ms {
            return Err("invalid schedule revision or timestamps".into());
        }
        validate_id("created_by", &self.created_by)?;
        validate_id("updated_by", &self.updated_by)?;
        validate_definition(
            &self.definition,
            0,
            RecordingScheduleValidationLimits {
                max_future_ms: i64::MAX,
                max_enabled_schedules_per_rover: usize::MAX,
            },
        )
    }
}

impl RecordingScheduleCommandResult {
    pub fn validate(&self) -> Result<(), String> {
        validate_protocol(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        if self
            .detail
            .as_ref()
            .is_some_and(|detail| detail.len() > MAX_DETAIL_LEN)
        {
            return Err("schedule result detail exceeds 256 characters".into());
        }
        if self.accepted {
            if self.reason_code.is_some() || self.current_schedule.is_some() {
                return Err("accepted schedule result has rejection fields".into());
            }
        } else if self.reason_code.is_none() {
            return Err("rejected schedule result requires reason_code".into());
        }
        self.schedule
            .iter()
            .try_for_each(RecordingSchedule::validate)?;
        self.current_schedule
            .iter()
            .try_for_each(RecordingSchedule::validate)
    }
}

impl RecordingScheduleSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        validate_protocol(self.protocol_version)?;
        validate_uuid("request_id", &self.request_id)?;
        validate_id("entity_id", &self.entity_id)?;
        self.schedules
            .iter()
            .try_for_each(RecordingSchedule::validate)
    }
}

impl RecordingOccurrence {
    pub fn validate(&self) -> Result<(), String> {
        validate_uuid("occurrence_id", &self.occurrence_id)?;
        validate_uuid("schedule_id", &self.schedule_id)?;
        validate_id("entity_id", &self.entity_id)?;
        validate_uuid("start_request_id", &self.start_request_id)?;
        if self.schedule_revision == 0
            || self.planned_start_ms < 0
            || self.planned_end_ms <= self.planned_start_ms
        {
            return Err("invalid occurrence bounds or revision".into());
        }
        if self.state.is_terminal() {
            let terminal_at_ms = self
                .terminal_at_ms
                .ok_or("terminal occurrence requires terminal_at_ms")?;
            let expires_at_ms = self
                .expires_at_ms
                .ok_or("terminal occurrence requires expires_at_ms")?;
            if terminal_at_ms < self.created_at_ms
                || terminal_at_ms > self.updated_at_ms
                || expires_at_ms != terminal_at_ms.saturating_add(TERMINAL_OCCURRENCE_RETENTION_MS)
            {
                return Err("terminal occurrence has invalid retention timestamps".into());
            }
        } else if self.terminal_at_ms.is_some() || self.expires_at_ms.is_some() {
            return Err("nonterminal occurrence cannot have retention timestamps".into());
        }
        if self
            .last_error
            .as_ref()
            .is_some_and(|error| error.detail.len() > MAX_DETAIL_LEN)
        {
            return Err("occurrence error detail exceeds 256 characters".into());
        }
        Ok(())
    }
}

fn validate_definition(
    definition: &RecordingScheduleDefinition,
    now_ms: i64,
    limits: RecordingScheduleValidationLimits,
) -> Result<(), String> {
    validate_id("entity_id", &definition.entity_id)?;
    if definition.title.is_empty() || definition.title.len() > MAX_TITLE_LEN {
        return Err("invalid title".into());
    }
    let local_start = definition.recurrence.local_start();
    let resolved_local_start = resolve_local_schedule_start(local_start)?;
    validate_relative_directory(&definition.relative_directory_template)?;
    if !(MIN_DURATION_MS..=MAX_DURATION_MS).contains(&definition.duration_ms) {
        return Err("duration_ms must be between 60000 and 86400000".into());
    }
    match &definition.recurrence {
        RecordingScheduleRecurrence::OneTime { .. } => {
            validate_one_time_start(resolved_local_start, now_ms, limits.max_future_ms)
        }
        RecordingScheduleRecurrence::Daily { .. } => Ok(()),
        RecordingScheduleRecurrence::Weekly { weekdays, .. } if weekdays.is_empty() => {
            Err("weekly recurrence requires weekdays".into())
        }
        RecordingScheduleRecurrence::Weekly { weekdays, .. }
            if weekdays.windows(2).any(|pair| pair[0] == pair[1]) =>
        {
            Err("weekly recurrence has duplicate weekdays".into())
        }
        RecordingScheduleRecurrence::Weekly { .. } => Ok(()),
    }
}

/// Resolves schedule intent with the frozen DST rule: gap-forward and earlier fold.
pub fn resolve_local_schedule_start(value: &RecordingLocalStart) -> Result<DateTime<Tz>, String> {
    let date =
        NaiveDate::parse_from_str(&value.date, "%Y-%m-%d").map_err(|_| "invalid local date")?;
    let time = NaiveTime::parse_from_str(&value.time, "%H:%M:%S")
        .or_else(|_| NaiveTime::parse_from_str(&value.time, "%H:%M"))
        .map_err(|_| "invalid local time")?;
    let timezone = Tz::from_str(&value.timezone).map_err(|_| "invalid IANA timezone")?;
    resolve_local_datetime(timezone, NaiveDateTime::new(date, time))
}

fn resolve_local_datetime(timezone: Tz, local: NaiveDateTime) -> Result<DateTime<Tz>, String> {
    match timezone.from_local_datetime(&local) {
        LocalResult::Single(resolved) => Ok(resolved),
        LocalResult::Ambiguous(first, second) => Ok(first.min(second)),
        LocalResult::None => {
            let mut candidate = local
                .date()
                .and_hms_opt(local.hour(), local.minute(), 0)
                .ok_or("invalid local time")?;
            for _ in 0..(24 * 60) {
                candidate += Duration::minutes(1);
                match timezone.from_local_datetime(&candidate) {
                    LocalResult::Single(resolved) => return Ok(resolved),
                    LocalResult::Ambiguous(first, second) => return Ok(first.min(second)),
                    LocalResult::None => {}
                }
            }
            Err("unable to resolve local DST gap".into())
        }
    }
}

fn validate_one_time_start(
    start: DateTime<Tz>,
    now_ms: i64,
    max_future_ms: i64,
) -> Result<(), String> {
    if max_future_ms == i64::MAX {
        return Ok(());
    }
    let now =
        DateTime::<Utc>::from_timestamp_millis(now_ms).ok_or("invalid validation timestamp")?;
    let deadline = DateTime::<Utc>::from_timestamp_millis(now_ms.saturating_add(max_future_ms))
        .ok_or("invalid validation deadline")?;
    let start = start.with_timezone(&Utc);
    if start < now || start > deadline {
        return Err("one-time start is outside the allowed future window".into());
    }
    Ok(())
}

fn validate_schedule_id_and_revision(
    schedule_id: &str,
    expected_revision: u64,
) -> Result<(), String> {
    validate_uuid("schedule_id", schedule_id)?;
    (expected_revision > 0)
        .then_some(())
        .ok_or_else(|| "expected_revision must be positive".into())
}

fn validate_protocol(version: u8) -> Result<(), String> {
    (version == RECORDING_SCHEDULE_PROTOCOL_VERSION)
        .then_some(())
        .ok_or_else(|| format!("unsupported recording schedule protocol version: {version}"))
}
