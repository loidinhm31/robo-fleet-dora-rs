use std::env;

use robo_rover_lib::RecordingScheduleValidationLimits;

pub const DEFAULT_HORIZON_DAYS: i64 = 35;
pub const DEFAULT_MAX_FUTURE_DAYS: i64 = 365;
pub const DEFAULT_MAX_SCHEDULES_PER_ENTITY: usize = 100;
pub const DEFAULT_RECONCILE_SECONDS: u64 = 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub mongodb_uri: String,
    pub mongodb_database: String,
    pub horizon_days: i64,
    pub max_future_days: i64,
    pub max_schedules_per_entity: usize,
    pub reconcile_seconds: u64,
}

impl SchedulerConfig {
    pub fn from_env() -> Result<Self, String> {
        let config = Self {
            mongodb_uri: required("MONGODB_URI")?,
            mongodb_database: required("MONGODB_DATABASE")?,
            horizon_days: parse("RECORDING_SCHEDULER_HORIZON_DAYS", DEFAULT_HORIZON_DAYS)?,
            max_future_days: parse(
                "RECORDING_SCHEDULER_MAX_FUTURE_DAYS",
                DEFAULT_MAX_FUTURE_DAYS,
            )?,
            max_schedules_per_entity: parse(
                "RECORDING_SCHEDULER_MAX_SCHEDULES_PER_ENTITY",
                DEFAULT_MAX_SCHEDULES_PER_ENTITY,
            )?,
            reconcile_seconds: parse(
                "RECORDING_SCHEDULER_RECONCILE_SECONDS",
                DEFAULT_RECONCILE_SECONDS,
            )?,
        };
        if !(7..=366).contains(&config.horizon_days)
            || !(7..=366).contains(&config.max_future_days)
            || config.max_schedules_per_entity == 0
            || config.reconcile_seconds == 0
        {
            return Err("scheduler limits are out of range".into());
        }
        Ok(config)
    }

    pub fn validation_limits(&self) -> RecordingScheduleValidationLimits {
        RecordingScheduleValidationLimits {
            max_future_ms: self.max_future_days * 24 * 60 * 60 * 1_000,
            max_enabled_schedules_per_rover: self.max_schedules_per_entity,
        }
    }
}

fn required(key: &str) -> Result<String, String> {
    env::var(key).map_err(|_| format!("{key} is required"))
}

fn parse<T>(key: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
{
    env::var(key).map_or(Ok(default), |value| {
        value.parse().map_err(|_| format!("invalid {key}"))
    })
}
