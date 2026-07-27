use robo_rover_lib::LifecycleRole;

pub const MIN_AUTO_IDLE_GRACE_MS: u64 = 300_000;

#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    pub role: LifecycleRole,
    pub entity_id: String,
    pub remote_authority_entity_id: Option<String>,
    pub idle_grace_ms: u64,
    pub min_awake_ms: u64,
    pub resource_freshness_ms: u64,
    pub max_domain_cpu_percent: f32,
    pub required_low_samples: u32,
    pub transition_timeout_ms: u64,
    pub max_transition_retries: u8,
    pub retry_backoff_ms: u64,
    pub journal_dir: String,
    pub journal_max_bytes: u64,
    pub journal_max_records: usize,
    pub journal_wake_reserve_bytes: u64,
    pub journal_wake_reserve_records: usize,
}

impl CoordinatorConfig {
    pub fn from_env() -> Result<Self, String> {
        let role = match required("POWER_COORDINATOR_ROLE")?.as_str() {
            "rover" => LifecycleRole::Rover,
            "orchestra" => LifecycleRole::Orchestra,
            _ => return Err("POWER_COORDINATOR_ROLE must be rover or orchestra".into()),
        };
        let config = Self {
            role,
            entity_id: required("ENTITY_ID")?,
            remote_authority_entity_id: std::env::var("POWER_REMOTE_AUTHORITY_ENTITY_ID")
                .ok()
                .filter(|entity_id| !entity_id.is_empty()),
            idle_grace_ms: number("POWER_IDLE_GRACE_MS", MIN_AUTO_IDLE_GRACE_MS)?,
            min_awake_ms: number("POWER_MIN_AWAKE_MS", 30_000)?,
            resource_freshness_ms: number("POWER_RESOURCE_FRESHNESS_MS", 15_000)?,
            max_domain_cpu_percent: decimal("POWER_MAX_DOMAIN_CPU_PERCENT", 10.0)?,
            required_low_samples: number("POWER_REQUIRED_LOW_SAMPLES", 3)?,
            transition_timeout_ms: number("POWER_TRANSITION_TIMEOUT_MS", 30_000)?,
            max_transition_retries: number("POWER_MAX_TRANSITION_RETRIES", 2)?,
            retry_backoff_ms: number("POWER_RETRY_BACKOFF_MS", 1_000)?,
            journal_dir: std::env::var("POWER_JOURNAL_DIR").unwrap_or_else(|_| {
                format!("/var/lib/robo-fleet/power-journal/{role:?}").to_lowercase()
            }),
            journal_max_bytes: number("POWER_JOURNAL_MAX_BYTES", 16 * 1024 * 1024)?,
            journal_max_records: number("POWER_JOURNAL_MAX_RECORDS", 10_000)?,
            journal_wake_reserve_bytes: number(
                "POWER_JOURNAL_WAKE_RESERVE_BYTES",
                1 * 1024 * 1024,
            )?,
            journal_wake_reserve_records: number("POWER_JOURNAL_WAKE_RESERVE_RECORDS", 10)?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn for_test(role: LifecycleRole, entity_id: &str) -> Self {
        Self {
            role,
            entity_id: entity_id.into(),
            remote_authority_entity_id: None,
            idle_grace_ms: MIN_AUTO_IDLE_GRACE_MS,
            min_awake_ms: 0,
            resource_freshness_ms: 15_000,
            max_domain_cpu_percent: 10.0,
            required_low_samples: 2,
            transition_timeout_ms: 1_000,
            max_transition_retries: 1,
            retry_backoff_ms: 10,
            journal_dir: std::env::temp_dir()
                .join(format!("power-coordinator-test-{}", uuid::Uuid::new_v4()))
                .display()
                .to_string(),
            journal_max_bytes: 1024 * 1024,
            journal_max_records: 100,
            journal_wake_reserve_bytes: 1024,
            journal_wake_reserve_records: 2,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.entity_id.is_empty()
            || self.entity_id.len() > 128
            || self
                .remote_authority_entity_id
                .as_ref()
                .is_some_and(|entity_id| entity_id.len() > 128)
            || self.idle_grace_ms < MIN_AUTO_IDLE_GRACE_MS
            || self.resource_freshness_ms == 0
            || self.required_low_samples == 0
            || self.transition_timeout_ms == 0
            || self.journal_max_bytes == 0
            || self.journal_max_records == 0
            || self.journal_wake_reserve_bytes >= self.journal_max_bytes
            || self.journal_wake_reserve_records >= self.journal_max_records
            || !self.max_domain_cpu_percent.is_finite()
            || !(0.0..=100.0).contains(&self.max_domain_cpu_percent)
        {
            return Err("invalid power coordinator configuration".into());
        }
        Ok(())
    }
}

fn required(key: &str) -> Result<String, String> {
    std::env::var(key).map_err(|_| format!("missing required {key}"))
}
fn number<T: std::str::FromStr>(key: &str, default: T) -> Result<T, String> {
    std::env::var(key).ok().map_or(Ok(default), |value| {
        value.parse().map_err(|_| format!("invalid {key}"))
    })
}
fn decimal(key: &str, default: f32) -> Result<f32, String> {
    number(key, default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_idle_grace_cannot_drop_below_five_minutes() {
        let mut config = CoordinatorConfig::for_test(LifecycleRole::Rover, "rover");
        config.idle_grace_ms = MIN_AUTO_IDLE_GRACE_MS - 1;
        assert!(config.validate().is_err());
    }
}
