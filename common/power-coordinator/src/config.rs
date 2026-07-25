use robo_rover_lib::LifecycleRole;

#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    pub role: LifecycleRole,
    pub entity_id: String,
    pub idle_grace_ms: u64,
    pub min_awake_ms: u64,
    pub resource_freshness_ms: u64,
    pub max_domain_cpu_percent: f32,
    pub required_low_samples: u32,
    pub transition_timeout_ms: u64,
    pub max_transition_retries: u8,
    pub retry_backoff_ms: u64,
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
            idle_grace_ms: number("POWER_IDLE_GRACE_MS", 300_000)?,
            min_awake_ms: number("POWER_MIN_AWAKE_MS", 30_000)?,
            resource_freshness_ms: number("POWER_RESOURCE_FRESHNESS_MS", 15_000)?,
            max_domain_cpu_percent: decimal("POWER_MAX_DOMAIN_CPU_PERCENT", 10.0)?,
            required_low_samples: number("POWER_REQUIRED_LOW_SAMPLES", 3)?,
            transition_timeout_ms: number("POWER_TRANSITION_TIMEOUT_MS", 30_000)?,
            max_transition_retries: number("POWER_MAX_TRANSITION_RETRIES", 2)?,
            retry_backoff_ms: number("POWER_RETRY_BACKOFF_MS", 1_000)?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn for_test(role: LifecycleRole, entity_id: &str) -> Self {
        Self {
            role,
            entity_id: entity_id.into(),
            idle_grace_ms: 300_000,
            min_awake_ms: 0,
            resource_freshness_ms: 15_000,
            max_domain_cpu_percent: 10.0,
            required_low_samples: 2,
            transition_timeout_ms: 1_000,
            max_transition_retries: 1,
            retry_backoff_ms: 10,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.entity_id.is_empty()
            || self.entity_id.len() > 128
            || self.idle_grace_ms == 0
            || self.resource_freshness_ms == 0
            || self.required_low_samples == 0
            || self.transition_timeout_ms == 0
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
