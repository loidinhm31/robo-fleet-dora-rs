#[derive(Debug, Clone)]
pub struct ProjectorConfig {
    pub deployment_id: String,
    pub mongodb_uri: String,
    pub mongodb_database: String,
    pub retry_attempts: u8,
    pub retry_backoff_ms: u64,
}

impl ProjectorConfig {
    pub fn from_env() -> Result<Self, String> {
        let config = Self {
            deployment_id: std::env::var("POWER_DEPLOYMENT_ID")
                .unwrap_or_else(|_| "default".into()),
            mongodb_uri: required("MONGODB_URI")?,
            mongodb_database: required("MONGODB_DATABASE")?,
            retry_attempts: optional_u8("POWER_PROJECTOR_RETRY_ATTEMPTS", 3)?,
            retry_backoff_ms: optional_u64("POWER_PROJECTOR_RETRY_BACKOFF_MS", 100)?,
        };
        (valid_id(&config.deployment_id)
            && !config.mongodb_uri.is_empty()
            && !config.mongodb_database.is_empty()
            && config.retry_attempts > 0
            && config.retry_attempts <= 5
            && config.retry_backoff_ms > 0
            && config.retry_backoff_ms <= 1_000)
            .then_some(config)
            .ok_or_else(|| "invalid power projector configuration".into())
    }
}

fn optional_u8(key: &str, default: u8) -> Result<u8, String> {
    std::env::var(key).map_or(Ok(default), |value| {
        value.parse().map_err(|_| format!("invalid {key}"))
    })
}
fn optional_u64(key: &str, default: u64) -> Result<u64, String> {
    std::env::var(key).map_or(Ok(default), |value| {
        value.parse().map_err(|_| format!("invalid {key}"))
    })
}

fn required(key: &str) -> Result<String, String> {
    std::env::var(key).map_err(|_| format!("missing {key}"))
}
fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}
