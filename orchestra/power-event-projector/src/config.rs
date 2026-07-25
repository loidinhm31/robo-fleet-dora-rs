#[derive(Debug, Clone)]
pub struct ProjectorConfig {
    pub deployment_id: String,
    pub mongodb_uri: String,
    pub mongodb_database: String,
}

impl ProjectorConfig {
    pub fn from_env() -> Result<Self, String> {
        let config = Self {
            deployment_id: std::env::var("POWER_DEPLOYMENT_ID")
                .unwrap_or_else(|_| "default".into()),
            mongodb_uri: required("MONGODB_URI")?,
            mongodb_database: required("MONGODB_DATABASE")?,
        };
        (valid_id(&config.deployment_id)
            && !config.mongodb_uri.is_empty()
            && !config.mongodb_database.is_empty())
        .then_some(config)
        .ok_or_else(|| "invalid power projector configuration".into())
    }
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
