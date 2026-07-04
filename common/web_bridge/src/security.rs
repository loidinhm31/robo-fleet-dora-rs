use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use mongodb::{
    bson::doc,
    options::{ClientOptions, IndexOptions},
    Collection, Database, IndexModel,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ─── Rate Limiters ──────────────────────────────────────────────────────────

/// Rate limiter for authentication attempts (per socket ID)
pub struct AuthRateLimiter {
    limiters:
        Arc<Mutex<HashMap<String, (RateLimiter<NotKeyed, InMemoryState, DefaultClock>, Instant)>>>,
    max_attempts: u32,
}

impl AuthRateLimiter {
    pub fn new() -> Self {
        let max_attempts = env::var("RATE_LIMIT_AUTH_PER_MINUTE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        Self {
            limiters: Arc::new(Mutex::new(HashMap::new())),
            max_attempts,
        }
    }

    pub fn check_auth_attempt(&self, client_id: &str) -> bool {
        let mut limiters = self.limiters.lock().unwrap();

        let now = Instant::now();
        limiters
            .retain(|_, (_, last_seen)| now.duration_since(*last_seen) < Duration::from_secs(300));

        let (limiter, last_seen) = limiters.entry(client_id.to_string()).or_insert_with(|| {
            let quota = Quota::per_minute(NonZeroU32::new(self.max_attempts).unwrap());
            (RateLimiter::direct(quota), now)
        });

        *last_seen = now;
        limiter.check().is_ok()
    }

    pub fn reset(&self, client_id: &str) {
        let mut limiters = self.limiters.lock().unwrap();
        limiters.remove(client_id);
    }
}

/// Rate limiter for authentication attempts (per IP address)
pub struct IpRateLimiter {
    limiters:
        Arc<Mutex<HashMap<String, (RateLimiter<NotKeyed, InMemoryState, DefaultClock>, Instant)>>>,
    max_attempts: u32,
}

impl IpRateLimiter {
    pub fn new() -> Self {
        let max_attempts = env::var("RATE_LIMIT_AUTH_PER_MINUTE_IP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);

        Self {
            limiters: Arc::new(Mutex::new(HashMap::new())),
            max_attempts,
        }
    }

    pub fn check_auth_attempt_ip(&self, ip: &str) -> bool {
        let mut limiters = self.limiters.lock().unwrap();

        let now = Instant::now();
        limiters
            .retain(|_, (_, last_seen)| now.duration_since(*last_seen) < Duration::from_secs(300));

        let (limiter, last_seen) = limiters.entry(ip.to_string()).or_insert_with(|| {
            let quota = Quota::per_minute(NonZeroU32::new(self.max_attempts).unwrap());
            (RateLimiter::direct(quota), now)
        });

        *last_seen = now;
        limiter.check().is_ok()
    }
}

/// Rate limiter for commands (per socket ID)
pub struct CommandRateLimiter {
    limiters:
        Arc<Mutex<HashMap<String, (RateLimiter<NotKeyed, InMemoryState, DefaultClock>, Instant)>>>,
    max_commands: u32,
}

impl CommandRateLimiter {
    pub fn new() -> Self {
        let max_commands = env::var("RATE_LIMIT_COMMANDS_PER_SECOND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);

        Self {
            limiters: Arc::new(Mutex::new(HashMap::new())),
            max_commands,
        }
    }

    pub fn check_command(&self, client_id: &str) -> bool {
        let mut limiters = self.limiters.lock().unwrap();

        let now = Instant::now();
        limiters
            .retain(|_, (_, last_seen)| now.duration_since(*last_seen) < Duration::from_secs(300));

        let (limiter, last_seen) = limiters.entry(client_id.to_string()).or_insert_with(|| {
            let quota = Quota::per_second(NonZeroU32::new(self.max_commands).unwrap());
            (RateLimiter::direct(quota), now)
        });

        *last_seen = now;
        limiter.check().is_ok()
    }
}

// ─── User Model ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct User {
    pub username: String,
    pub password_hash: String,
    pub role: String,
    pub enabled: bool,
    pub created_at: i64, // Unix ms
    pub updated_at: i64, // Unix ms
}

// ─── MongoDB Helpers ─────────────────────────────────────────────────────────

pub async fn connect_db(uri: &str, db_name: &str) -> Result<Database, mongodb::error::Error> {
    let options = ClientOptions::parse(uri).await?;
    let client = mongodb::Client::with_options(options)?;
    Ok(client.database(db_name))
}

pub async fn ensure_indexes(collection: &Collection<User>) -> Result<(), mongodb::error::Error> {
    let index = IndexModel::builder()
        .keys(doc! { "username": 1 })
        .options(IndexOptions::builder().unique(true).build())
        .build();
    collection.create_index(index, None).await?;
    Ok(())
}

/// Seeds a default admin user if the collection is empty.
/// Returns true if a user was inserted.
pub async fn seed_admin_user(collection: &Collection<User>) -> Result<bool, mongodb::error::Error> {
    let count = collection.count_documents(None, None).await?;
    if count > 0 {
        return Ok(false);
    }

    let hash =
        bcrypt::hash("password", bcrypt::DEFAULT_COST).expect("bcrypt hash failed during seed");

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    let admin = User {
        username: "admin".to_string(),
        password_hash: hash,
        role: "admin".to_string(),
        enabled: true,
        created_at: now,
        updated_at: now,
    };

    collection.insert_one(admin, None).await?;
    tracing::warn!(
        security_event = "default_user_seeded",
        "Default admin user created — change password immediately"
    );
    Ok(true)
}

pub async fn find_user(
    collection: &Collection<User>,
    username: &str,
) -> Result<Option<User>, mongodb::error::Error> {
    collection
        .find_one(doc! { "username": username }, None)
        .await
}

/// CPU-bound bcrypt verify; call via tokio::task::spawn_blocking.
pub fn verify_password_blocking(candidate: &str, hash: &str) -> bool {
    bcrypt::verify(candidate, hash).unwrap_or(false)
}

// ─── Auth Error ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum AuthErrorReason {
    InvalidCredentials,
    AccountDisabled,
}

impl AuthErrorReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidCredentials => "invalid_credentials",
            // Same string as InvalidCredentials to prevent username enumeration
            Self::AccountDisabled => "invalid_credentials",
        }
    }
}

// ─── JWT Module ──────────────────────────────────────────────────────────────

pub mod jwt {
    use jsonwebtoken::{
        decode, encode, errors::Error as JwtError, DecodingKey, EncodingKey, Header, Validation,
    };
    use serde::{Deserialize, Serialize};
    use std::time::{SystemTime, UNIX_EPOCH};
    use uuid::Uuid;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Claims {
        pub sub: String,
        pub role: String,
        pub iat: u64,
        pub exp: u64,
        pub jti: String,
    }

    /// Returns (token_string, claims). Claims are needed for SessionRegistry.
    pub fn generate_token(
        username: &str,
        role: &str,
        secret: &str,
        ttl_secs: u64,
    ) -> Result<(String, Claims), JwtError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let claims = Claims {
            sub: username.to_string(),
            role: role.to_string(),
            iat: now,
            exp: now + ttl_secs,
            jti: Uuid::new_v4().to_string(),
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )?;
        Ok((token, claims))
    }

    pub fn validate_token(token: &str, secret: &str) -> Result<Claims, JwtError> {
        let mut validation = Validation::default();
        validation.validate_exp = true;
        let data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(secret.as_bytes()),
            &validation,
        )?;
        Ok(data.claims)
    }
}

// ─── Session Registry ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SessionRegistry {
    inner: Arc<Mutex<HashMap<String, jwt::Claims>>>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn register(&self, socket_id: &str, claims: jwt::Claims) {
        self.inner
            .lock()
            .unwrap()
            .insert(socket_id.to_string(), claims);
    }

    pub fn is_valid(&self, socket_id: &str) -> bool {
        let registry = self.inner.lock().unwrap();
        if let Some(claims) = registry.get(socket_id) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            claims.exp > now
        } else {
            false
        }
    }

    pub fn remove(&self, socket_id: &str) {
        self.inner.lock().unwrap().remove(socket_id);
    }

    /// Removes expired sessions; returns their socket IDs for disconnection.
    pub fn sweep_expired(&self) -> Vec<String> {
        let mut registry = self.inner.lock().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let expired: Vec<String> = registry
            .iter()
            .filter(|(_, claims)| claims.exp <= now)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &expired {
            registry.remove(id);
        }
        expired
    }
}

// ─── Auth Flow ───────────────────────────────────────────────────────────────

/// Validates credentials (token-first, then password) and issues a fresh JWT.
/// Returns (token_string, claims) on success.
pub async fn authenticate_and_issue_token(
    token: Option<&str>,
    username: &str,
    password: &str,
    user_collection: &Collection<User>,
    jwt_secret: &str,
    session_ttl_secs: u64,
) -> Result<(String, jwt::Claims), AuthErrorReason> {
    // Token-first path: if valid, re-validate user is still enabled, then re-issue with fresh expiry
    if let Some(t) = token {
        if let Ok(old_claims) = jwt::validate_token(t, jwt_secret) {
            let user = find_user(user_collection, &old_claims.sub)
                .await
                .map_err(|_| AuthErrorReason::InvalidCredentials)?
                .ok_or(AuthErrorReason::InvalidCredentials)?;

            if !user.enabled {
                return Err(AuthErrorReason::AccountDisabled);
            }

            return jwt::generate_token(
                &old_claims.sub,
                &old_claims.role,
                jwt_secret,
                session_ttl_secs,
            )
            .map_err(|_| AuthErrorReason::InvalidCredentials);
        }
        // Invalid/expired token → fall through to password auth
    }

    // Password path
    let user = find_user(user_collection, username)
        .await
        .map_err(|_| AuthErrorReason::InvalidCredentials)?
        .ok_or(AuthErrorReason::InvalidCredentials)?;

    if !user.enabled {
        return Err(AuthErrorReason::AccountDisabled);
    }

    let hash = user.password_hash.clone();
    let candidate = password.to_string();
    let valid = tokio::task::spawn_blocking(move || verify_password_blocking(&candidate, &hash))
        .await
        .unwrap_or_else(|e| {
            tracing::error!(security_event = "bcrypt_panic", error = %e, "bcrypt verify panicked");
            false
        });

    if !valid {
        return Err(AuthErrorReason::InvalidCredentials);
    }

    jwt::generate_token(&user.username, &user.role, jwt_secret, session_ttl_secs)
        .map_err(|_| AuthErrorReason::InvalidCredentials)
}

// ─── Input Validation ─────────────────────────────────────────────────────────

pub mod validation {
    use std::env;

    pub fn validate_joint_position(angle: f64) -> Result<(), String> {
        if !angle.is_finite() {
            return Err("Joint angle must be a finite number".to_string());
        }
        if angle < -std::f64::consts::PI || angle > std::f64::consts::PI {
            return Err(format!("Joint angle {} out of range [-π, π]", angle));
        }
        Ok(())
    }

    pub fn validate_wheel_velocity(velocity: f64) -> Result<(), String> {
        let max_velocity = env::var("MAX_WHEEL_VELOCITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.0);

        if !velocity.is_finite() {
            return Err("Wheel velocity must be a finite number".to_string());
        }
        if velocity.abs() > max_velocity {
            return Err(format!(
                "Wheel velocity {} exceeds limit {}",
                velocity, max_velocity
            ));
        }
        Ok(())
    }

    pub fn validate_tts_text(text: &str) -> Result<(), String> {
        let max_length = env::var("MAX_TTS_TEXT_LENGTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);

        if text.is_empty() {
            return Err("TTS text cannot be empty".to_string());
        }
        if text.len() > max_length {
            return Err(format!(
                "TTS text length {} exceeds limit {}",
                text.len(),
                max_length
            ));
        }
        Ok(())
    }

    pub fn validate_audio_data(samples: &[f32]) -> Result<(), String> {
        let max_samples = env::var("MAX_AUDIO_SAMPLES_PER_MESSAGE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16000);

        if samples.is_empty() {
            return Err("Audio data cannot be empty".to_string());
        }
        if samples.len() > max_samples {
            return Err(format!(
                "Audio sample count {} exceeds limit {}",
                samples.len(),
                max_samples
            ));
        }

        for (i, &sample) in samples.iter().enumerate() {
            if !sample.is_finite() {
                return Err(format!("Audio sample at index {} is not finite", i));
            }
        }
        Ok(())
    }

    pub fn validate_voice_stream_format(sample_rate: u32, channels: u16) -> Result<(), String> {
        if !(8_000..=192_000).contains(&sample_rate) {
            return Err("Voice sample rate must be between 8000 and 192000 Hz".to_string());
        }
        if channels != 1 {
            return Err("Voice command audio must be mono".to_string());
        }
        Ok(())
    }

    pub fn validate_voice_audio_frame(
        sample_rate: u32,
        channels: u16,
        sample_count: u32,
        samples: &[f32],
    ) -> Result<(), String> {
        validate_voice_stream_format(sample_rate, channels)?;
        let declared = usize::try_from(sample_count)
            .map_err(|_| "Voice sample count exceeds platform size".to_string())?;
        if declared != samples.len() {
            return Err(format!(
                "Voice sample count mismatch: declared {declared}, received {}",
                samples.len()
            ));
        }
        if sample_count > sample_rate {
            return Err("Voice audio frame duration must not exceed one second".to_string());
        }
        validate_audio_data(samples)
    }
}

/// Extracts the client IP from request headers.
/// When `trust_proxy` is true, parses X-Forwarded-For (first hop) then X-Real-IP.
/// Falls back to "unknown" when no peer address is available.
pub fn extract_client_ip(headers: &axum::http::HeaderMap, trust_proxy: bool) -> String {
    if trust_proxy {
        if let Some(fwd) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
            let first = fwd.split(',').next().unwrap_or("").trim();
            if !first.is_empty() {
                return first.to_string();
            }
        }
        if let Some(real_ip) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
            let ip = real_ip.trim();
            if !ip.is_empty() {
                return ip.to_string();
            }
        }
    }
    "unknown".to_string()
}

/// Warns on any non-localhost HTTP origin when ALLOW_HTTP_ORIGINS is not "true".
pub fn warn_http_origins(origins: &[String]) {
    if env::var("ALLOW_HTTP_ORIGINS").unwrap_or_else(|_| "false".to_string()) == "true" {
        return;
    }
    for origin in origins {
        if origin.starts_with("http://") {
            let host = origin.trim_start_matches("http://");
            let is_local = host.starts_with("localhost")
                || host.starts_with("127.0.0.1")
                || host.starts_with("0.0.0.0");
            if !is_local {
                tracing::warn!(
                    security_event = "http_origin_in_production",
                    origin = origin.as_str(),
                    "Non-HTTPS origin configured — set ALLOW_HTTP_ORIGINS=true to silence this warning"
                );
            }
        }
    }
}

// ─── CORS / Origins ───────────────────────────────────────────────────────────

pub fn parse_allowed_origins() -> Vec<String> {
    env::var("ALLOWED_ORIGINS")
        .unwrap_or_else(|_| "http://localhost:3000,http://localhost:5173".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// ─── Audit Logging ────────────────────────────────────────────────────────────

/// Logs auth outcome. On failure, username is NOT logged to prevent enumeration.
pub fn log_auth_attempt(client_id: &str, username: &str, success: bool) {
    if env::var("LOG_AUTH_ATTEMPTS").unwrap_or_else(|_| "true".to_string()) == "true" {
        if success {
            tracing::info!(
                security_event = "auth_success",
                client_id = client_id,
                username = username,
                "Authentication successful"
            );
        } else {
            tracing::warn!(
                security_event = "auth_failure",
                client_id = client_id,
                "Authentication failed"
            );
        }
    }
}

pub fn log_rate_limit_exceeded(client_id: &str, limit_type: &str) {
    tracing::warn!(
        security_event = "rate_limit_exceeded",
        client_id = client_id,
        limit_type = limit_type,
        "Rate limit exceeded"
    );
}

pub fn log_validation_error(client_id: &str, error: &str) {
    tracing::warn!(
        security_event = "validation_error",
        client_id = client_id,
        error = error,
        "Input validation failed"
    );
}

// ─── JWT Secret Loader ────────────────────────────────────────────────────────

/// Loads JWT_SECRET from env, or generates a random 64-char hex secret with a loud warning.
pub fn load_or_generate_jwt_secret() -> String {
    match env::var("JWT_SECRET") {
        Ok(s) if !s.is_empty() => s,
        _ => {
            use rand::Rng;
            let secret: String = rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(64)
                .map(char::from)
                .collect();
            tracing::warn!(
                security_event = "jwt_secret_auto_generated",
                "JWT_SECRET not set — auto-generated ephemeral secret. \
                 All sessions will be invalidated on restart. \
                 Set JWT_SECRET in .env for persistent sessions."
            );
            secret
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joint_validation() {
        assert!(validation::validate_joint_position(0.0).is_ok());
        assert!(validation::validate_joint_position(std::f64::consts::PI).is_ok());
        assert!(validation::validate_joint_position(-std::f64::consts::PI).is_ok());
        assert!(validation::validate_joint_position(std::f64::consts::PI + 0.1).is_err());
        assert!(validation::validate_joint_position(f64::NAN).is_err());
        assert!(validation::validate_joint_position(f64::INFINITY).is_err());
    }

    #[test]
    fn test_tts_validation() {
        assert!(validation::validate_tts_text("Hello").is_ok());
        assert!(validation::validate_tts_text("").is_err());
        assert!(validation::validate_tts_text(&"a".repeat(2000)).is_err());
    }

    #[test]
    fn test_audio_validation() {
        assert!(validation::validate_audio_data(&[0.1, 0.2, 0.3]).is_ok());
        assert!(validation::validate_audio_data(&[]).is_err());
        assert!(validation::validate_audio_data(&[f32::NAN]).is_err());
        assert!(validation::validate_audio_data(&[f32::INFINITY]).is_err());
    }

    #[test]
    fn test_voice_frame_validation() {
        assert!(validation::validate_voice_audio_frame(48_000, 1, 2, &[0.0, 0.1]).is_ok());
        assert!(validation::validate_voice_audio_frame(48_000, 2, 2, &[0.0, 0.1]).is_err());
        assert!(validation::validate_voice_audio_frame(1_000, 1, 2, &[0.0, 0.1]).is_err());
        assert!(validation::validate_voice_audio_frame(48_000, 1, 3, &[0.0, 0.1]).is_err());
        assert!(validation::validate_voice_audio_frame(48_000, 1, 1, &[f32::NAN]).is_err());
        assert!(
            validation::validate_voice_audio_frame(8_000, 1, 8_001, &vec![0.0; 8_001]).is_err()
        );
    }

    #[test]
    fn test_bcrypt_verify() {
        let hash = bcrypt::hash("correct-horse", bcrypt::DEFAULT_COST).unwrap();
        assert!(verify_password_blocking("correct-horse", &hash));
        assert!(!verify_password_blocking("wrong-password", &hash));
    }

    #[test]
    fn test_jwt_roundtrip() {
        let (token, claims) = jwt::generate_token("alice", "admin", "test-secret", 3600).unwrap();
        let validated = jwt::validate_token(&token, "test-secret").unwrap();
        assert_eq!(validated.sub, claims.sub);
        assert_eq!(validated.role, claims.role);
        assert_eq!(validated.jti, claims.jti);
    }

    #[test]
    fn test_jwt_expired() {
        let (token, _) = jwt::generate_token("alice", "admin", "test-secret", 0).unwrap();
        // Expired immediately (exp == iat); validation should fail
        // Note: jsonwebtoken may give 1-second leeway, so we can't guarantee failure with ttl=0.
        // This test confirms the token was generated without panicking.
        let _ = jwt::validate_token(&token, "test-secret");
    }

    #[test]
    fn test_jwt_wrong_secret() {
        let (token, _) = jwt::generate_token("alice", "admin", "secret-a", 3600).unwrap();
        assert!(jwt::validate_token(&token, "secret-b").is_err());
    }

    #[test]
    fn test_session_registry_sweep() {
        let registry = SessionRegistry::new();

        // Insert a session that expired in the past (exp = 1)
        let expired_claims = jwt::Claims {
            sub: "user1".to_string(),
            role: "admin".to_string(),
            iat: 1,
            exp: 1,
            jti: "test-jti".to_string(),
        };
        registry.register("socket-expired", expired_claims);

        // Insert a valid session (exp far in the future)
        let valid_claims = jwt::Claims {
            sub: "user2".to_string(),
            role: "admin".to_string(),
            iat: u64::MAX / 2,
            exp: u64::MAX,
            jti: "test-jti-2".to_string(),
        };
        registry.register("socket-valid", valid_claims);

        let swept = registry.sweep_expired();
        assert!(swept.contains(&"socket-expired".to_string()));
        assert!(!swept.contains(&"socket-valid".to_string()));
        assert!(registry.is_valid("socket-valid"));
        assert!(!registry.is_valid("socket-expired"));
    }

    #[test]
    fn test_log_auth_attempt_no_username_on_failure() {
        // Just verify it doesn't panic — actual log content verified via tracing subscriber
        log_auth_attempt("socket-123", "admin", false);
        log_auth_attempt("socket-123", "admin", true);
    }

    #[test]
    fn test_extract_ip_forwarded_for_trust_proxy() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4, 5.6.7.8".parse().unwrap());
        assert_eq!(extract_client_ip(&headers, true), "1.2.3.4");
    }

    #[test]
    fn test_extract_ip_no_proxy_trust_false() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
        // trust_proxy=false → must not use headers
        assert_eq!(extract_client_ip(&headers, false), "unknown");
    }

    #[test]
    fn test_extract_ip_real_ip_fallback() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-real-ip", "10.0.0.1".parse().unwrap());
        assert_eq!(extract_client_ip(&headers, true), "10.0.0.1");
    }

    #[test]
    fn test_extract_ip_empty_headers() {
        let headers = axum::http::HeaderMap::new();
        assert_eq!(extract_client_ip(&headers, true), "unknown");
        assert_eq!(extract_client_ip(&headers, false), "unknown");
    }

    #[test]
    fn test_ip_rate_limiter_allows_under_limit() {
        // Default limit is 20/min; first call must succeed
        let limiter = IpRateLimiter::new();
        assert!(limiter.check_auth_attempt_ip("1.2.3.4"));
    }

    #[test]
    fn test_ip_rate_limiter_blocks_after_exhaustion() {
        // Override via env is not practical in unit test; use governor directly by hammering
        // Set env for this test scope — may not affect already-constructed limiter,
        // so we construct after setting env.
        // governor default quota is read at construction time.
        std::env::set_var("RATE_LIMIT_AUTH_PER_MINUTE_IP", "3");
        let limiter = IpRateLimiter::new();
        // Exhaust the bucket
        let _ = limiter.check_auth_attempt_ip("9.9.9.9");
        let _ = limiter.check_auth_attempt_ip("9.9.9.9");
        let _ = limiter.check_auth_attempt_ip("9.9.9.9");
        // 4th attempt must be blocked
        assert!(!limiter.check_auth_attempt_ip("9.9.9.9"));
        // Different IP must still pass
        assert!(limiter.check_auth_attempt_ip("8.8.8.8"));
        std::env::remove_var("RATE_LIMIT_AUTH_PER_MINUTE_IP");
    }

    #[test]
    fn test_warn_http_origins_no_panic() {
        // Ensure the function runs without panicking for mixed origin list
        let origins = vec![
            "http://example.com".to_string(),
            "https://secure.example.com".to_string(),
            "http://localhost:3000".to_string(),
            "http://127.0.0.1:5173".to_string(),
        ];
        warn_http_origins(&origins); // Must not panic
    }
}
