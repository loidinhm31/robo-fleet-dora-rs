use governor::{Quota, RateLimiter, clock::DefaultClock, state::{InMemoryState, NotKeyed}};
use std::collections::HashMap;
use std::env;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Rate limiter for authentication attempts (per IP/client)
pub struct AuthRateLimiter {
    limiters: Arc<Mutex<HashMap<String, (RateLimiter<NotKeyed, InMemoryState, DefaultClock>, Instant)>>>,
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

        // Clean up old entries (older than 5 minutes)
        let now = Instant::now();
        limiters.retain(|_, (_, last_seen)| now.duration_since(*last_seen) < Duration::from_secs(300));

        // Get or create rate limiter for this client
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

/// Rate limiter for commands (per client)
pub struct CommandRateLimiter {
    limiters: Arc<Mutex<HashMap<String, (RateLimiter<NotKeyed, InMemoryState, DefaultClock>, Instant)>>>,
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

        // Clean up old entries (older than 5 minutes)
        let now = Instant::now();
        limiters.retain(|_, (_, last_seen)| now.duration_since(*last_seen) < Duration::from_secs(300));

        // Get or create rate limiter for this client
        let (limiter, last_seen) = limiters.entry(client_id.to_string()).or_insert_with(|| {
            let quota = Quota::per_second(NonZeroU32::new(self.max_commands).unwrap());
            (RateLimiter::direct(quota), now)
        });

        *last_seen = now;
        limiter.check().is_ok()
    }
}

/// Input validation utilities
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
            return Err(format!("Wheel velocity {} exceeds limit {}", velocity, max_velocity));
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
            return Err(format!("TTS text length {} exceeds limit {}", text.len(), max_length));
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
            return Err(format!("Audio sample count {} exceeds limit {}", samples.len(), max_samples));
        }

        // Validate all samples are finite
        for (i, &sample) in samples.iter().enumerate() {
            if !sample.is_finite() {
                return Err(format!("Audio sample at index {} is not finite", i));
            }
        }
        Ok(())
    }

    pub fn validate_detection_index(index: usize, max: usize) -> Result<(), String> {
        if index >= max {
            return Err(format!("Detection index {} out of bounds (max: {})", index, max));
        }
        Ok(())
    }
}

/// CORS origin validation
pub fn parse_allowed_origins() -> Vec<String> {
    env::var("ALLOWED_ORIGINS")
        .unwrap_or_else(|_| "http://localhost:3000,http://localhost:5173".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Audit logging for security events
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
                username = username,
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
}
