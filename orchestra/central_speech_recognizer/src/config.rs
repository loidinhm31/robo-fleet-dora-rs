use crate::profile_catalog::{self, ModelPaths};
use robo_rover_lib::SttProfile;
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::path::PathBuf;

pub const VAD_SAMPLE_RATE: i32 = 16_000;
pub const VAD_WINDOW_SIZE: usize = 512;
pub const MAX_VAD_DURATION_SECONDS: f32 = 120.0;

const ENV_NAMES: [&str; 10] = [
    "STT_PROFILE",
    "STT_MODEL_ROOT",
    "STT_NUM_THREADS",
    "STT_VAD_THRESHOLD",
    "STT_VAD_MIN_SILENCE_SECONDS",
    "STT_VAD_MIN_SPEECH_SECONDS",
    "STT_VAD_MAX_SPEECH_SECONDS",
    "STT_DECODE_QUEUE_CAPACITY",
    "STT_SAMPLE_RATE",
    "STT_VAD_WINDOW_SIZE",
];

#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    pub threshold: f32,
    pub min_silence_seconds: f32,
    pub min_speech_seconds: f32,
    pub max_speech_seconds: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SttConfig {
    pub models: ModelPaths,
    pub num_threads: i32,
    pub vad: VadConfig,
    pub decode_queue_capacity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    Invalid(&'static str),
    MissingModel(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(name) => write!(formatter, "invalid {name}"),
            Self::MissingModel(name) => write!(formatter, "required model file missing: {name}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl SttConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut values = HashMap::new();
        for name in ENV_NAMES {
            match env::var(name) {
                Ok(value) => {
                    values.insert(name.to_string(), value);
                }
                Err(env::VarError::NotPresent) => {}
                Err(env::VarError::NotUnicode(_)) => return Err(ConfigError::Invalid(name)),
            }
        }
        Self::from_values(&values)
    }

    pub(crate) fn from_values(values: &HashMap<String, String>) -> Result<Self, ConfigError> {
        validate_fixed(values, "STT_SAMPLE_RATE", VAD_SAMPLE_RATE)?;
        validate_fixed(values, "STT_VAD_WINDOW_SIZE", VAD_WINDOW_SIZE)?;

        let profile = parse_profile(value(
            values,
            "STT_PROFILE",
            profile_catalog::DEFAULT_PROFILE,
        ))?;
        let model_root = PathBuf::from(value(
            values,
            "STT_MODEL_ROOT",
            profile_catalog::DEFAULT_MODEL_ROOT,
        ));
        let models = profile_catalog::resolve(profile, &model_root);

        let config = Self {
            models,
            num_threads: parse(values, "STT_NUM_THREADS", 2)?,
            vad: VadConfig {
                threshold: parse(values, "STT_VAD_THRESHOLD", 0.5)?,
                min_silence_seconds: parse(values, "STT_VAD_MIN_SILENCE_SECONDS", 0.25)?,
                min_speech_seconds: parse(values, "STT_VAD_MIN_SPEECH_SECONDS", 0.25)?,
                max_speech_seconds: parse(values, "STT_VAD_MAX_SPEECH_SECONDS", 8.0)?,
            },
            decode_queue_capacity: parse(values, "STT_DECODE_QUEUE_CAPACITY", 8)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if !(1..=64).contains(&self.num_threads) {
            return Err(ConfigError::Invalid("STT_NUM_THREADS"));
        }
        if !(0.0..1.0).contains(&self.vad.threshold) || self.vad.threshold == 0.0 {
            return Err(ConfigError::Invalid("STT_VAD_THRESHOLD"));
        }
        if !self.vad.min_silence_seconds.is_finite()
            || !(0.0..=MAX_VAD_DURATION_SECONDS).contains(&self.vad.min_silence_seconds)
        {
            return Err(ConfigError::Invalid("STT_VAD_MIN_SILENCE_SECONDS"));
        }
        if !self.vad.min_speech_seconds.is_finite()
            || self.vad.min_speech_seconds <= 0.0
            || self.vad.min_speech_seconds > MAX_VAD_DURATION_SECONDS
        {
            return Err(ConfigError::Invalid("STT_VAD_MIN_SPEECH_SECONDS"));
        }
        if !self.vad.max_speech_seconds.is_finite()
            || self.vad.max_speech_seconds <= self.vad.min_speech_seconds
            || self.vad.max_speech_seconds > MAX_VAD_DURATION_SECONDS
        {
            return Err(ConfigError::Invalid("STT_VAD_MAX_SPEECH_SECONDS"));
        }
        if !(1..=1024).contains(&self.decode_queue_capacity) {
            return Err(ConfigError::Invalid("STT_DECODE_QUEUE_CAPACITY"));
        }
        for (name, path) in self.models.required_files() {
            if !path.is_file() {
                return Err(ConfigError::MissingModel(name));
            }
        }
        Ok(())
    }
}

fn parse_profile(value: &str) -> Result<SttProfile, ConfigError> {
    match value {
        "en-vad-offline" => Ok(SttProfile::EnVadOffline),
        "vi-vad-offline" => Ok(SttProfile::ViVadOffline),
        _ => Err(ConfigError::Invalid("STT_PROFILE")),
    }
}

fn value<'a>(values: &'a HashMap<String, String>, name: &str, default: &'a str) -> &'a str {
    values.get(name).map(String::as_str).unwrap_or(default)
}

fn parse<T>(
    values: &HashMap<String, String>,
    name: &'static str,
    default: T,
) -> Result<T, ConfigError>
where
    T: std::str::FromStr,
{
    values
        .get(name)
        .map(|raw| raw.parse().map_err(|_| ConfigError::Invalid(name)))
        .unwrap_or(Ok(default))
}

fn validate_fixed<T>(
    values: &HashMap<String, String>,
    name: &'static str,
    expected: T,
) -> Result<(), ConfigError>
where
    T: std::str::FromStr + PartialEq,
{
    if let Some(raw) = values.get(name) {
        if raw.parse().ok().as_ref() != Some(&expected) {
            return Err(ConfigError::Invalid(name));
        }
    }
    Ok(())
}
