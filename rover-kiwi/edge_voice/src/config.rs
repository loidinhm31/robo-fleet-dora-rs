use std::{
    env,
    path::{Path, PathBuf},
};

use eyre::{eyre, Result};
use robo_rover_lib::TtsRuntimeConfig;

pub const DEFAULT_MODEL_DIR: &str =
    "models/.cache/sherpa-onnx/tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11";
pub const DEFAULT_QUEUE_CAPACITY: usize = 8;
pub const EXPECTED_SAMPLE_RATE: i32 = 44_100;
pub const EXPECTED_SPEAKERS: i32 = 10;

#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    pub entity_id: String,
    pub model_dir: PathBuf,
    pub num_threads: i32,
    pub queue_capacity: usize,
    pub debug: bool,
    pub default_runtime: TtsRuntimeConfig,
}

impl DeploymentConfig {
    pub fn from_env() -> Result<Self> {
        let default_runtime = TtsRuntimeConfig {
            language: env_language("TTS_DEFAULT_LANGUAGE", TtsRuntimeConfig::default().language)?,
            speaker_id: env_parse(
                "TTS_DEFAULT_SPEAKER_ID",
                TtsRuntimeConfig::default().speaker_id,
            )?,
            speed: env_parse("TTS_DEFAULT_SPEED", TtsRuntimeConfig::default().speed)?,
            num_steps: env_parse("TTS_DEFAULT_STEPS", TtsRuntimeConfig::default().num_steps)?,
            volume: env_parse("TTS_DEFAULT_VOLUME", TtsRuntimeConfig::default().volume)?,
        };
        default_runtime.validate().map_err(eyre::Report::msg)?;

        let queue_capacity = env_parse("EDGE_VOICE_QUEUE_CAPACITY", DEFAULT_QUEUE_CAPACITY)?;
        if queue_capacity == 0 {
            return Err(eyre!("EDGE_VOICE_QUEUE_CAPACITY must be greater than zero"));
        }

        let num_threads = env_parse("EDGE_VOICE_NUM_THREADS", env_parse("TTS_NUM_THREADS", 2)?)?;
        if num_threads <= 0 {
            return Err(eyre!("EDGE_VOICE_NUM_THREADS must be greater than zero"));
        }

        Ok(Self {
            entity_id: env::var("ENTITY_ID").unwrap_or_else(|_| "rover-kiwi".to_string()),
            model_dir: resolve_runtime_path(
                "EDGE_VOICE_MODEL_DIR",
                &env::var("EDGE_VOICE_MODEL_DIR")
                    .unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string()),
            ),
            num_threads,
            queue_capacity,
            debug: env_bool("EDGE_VOICE_DEBUG", false)?,
            default_runtime,
        })
    }
}

fn resolve_runtime_path(label: &str, raw_path: &str) -> PathBuf {
    let raw = Path::new(raw_path);
    if raw.is_absolute() || raw.exists() {
        return raw.to_path_buf();
    }

    let mut candidate_bases = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        candidate_bases.push(cwd);
    }
    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidate_bases.extend(exe_dir.ancestors().map(Path::to_path_buf));
        }
    }

    for base in candidate_bases {
        let candidate = base.join(raw);
        if candidate.exists() {
            tracing::info!(
                path_label = label,
                configured_path = raw_path,
                resolved_path = %candidate.display(),
                "Resolved edge voice runtime asset path"
            );
            return candidate;
        }
    }

    tracing::warn!(
        path_label = label,
        configured_path = raw_path,
        "Edge voice runtime asset path could not be resolved; using configured value"
    );
    raw.to_path_buf()
}

fn env_parse<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|error| eyre!("{name} is invalid: {error}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(eyre!("{name} is invalid: {error}")),
    }
}

fn env_bool(name: &str, default: bool) -> Result<bool> {
    match env::var(name) {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(eyre!("{name} must be a boolean")),
        },
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(eyre!("{name} is invalid: {error}")),
    }
}

fn env_language(
    name: &str,
    default: robo_rover_lib::TtsLanguage,
) -> Result<robo_rover_lib::TtsLanguage> {
    match env::var(name) {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "en" => Ok(robo_rover_lib::TtsLanguage::En),
            "vi" => Ok(robo_rover_lib::TtsLanguage::Vi),
            _ => Err(eyre!("{name} must be en or vi")),
        },
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(eyre!("{name} is invalid: {error}")),
    }
}
