use crate::config::{
    ConfigError, SttConfig, MAX_VAD_DURATION_SECONDS, VAD_SAMPLE_RATE, VAD_WINDOW_SIZE,
};
use robo_rover_lib::SttProfile;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn defaults_select_english_profile() {
    let root = model_root(SttProfile::EnVadOffline);
    let config = load(&root, &[]).unwrap();

    assert_eq!(config.models.profile, SttProfile::EnVadOffline);
    assert_eq!(config.models.language, "en");
    assert_eq!(config.num_threads, 2);
    assert_eq!(config.decode_queue_capacity, 8);
    assert!(config
        .models
        .encoder
        .ends_with("exp/encoder-epoch-30-avg-4.int8.onnx"));
    assert!(config
        .models
        .tokens
        .ends_with("data/lang_bpe_500/tokens.txt"));
    cleanup(root);
}

#[test]
fn maps_vietnamese_profile() {
    let root = model_root(SttProfile::ViVadOffline);
    let config = load(&root, &[("STT_PROFILE", "vi-vad-offline")]).unwrap();

    assert_eq!(config.models.profile, SttProfile::ViVadOffline);
    assert_eq!(config.models.language, "vi");
    assert!(config.models.encoder.ends_with("encoder.int8.onnx"));
    cleanup(root);
}

#[test]
fn rejects_unknown_profile_and_missing_files() {
    let root = model_root(SttProfile::EnVadOffline);
    assert_eq!(
        load(&root, &[("STT_PROFILE", "runtime-path")]).unwrap_err(),
        ConfigError::Invalid("STT_PROFILE")
    );
    fs::remove_file(root.join("silero/silero_vad.onnx")).unwrap();
    assert_eq!(
        load(&root, &[]).unwrap_err(),
        ConfigError::MissingModel("Silero VAD")
    );
    cleanup(root);
}

#[test]
fn rejects_each_missing_model_role() {
    for (name, index) in [
        ("Silero VAD", 0),
        ("encoder", 1),
        ("decoder", 2),
        ("joiner", 3),
        ("tokens", 4),
    ] {
        let root = model_root(SttProfile::EnVadOffline);
        let paths = crate::profile_catalog::resolve(SttProfile::EnVadOffline, &root);
        fs::remove_file(paths.required_files()[index].1).unwrap();
        assert_eq!(
            load(&root, &[]).unwrap_err(),
            ConfigError::MissingModel(name)
        );
        cleanup(root);
    }
}

#[test]
fn rejects_invalid_vad_and_fixed_audio_values() {
    let root = model_root(SttProfile::EnVadOffline);
    for (name, value) in [
        ("STT_VAD_THRESHOLD", "1"),
        ("STT_VAD_MIN_SILENCE_SECONDS", "-0.1"),
        ("STT_VAD_MIN_SILENCE_SECONDS", "121"),
        ("STT_VAD_MIN_SPEECH_SECONDS", "0"),
        ("STT_VAD_MAX_SPEECH_SECONDS", "0.1"),
        ("STT_VAD_MAX_SPEECH_SECONDS", "121"),
        ("STT_SAMPLE_RATE", "8000"),
        ("STT_VAD_WINDOW_SIZE", "256"),
    ] {
        assert_eq!(
            load(&root, &[(name, value)]).unwrap_err(),
            ConfigError::Invalid(name)
        );
    }
    assert_eq!(VAD_SAMPLE_RATE, 16_000);
    assert_eq!(VAD_WINDOW_SIZE, 512);
    assert_eq!(MAX_VAD_DURATION_SECONDS, 120.0);
    cleanup(root);
}

#[test]
fn rejects_thread_and_queue_bounds() {
    let root = model_root(SttProfile::EnVadOffline);
    for (name, value) in [
        ("STT_NUM_THREADS", "0"),
        ("STT_NUM_THREADS", "65"),
        ("STT_DECODE_QUEUE_CAPACITY", "0"),
        ("STT_DECODE_QUEUE_CAPACITY", "1025"),
    ] {
        assert_eq!(
            load(&root, &[(name, value)]).unwrap_err(),
            ConfigError::Invalid(name)
        );
    }
    cleanup(root);
}

fn load(root: &Path, overrides: &[(&str, &str)]) -> Result<SttConfig, ConfigError> {
    let mut values = HashMap::from([(
        "STT_MODEL_ROOT".to_string(),
        root.to_string_lossy().into_owned(),
    )]);
    values.extend(
        overrides
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string())),
    );
    SttConfig::from_values(&values)
}

fn model_root(profile: SttProfile) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "robo-fleet-stt-config-{}-{suffix}",
        std::process::id()
    ));
    let paths = crate::profile_catalog::resolve(profile, &root);
    for (_, path) in paths.required_files() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"fixture").unwrap();
    }
    root
}

fn cleanup(root: PathBuf) {
    fs::remove_dir_all(root).unwrap();
}
