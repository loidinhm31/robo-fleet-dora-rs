use std::path::{Path, PathBuf};

use eyre::{eyre, Result};
use sherpa_onnx::{
    OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsSupertonicModelConfig,
};

use crate::config::{DeploymentConfig, EXPECTED_SAMPLE_RATE, EXPECTED_SPEAKERS};

const REQUIRED_FILES: [&str; 7] = [
    "duration_predictor.int8.onnx",
    "text_encoder.int8.onnx",
    "vector_estimator.int8.onnx",
    "vocoder.int8.onnx",
    "tts.json",
    "unicode_indexer.bin",
    "voice.bin",
];

#[derive(Debug, Clone)]
pub struct SupertonicModelPaths {
    pub duration_predictor: PathBuf,
    pub text_encoder: PathBuf,
    pub vector_estimator: PathBuf,
    pub vocoder: PathBuf,
    pub tts_json: PathBuf,
    pub unicode_indexer: PathBuf,
    pub voice_style: PathBuf,
}

impl SupertonicModelPaths {
    pub fn validate(model_dir: &Path) -> Result<Self> {
        let missing = REQUIRED_FILES
            .iter()
            .filter(|file| !model_dir.join(file).is_file())
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(eyre!(
                "missing Supertonic model files: {}",
                missing.join(", ")
            ));
        }

        Ok(Self {
            duration_predictor: model_dir.join(REQUIRED_FILES[0]),
            text_encoder: model_dir.join(REQUIRED_FILES[1]),
            vector_estimator: model_dir.join(REQUIRED_FILES[2]),
            vocoder: model_dir.join(REQUIRED_FILES[3]),
            tts_json: model_dir.join(REQUIRED_FILES[4]),
            unicode_indexer: model_dir.join(REQUIRED_FILES[5]),
            voice_style: model_dir.join(REQUIRED_FILES[6]),
        })
    }
}

pub fn create_engine(config: &DeploymentConfig) -> Result<OfflineTts> {
    let paths = SupertonicModelPaths::validate(&config.model_dir)?;
    let tts_config = OfflineTtsConfig {
        model: OfflineTtsModelConfig {
            supertonic: OfflineTtsSupertonicModelConfig {
                duration_predictor: Some(paths.duration_predictor.to_string_lossy().to_string()),
                text_encoder: Some(paths.text_encoder.to_string_lossy().to_string()),
                vector_estimator: Some(paths.vector_estimator.to_string_lossy().to_string()),
                vocoder: Some(paths.vocoder.to_string_lossy().to_string()),
                tts_json: Some(paths.tts_json.to_string_lossy().to_string()),
                unicode_indexer: Some(paths.unicode_indexer.to_string_lossy().to_string()),
                voice_style: Some(paths.voice_style.to_string_lossy().to_string()),
            },
            num_threads: config.num_threads,
            debug: config.debug,
            provider: Some("cpu".to_string()),
            ..Default::default()
        },
        max_num_sentences: 1,
        silence_scale: 0.2,
        ..Default::default()
    };

    let engine =
        OfflineTts::create(&tts_config).ok_or_else(|| eyre!("Supertonic engine failed"))?;
    let sample_rate = engine.sample_rate();
    let speakers = engine.num_speakers();
    if sample_rate != EXPECTED_SAMPLE_RATE {
        return Err(eyre!("unexpected Supertonic sample rate: {sample_rate}"));
    }
    if speakers != EXPECTED_SPEAKERS {
        return Err(eyre!("unexpected Supertonic speaker count: {speakers}"));
    }
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::SystemTime};

    #[test]
    fn validates_required_supertonic_files() {
        let dir = std::env::temp_dir().join(format!(
            "edge-voice-model-test-{:?}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        for file in REQUIRED_FILES {
            fs::write(dir.join(file), b"x").unwrap();
        }
        assert!(SupertonicModelPaths::validate(&dir).is_ok());
        fs::remove_file(dir.join("voice.bin")).unwrap();
        let error = SupertonicModelPaths::validate(&dir)
            .unwrap_err()
            .to_string();
        assert!(error.contains("voice.bin"));
        fs::remove_dir_all(dir).unwrap();
    }
}
