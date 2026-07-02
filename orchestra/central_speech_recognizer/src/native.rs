use crate::config::{SttConfig, VAD_SAMPLE_RATE, VAD_WINDOW_SIZE};
use eyre::{eyre, Result};
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, VadModelConfig, VoiceActivityDetector,
};

pub struct NativeModels {
    pub vad: VoiceActivityDetector,
    pub recognizer: OfflineRecognizer,
}

pub fn load_models(config: &SttConfig) -> Result<NativeModels> {
    let mut vad_config = VadModelConfig::default();
    vad_config.silero_vad.model = Some(path(&config.models.vad));
    vad_config.silero_vad.threshold = config.vad.threshold;
    vad_config.silero_vad.min_silence_duration = config.vad.min_silence_seconds;
    vad_config.silero_vad.min_speech_duration = config.vad.min_speech_seconds;
    vad_config.silero_vad.max_speech_duration = config.vad.max_speech_seconds;
    vad_config.silero_vad.window_size = VAD_WINDOW_SIZE as i32;
    vad_config.sample_rate = VAD_SAMPLE_RATE;
    vad_config.num_threads = config.num_threads;
    vad_config.provider = Some("cpu".to_string());

    let vad = VoiceActivityDetector::create(&vad_config, config.vad.max_speech_seconds * 2.0)
        .ok_or_else(|| eyre!("failed to initialize Silero VAD"))?;

    let mut recognizer_config = OfflineRecognizerConfig::default();
    recognizer_config.model_config.transducer.encoder = Some(path(&config.models.encoder));
    recognizer_config.model_config.transducer.decoder = Some(path(&config.models.decoder));
    recognizer_config.model_config.transducer.joiner = Some(path(&config.models.joiner));
    recognizer_config.model_config.tokens = Some(path(&config.models.tokens));
    recognizer_config.model_config.num_threads = config.num_threads;
    recognizer_config.model_config.provider = Some("cpu".to_string());
    let recognizer = OfflineRecognizer::create(&recognizer_config)
        .ok_or_else(|| eyre!("failed to initialize offline recognizer"))?;

    Ok(NativeModels { vad, recognizer })
}

fn path(value: &std::path::Path) -> String {
    value.to_string_lossy().into_owned()
}
