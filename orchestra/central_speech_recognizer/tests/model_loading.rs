use central_speech_recognizer::config::SttConfig;
use central_speech_recognizer::native::load_models;

#[test]
#[ignore = "requires downloaded Sherpa ASR models"]
fn selected_profile_loads_native_models() {
    let config = SttConfig::from_env().expect("profile configuration should validate");
    let _models = load_models(&config).expect("native Sherpa models should load");
}
