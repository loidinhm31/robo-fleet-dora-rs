use central_speech_recognizer::config::SttConfig;
use central_speech_recognizer::native::load_models;
use dora_node_api::{DoraNode, Event};
use eyre::{eyre, Result};
use robo_rover_lib::init_tracing;

fn main() -> Result<()> {
    let _guard = init_tracing();
    let config = SttConfig::from_env().map_err(|error| eyre!(error))?;

    tracing::info!(
        profile = ?config.models.profile,
        language = config.models.language,
        bundle = config.models.bundle_name,
        threads = config.num_threads,
        "validated Sherpa STT configuration"
    );

    let (_node, mut events) = DoraNode::init_from_env()?;
    let _models = load_models(&config)?;
    tracing::warn!(
        "Sherpa STT models loaded, but audio inputs are intentionally disabled until Phase 03"
    );

    while let Some(event) = events.recv() {
        if matches!(event, Event::Stop(_)) {
            break;
        }
    }
    Ok(())
}
