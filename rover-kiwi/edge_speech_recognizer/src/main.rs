use dora_node_api::{DoraNode, Event};
use eyre::Result;
use robo_rover_lib::init_tracing;

fn main() -> Result<()> {
    let _guard = init_tracing();
    let (_node, mut events) = DoraNode::init_from_env()?;
    tracing::warn!("edge speech recognition is a placeholder and produces no transcription");

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } => {
                tracing::debug!(input = %id, "edge speech recognizer ignored audio input");
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}
