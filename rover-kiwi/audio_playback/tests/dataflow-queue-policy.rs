use std::{fs, path::PathBuf};

use serde_yaml::Value;

fn load_yaml(path: &PathBuf) -> Value {
    let yaml = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_yaml::from_str(&yaml)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn assert_queue_size(doc: &Value, node_id: &str, input_id: &str, expected: i64) {
    let nodes = doc["nodes"]
        .as_sequence()
        .unwrap_or_else(|| panic!("nodes must be a sequence"));
    let node = nodes
        .iter()
        .find(|node| node["id"].as_str() == Some(node_id))
        .unwrap_or_else(|| panic!("missing node `{node_id}`"));
    let queue_size = node["inputs"][input_id]["queue_size"]
        .as_i64()
        .unwrap_or_else(|| panic!("missing queue_size for {node_id}.{input_id}"));
    assert_eq!(
        queue_size, expected,
        "unexpected queue_size for {node_id}.{input_id}"
    );
}

fn assert_lifecycle_wiring(doc: &Value, node_id: &str) {
    assert_queue_size(doc, node_id, "lifecycle_command", 2);
    let node = doc["nodes"]
        .as_sequence()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|node| node["id"].as_str() == Some(node_id))
        })
        .unwrap_or_else(|| panic!("missing node `{node_id}`"));
    assert!(
        node["outputs"].as_sequence().is_some_and(|outputs| outputs
            .iter()
            .any(|output| { output.as_str() == Some("lifecycle_component_status") })),
        "missing lifecycle status output for {node_id}"
    );
}

#[test]
fn queue_policies_remain_explicit_in_rover_and_orchestra_dataflows() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let rover_remote = crate_dir.join("../rover-kiwi-dataflow.yml");
    let rover_direct = crate_dir.join("../rover-kiwi-direct-dataflow.yml");
    let orchestra = crate_dir.join("../../orchestra/orchestra-dataflow.yml");
    let rover_remote = load_yaml(&rover_remote);
    let rover_direct = load_yaml(&rover_direct);
    let orchestra = load_yaml(&orchestra);

    for (node_id, input_id, expected) in [
        ("audio-capture", "playback_state", 8),
        ("edge-voice", "tts_command", 8),
        ("edge-voice", "tts_config_command", 8),
        ("edge-voice", "playback_result", 8),
        ("audio-playback", "walkie_audio", 4),
        ("audio-playback", "tts_audio", 4),
        ("audio-playback", "tts_synthesis_state", 8),
    ] {
        assert_queue_size(&rover_remote, node_id, input_id, expected);
    }

    for (node_id, input_id, expected) in [
        ("audio-capture", "playback_state", 8),
        ("edge-voice", "tts_command", 8),
        ("edge-voice", "tts_config_command", 8),
        ("edge-voice", "playback_result", 8),
        ("audio-playback", "walkie_audio", 4),
        ("audio-playback", "tts_audio", 4),
        ("audio-playback", "tts_synthesis_state", 8),
    ] {
        assert_queue_size(&rover_direct, node_id, input_id, expected);
    }

    for (node_id, input_id, expected) in [
        ("orchestra-bridge", "tts_command_web", 8),
        ("orchestra-bridge", "tts_config_command", 8),
        ("orchestra-bridge", "audio_command_web", 8),
        ("orchestra-bridge", "audio_stream_web", 4),
        ("central-speech-recognizer", "audio_rover", 4),
        ("central-speech-recognizer", "audio_browser", 4),
        ("central-speech-recognizer", "browser_control", 8),
        ("central-speech-recognizer", "stt_status_request", 8),
        ("web-bridge", "voice_status", 8),
        ("web-bridge", "tts_command_result", 8),
    ] {
        assert_queue_size(&orchestra, node_id, input_id, expected);
    }
}

#[test]
fn voice_and_playback_lifecycle_adapters_are_wired_in_both_rover_modes() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for dataflow in [
        load_yaml(&crate_dir.join("../rover-kiwi-dataflow.yml")),
        load_yaml(&crate_dir.join("../rover-kiwi-direct-dataflow.yml")),
    ] {
        assert_lifecycle_wiring(&dataflow, "edge-voice");
        assert_lifecycle_wiring(&dataflow, "audio-playback");
    }
}
