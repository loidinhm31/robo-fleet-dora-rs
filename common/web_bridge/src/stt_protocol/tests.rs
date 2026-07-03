use super::*;

#[test]
fn client_cannot_supply_authoritative_target_or_entity() {
    let target = serde_json::json!({
        "command": "start",
        "stream_id": Uuid::new_v4(),
        "sample_rate": 48_000,
        "channels": 1,
        "target_entity_id": "rover-b"
    });
    assert!(serde_json::from_value::<VoiceCommandControl>(target).is_err());

    let entity = serde_json::json!({
        "stream_id": Uuid::new_v4(),
        "frame_id": 0,
        "sample_rate": 48_000,
        "channels": 1,
        "sample_count": 2,
        "audio_data": [0.0, 0.1],
        "entity_id": "rover-b"
    });
    assert!(serde_json::from_value::<VoiceCommandAudioFrame>(entity).is_err());
}

#[test]
fn client_protocol_accepts_expected_start_stop_and_audio_shapes() {
    let stream_id = Uuid::new_v4();
    let start = serde_json::json!({
        "command": "start",
        "stream_id": stream_id,
        "sample_rate": 48_000,
        "channels": 1
    });
    assert!(matches!(
        serde_json::from_value::<VoiceCommandControl>(start).unwrap(),
        VoiceCommandControl::Start { .. }
    ));

    let stop = serde_json::json!({"command": "stop", "stream_id": stream_id});
    assert!(matches!(
        serde_json::from_value::<VoiceCommandControl>(stop).unwrap(),
        VoiceCommandControl::Stop { .. }
    ));
}
