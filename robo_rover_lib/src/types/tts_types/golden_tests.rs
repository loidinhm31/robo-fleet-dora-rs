use serde_json::to_string;

use super::*;

const COMMAND_ID: &str = "550e8400-e29b-41d4-a716-446655440000";
const TIMESTAMP: u64 = 1_720_000_000_000;

fn default_status() -> VoiceStatus {
    VoiceStatus {
        entity_id: "rover-kiwi".into(),
        state: VoiceState::Ready,
        applied_revision: 0,
        applied_config: TtsRuntimeConfig::default(),
        active_command_id: None,
        timestamp: TIMESTAMP,
        reason_code: None,
        detail: None,
    }
}

#[test]
fn command_and_config_match_typescript_fixtures() {
    let command = TtsCommand {
        command_id: COMMAND_ID.into(),
        text: "Hello rover".into(),
        timestamp: TIMESTAMP,
        priority: TtsPriority::Normal,
    };
    assert_eq!(
        to_string(&command).unwrap(),
        r#"{"command_id":"550e8400-e29b-41d4-a716-446655440000","text":"Hello rover","timestamp":1720000000000,"priority":"normal"}"#
    );
    assert_eq!(
        to_string(&TtsRuntimeConfig::default()).unwrap(),
        r#"{"language":"en","speaker_id":5,"speed":1,"num_steps":8,"volume":0.8}"#
    );
    assert_eq!(
        to_string(&TtsConfigCommand {
            revision: 0,
            config: TtsRuntimeConfig::default(),
        })
        .unwrap(),
        r#"{"revision":0,"config":{"language":"en","speaker_id":5,"speed":1,"num_steps":8,"volume":0.8}}"#
    );
    assert_eq!(
        to_string(&TtsConfigUpdate {
            base_revision: 0,
            config: TtsRuntimeConfig::default(),
        })
        .unwrap(),
        r#"{"base_revision":0,"config":{"language":"en","speaker_id":5,"speed":1,"num_steps":8,"volume":0.8}}"#
    );
    command.validate().unwrap();
}

#[test]
fn lifecycle_optional_fields_match_typescript_fixtures() {
    let accepted = TtsCommandAck {
        command_id: COMMAND_ID.into(),
        target_entity_id: "rover-kiwi".into(),
        state: TtsAckState::Accepted,
        timestamp: TIMESTAMP,
        reason_code: None,
        detail: None,
    };
    assert_eq!(
        to_string(&accepted).unwrap(),
        r#"{"command_id":"550e8400-e29b-41d4-a716-446655440000","target_entity_id":"rover-kiwi","state":"accepted","timestamp":1720000000000}"#
    );
    accepted.validate().unwrap();

    let rejected = TtsCommandResult {
        command_id: COMMAND_ID.into(),
        entity_id: "rover-kiwi".into(),
        state: TtsResultState::Rejected,
        timestamp: TIMESTAMP,
        reason_code: Some(VoiceReasonCode::QueueFull),
        detail: Some("voice queue saturated".into()),
    };
    assert_eq!(
        to_string(&rejected).unwrap(),
        r#"{"command_id":"550e8400-e29b-41d4-a716-446655440000","entity_id":"rover-kiwi","state":"rejected","timestamp":1720000000000,"reason_code":"queue_full","detail":"voice queue saturated"}"#
    );
    rejected.validate().unwrap();

    let interrupted = TtsCommandResult {
        command_id: COMMAND_ID.into(),
        entity_id: "rover-kiwi".into(),
        state: TtsResultState::Interrupted,
        timestamp: TIMESTAMP,
        reason_code: Some(VoiceReasonCode::InterruptedByWalkie),
        detail: Some("live walkie started".into()),
    };
    assert_eq!(
        to_string(&interrupted).unwrap(),
        r#"{"command_id":"550e8400-e29b-41d4-a716-446655440000","entity_id":"rover-kiwi","state":"interrupted","timestamp":1720000000000,"reason_code":"interrupted_by_walkie","detail":"live walkie started"}"#
    );
    interrupted.validate().unwrap();
}

#[test]
fn status_config_state_and_playback_match_typescript_fixtures() {
    let status = default_status();
    let config_state = TtsConfigState {
        desired_revision: 0,
        desired_config: TtsRuntimeConfig::default(),
        applied_rovers: 1,
        active_rovers: 1,
        rovers: vec![status.clone()],
        timestamp: TIMESTAMP,
    };
    assert_eq!(
        to_string(&status).unwrap(),
        r#"{"entity_id":"rover-kiwi","state":"ready","applied_revision":0,"applied_config":{"language":"en","speaker_id":5,"speed":1,"num_steps":8,"volume":0.8},"timestamp":1720000000000}"#
    );
    assert_eq!(
        to_string(&config_state).unwrap(),
        r#"{"desired_revision":0,"desired_config":{"language":"en","speaker_id":5,"speed":1,"num_steps":8,"volume":0.8},"applied_rovers":1,"active_rovers":1,"rovers":[{"entity_id":"rover-kiwi","state":"ready","applied_revision":0,"applied_config":{"language":"en","speaker_id":5,"speed":1,"num_steps":8,"volume":0.8},"timestamp":1720000000000}],"timestamp":1720000000000}"#
    );
    config_state.validate().unwrap();

    let playback = PlaybackState {
        entity_id: "rover-kiwi".into(),
        state: PlaybackStateKind::Active,
        source: Some(PlaybackSource::Tts),
        command_id: Some(COMMAND_ID.into()),
        timestamp: TIMESTAMP,
        reason_code: None,
        detail: None,
    };
    assert_eq!(
        to_string(&playback).unwrap(),
        r#"{"entity_id":"rover-kiwi","state":"active","source":"tts","command_id":"550e8400-e29b-41d4-a716-446655440000","timestamp":1720000000000}"#
    );
    playback.validate().unwrap();

    let speaking = VoiceStatus {
        entity_id: "rover-kiwi".into(),
        state: VoiceState::Speaking,
        applied_revision: 0,
        applied_config: TtsRuntimeConfig::default(),
        active_command_id: Some(COMMAND_ID.into()),
        timestamp: TIMESTAMP,
        reason_code: None,
        detail: None,
    };
    assert_eq!(
        to_string(&speaking).unwrap(),
        r#"{"entity_id":"rover-kiwi","state":"speaking","applied_revision":0,"applied_config":{"language":"en","speaker_id":5,"speed":1,"num_steps":8,"volume":0.8},"active_command_id":"550e8400-e29b-41d4-a716-446655440000","timestamp":1720000000000}"#
    );

    let walkie_preemption = PlaybackState {
        entity_id: "rover-kiwi".into(),
        state: PlaybackStateKind::Active,
        source: Some(PlaybackSource::Walkie),
        command_id: Some(COMMAND_ID.into()),
        timestamp: TIMESTAMP,
        reason_code: Some(VoiceReasonCode::InterruptedByWalkie),
        detail: Some("live walkie started".into()),
    };
    assert_eq!(
        to_string(&walkie_preemption).unwrap(),
        r#"{"entity_id":"rover-kiwi","state":"active","source":"walkie","command_id":"550e8400-e29b-41d4-a716-446655440000","timestamp":1720000000000,"reason_code":"interrupted_by_walkie","detail":"live walkie started"}"#
    );
    walkie_preemption.validate().unwrap();
}
