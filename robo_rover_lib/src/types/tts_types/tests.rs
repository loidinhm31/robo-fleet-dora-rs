use serde::Serialize;
use serde_json::to_value;

use super::validation::validate_external_detail;
use super::*;

fn assert_names<T: Copy + Serialize>(cases: &[(T, &str)]) {
    for (value, expected) in cases {
        assert_eq!(to_value(value).unwrap(), *expected);
    }
}

#[test]
fn every_enum_wire_name_is_stable() {
    assert_names(&[(TtsLanguage::En, "en"), (TtsLanguage::Vi, "vi")]);
    assert_names(&[
        (TtsPriority::Low, "low"),
        (TtsPriority::Normal, "normal"),
        (TtsPriority::High, "high"),
        (TtsPriority::Emergency, "emergency"),
    ]);
    assert_names(&[
        (TtsAckState::Accepted, "accepted"),
        (TtsAckState::Rejected, "rejected"),
    ]);
    assert_names(&[
        (TtsResultState::Completed, "completed"),
        (TtsResultState::Rejected, "rejected"),
        (TtsResultState::Interrupted, "interrupted"),
        (TtsResultState::Failed, "failed"),
    ]);
    assert_names(&[
        (VoiceState::Loading, "loading"),
        (VoiceState::Ready, "ready"),
        (VoiceState::Speaking, "speaking"),
        (VoiceState::Error, "error"),
        (VoiceState::Unavailable, "unavailable"),
    ]);
    assert_names(&[
        (PlaybackSource::Tts, "tts"),
        (PlaybackSource::Walkie, "walkie"),
    ]);
    assert_names(&[
        (PlaybackStateKind::Idle, "idle"),
        (PlaybackStateKind::Active, "active"),
        (PlaybackStateKind::Unavailable, "unavailable"),
    ]);
    assert_names(&[
        (VoiceReasonCode::InvalidCommand, "invalid_command"),
        (VoiceReasonCode::InvalidConfig, "invalid_config"),
        (VoiceReasonCode::StaleRevision, "stale_revision"),
        (VoiceReasonCode::QueueFull, "queue_full"),
        (VoiceReasonCode::VoiceNotReady, "voice_not_ready"),
        (VoiceReasonCode::WalkieActive, "walkie_active"),
        (
            VoiceReasonCode::InterruptedByWalkie,
            "interrupted_by_walkie",
        ),
        (VoiceReasonCode::Cancelled, "cancelled"),
        (VoiceReasonCode::SynthesisFailed, "synthesis_failed"),
        (VoiceReasonCode::PlaybackFailed, "playback_failed"),
        (VoiceReasonCode::PlaybackUnavailable, "playback_unavailable"),
        (VoiceReasonCode::InternalError, "internal_error"),
    ]);
}

#[test]
fn invalid_values_and_external_paths_are_rejected_or_sanitized() {
    let mut config = TtsRuntimeConfig::default();
    config.speed = f32::NAN;
    assert!(config.validate().is_err());
    assert!(serde_json::to_string(&config).is_err());

    let detail = sanitize_external_detail("model failed at /srv/models/voice.onnx\nretrying");
    assert_eq!(
        detail.as_deref(),
        Some("model failed at <redacted-path> retrying")
    );
    assert_eq!(
        sanitize_external_detail("model failed at file:///srv/model.onnx"),
        Some("model failed at <redacted-path>".into())
    );
    assert_eq!(
        sanitize_external_detail("model failed at path:/srv/model.onnx"),
        Some("model failed at <redacted-path>".into())
    );
    assert_eq!(
        sanitize_external_detail("model failed at C:\\models\\voice.onnx"),
        Some("model failed at <redacted-path>".into())
    );
    let exactly_bounded = "x".repeat(256);
    assert_eq!(
        sanitize_external_detail(&exactly_bounded).as_deref(),
        Some(exactly_bounded.as_str())
    );
    assert_eq!(
        sanitize_external_detail(&"x".repeat(257))
            .unwrap()
            .chars()
            .count(),
        256
    );
    assert!(validate_external_detail(&Some("bad file:///srv/model.onnx".into())).is_err());
    assert!(validate_external_detail(&Some("bad path:/srv/model.onnx".into())).is_err());
}

#[test]
fn legacy_internal_command_parses_but_requires_upgrade_before_execution() {
    let command: TtsCommand =
        serde_json::from_str(r#"{"text":"legacy","timestamp":1720000000000,"priority":"normal"}"#)
            .unwrap();
    assert_eq!(command.text, "legacy");
    assert!(command.validate().is_err());
}

#[test]
fn config_and_command_bounds_cover_edge_and_error_paths() {
    let mut config = TtsRuntimeConfig {
        language: TtsLanguage::Vi,
        speaker_id: 9,
        speed: 0.5,
        num_steps: 20,
        volume: 0.0,
    };
    config.validate().unwrap();
    config.speaker_id = 10;
    assert!(config.validate().is_err());
    config.speaker_id = 9;
    config.num_steps = 0;
    assert!(config.validate().is_err());
    config.num_steps = 8;
    config.volume = f32::INFINITY;
    assert!(config.validate().is_err());

    let command = TtsCommand {
        command_id: "550e8400-e29b-41d4-a716-446655440000".into(),
        text: "x".repeat(MAX_TTS_TEXT_CHARS + 1),
        timestamp: 1_720_000_000_000,
        priority: TtsPriority::Normal,
    };
    assert!(command.validate().is_err());
}

#[test]
fn lifecycle_validation_rejects_inconsistent_optional_fields() {
    let mut ack = TtsCommandAck {
        command_id: "550e8400-e29b-41d4-a716-446655440000".into(),
        target_entity_id: "rover-kiwi".into(),
        state: TtsAckState::Accepted,
        timestamp: 1_720_000_000_000,
        reason_code: None,
        detail: Some("unexpected".into()),
    };
    assert!(ack.validate().is_err());
    ack.state = TtsAckState::Rejected;
    assert!(ack.validate().is_err());
    ack.detail = None;
    ack.reason_code = Some(VoiceReasonCode::QueueFull);
    assert!(ack.validate().is_err());

    let invalid_result = TtsCommandResult {
        command_id: "550e8400-e29b-41d4-a716-446655440000".into(),
        entity_id: "rover-kiwi".into(),
        state: TtsResultState::Rejected,
        timestamp: 1_720_000_000_000,
        reason_code: Some(VoiceReasonCode::SynthesisFailed),
        detail: None,
    };
    assert!(invalid_result.validate().is_err());

    let invalid_status = VoiceStatus {
        entity_id: "rover-kiwi".into(),
        state: VoiceState::Unavailable,
        applied_revision: 0,
        applied_config: TtsRuntimeConfig::default(),
        active_command_id: None,
        timestamp: 1_720_000_000_000,
        reason_code: Some(VoiceReasonCode::InterruptedByWalkie),
        detail: None,
    };
    assert!(invalid_status.validate().is_err());

    let invalid_playback = PlaybackState {
        entity_id: "rover-kiwi".into(),
        state: PlaybackStateKind::Active,
        source: Some(PlaybackSource::Tts),
        command_id: None,
        timestamp: 1_720_000_000_000,
        reason_code: None,
        detail: None,
    };
    assert!(invalid_playback.validate().is_err());
}
