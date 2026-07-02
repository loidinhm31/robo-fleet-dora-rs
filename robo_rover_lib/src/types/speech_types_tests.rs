use super::{SpeechTranscription, SttProfile, SttSourceKind, SttState, SttStatus};
use serde_json::json;

#[test]
fn browser_transcription_round_trip_preserves_explicit_nulls() {
    let transcription = SpeechTranscription {
        text: "move forward".into(),
        confidence: None,
        language: "en".into(),
        duration_ms: 1240,
        timestamp: 1_720_000_000_000,
        utterance_id: "utt-browser-1".into(),
        stream_id: "stream-browser-1".into(),
        source_kind: SttSourceKind::Browser,
        entity_id: None,
        target_entity_id: "rover-kiwi".into(),
        profile: SttProfile::EnVadOffline,
    };

    let value = serde_json::to_value(&transcription).unwrap();
    assert_eq!(value["confidence"], serde_json::Value::Null);
    assert_eq!(value["entity_id"], serde_json::Value::Null);
    assert_eq!(value["source_kind"], "browser");
    assert_eq!(value["profile"], "en-vad-offline");

    let round_trip: SpeechTranscription = serde_json::from_value(value).unwrap();
    assert_eq!(round_trip, transcription);
}

#[test]
fn browser_transcription_accepts_omitted_confidence_for_backward_parsing() {
    let value = json!({
        "text": "move forward",
        "language": "en",
        "duration_ms": 1240,
        "timestamp": 1_720_000_000_000i64,
        "utterance_id": "utt-browser-1",
        "stream_id": "stream-browser-1",
        "source_kind": "browser",
        "entity_id": null,
        "target_entity_id": "rover-kiwi",
        "profile": "en-vad-offline"
    });

    let parsed: SpeechTranscription = serde_json::from_value(value).unwrap();
    assert_eq!(parsed.confidence, None);
    assert_eq!(parsed.source_kind, SttSourceKind::Browser);
}

#[test]
fn rover_transcription_round_trip_preserves_identity_and_target() {
    let transcription = SpeechTranscription {
        text: "turn left".into(),
        confidence: Some(0.87),
        language: "en".into(),
        duration_ms: 1320,
        timestamp: 1_720_000_000_100,
        utterance_id: "utt-rover-1".into(),
        stream_id: "stream-rover-1".into(),
        source_kind: SttSourceKind::Rover,
        entity_id: Some("rover-alpha".into()),
        target_entity_id: "rover-alpha".into(),
        profile: SttProfile::EnVadOffline,
    };

    let value = serde_json::to_value(&transcription).unwrap();
    assert!((value["confidence"].as_f64().unwrap() - 0.87).abs() < 1e-6);
    assert_eq!(value["entity_id"], "rover-alpha");
    assert_eq!(value["source_kind"], "rover");

    let round_trip: SpeechTranscription = serde_json::from_value(value).unwrap();
    assert_eq!(round_trip, transcription);
}

#[test]
fn stt_status_round_trip_preserves_state_profile_and_error_shape() {
    let ready = SttStatus {
        state: SttState::Ready,
        profile: SttProfile::ViVadOffline,
        language: "vi".into(),
        timestamp: 1_720_000_000_200,
        error: None,
    };
    let failure = SttStatus {
        state: SttState::Error,
        profile: SttProfile::EnVadOffline,
        language: "en".into(),
        timestamp: 1_720_000_000_300,
        error: Some("model init failed".into()),
    };

    let ready_value = serde_json::to_value(&ready).unwrap();
    let failure_value = serde_json::to_value(&failure).unwrap();

    assert_eq!(ready_value["state"], "ready");
    assert_eq!(ready_value["error"], serde_json::Value::Null);
    assert_eq!(failure_value["state"], "error");
    assert_eq!(failure_value["error"], "model init failed");

    assert_eq!(
        serde_json::from_value::<SttStatus>(ready_value).unwrap(),
        ready
    );
    assert_eq!(
        serde_json::from_value::<SttStatus>(failure_value).unwrap(),
        failure
    );
}
