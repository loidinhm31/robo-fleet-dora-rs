use super::*;
use serde_json::json;

fn request_id() -> String {
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8".into()
}

#[test]
fn recording_start_golden_json_round_trip() {
    let command = RecordingSessionCommand {
        protocol_version: RECORDING_PROTOCOL_VERSION,
        request_id: request_id(),
        action: RecordingSessionAction::Start {
            entity_id: "rover-a".into(),
            relative_directory: "missions/alpha".into(),
        },
    };
    command.validate().unwrap();
    assert_eq!(
        serde_json::to_value(&command).unwrap(),
        json!({
            "protocol_version": 1, "request_id": request_id(), "action": "start",
            "entity_id": "rover-a", "relative_directory": "missions/alpha"
        })
    );
}

#[test]
fn recording_contract_rejects_malformed_ids_paths_and_versions() {
    let mut command = RecordingSessionCommand {
        protocol_version: RECORDING_PROTOCOL_VERSION,
        request_id: request_id(),
        action: RecordingSessionAction::Start {
            entity_id: "rover-a".into(),
            relative_directory: "safe".into(),
        },
    };
    command.protocol_version = 2;
    assert!(command.validate().is_err());
    command.protocol_version = 1;
    command.request_id = "not-a-uuid".into();
    assert!(command.validate().is_err());
    command.request_id = request_id();
    command.action = RecordingSessionAction::Start {
        entity_id: "rover/a".into(),
        relative_directory: "../escape".into(),
    };
    assert!(command.validate().is_err());
}

#[test]
fn targeted_media_control_is_sparse_and_versioned() {
    let control = TargetedMediaControl {
        protocol_version: 1,
        entity_id: "rover-a".into(),
        camera_enabled: None,
        jpeg_enabled: Some(true),
        microphone_enabled: None,
    };
    control.validate().unwrap();
    assert_eq!(
        serde_json::to_value(&control).unwrap(),
        json!({
            "protocol_version": 1, "entity_id": "rover-a", "camera_enabled": null,
            "jpeg_enabled": true, "microphone_enabled": null
        })
    );
    assert!(TargetedMediaControl {
        jpeg_enabled: None,
        ..control
    }
    .validate()
    .is_err());
}

#[test]
fn status_catalog_and_playback_types_validate_correlated_ids() {
    let recording_id = "550e8400-e29b-41d4-a716-446655440000".to_string();
    let status = RecordingSessionStatus {
        protocol_version: 1,
        request_id: request_id(),
        recording_id: recording_id.clone(),
        entity_id: "rover-a".into(),
        state: RecordingSessionState::Completed,
        started_at_ms: Some(1),
        duration_ms: 10,
        bytes_written: 12,
        reason_code: None,
    };
    status.validate().unwrap();
    let clip = RecordingClip {
        recording_id: recording_id.clone(),
        entity_id: "rover-a".into(),
        relative_path: "missions/a.mp4".into(),
        started_at_ms: 1,
        duration_ms: 10,
        bytes_written: 12,
        video_codec: RecordingVideoCodec::H264,
        audio_codec: RecordingAudioCodec::Aac,
    };
    RecordingClipListResult {
        protocol_version: 1,
        request_id: request_id(),
        clips: vec![clip],
    }
    .validate()
    .unwrap();
    RecordingPlaybackTicketRequest {
        protocol_version: 1,
        request_id: request_id(),
        recording_id,
    }
    .validate()
    .unwrap();
}

#[test]
fn recording_command_result_rejects_inconsistent_terminal_fields() {
    let recording_id = "550e8400-e29b-41d4-a716-446655440000".to_string();
    let accepted = RecordingSessionCommandResult {
        protocol_version: 1,
        request_id: request_id(),
        accepted: true,
        recording_id: Some(recording_id),
        reason_code: None,
        detail: None,
    };
    accepted.validate().unwrap();
    assert!(RecordingSessionCommandResult {
        recording_id: None,
        ..accepted.clone()
    }
    .validate()
    .is_err());
    assert!(RecordingSessionCommandResult {
        accepted: false,
        ..accepted
    }
    .validate()
    .is_err());
}

#[test]
fn recording_status_rejects_inconsistent_lifecycle_fields() {
    let status = RecordingSessionStatus {
        protocol_version: 1,
        request_id: request_id(),
        recording_id: "550e8400-e29b-41d4-a716-446655440000".into(),
        entity_id: "rover-a".into(),
        state: RecordingSessionState::Stopping,
        started_at_ms: Some(1),
        duration_ms: 1,
        bytes_written: 1,
        reason_code: None,
    };
    status.validate().unwrap();
    assert!(RecordingSessionStatus {
        started_at_ms: None,
        ..status.clone()
    }
    .validate()
    .is_err());
    assert!(RecordingSessionStatus {
        reason_code: Some(RecordingReasonCode::Internal),
        ..status
    }
    .validate()
    .is_err());
}

#[test]
fn recording_contract_golden_result_catalog_query_and_ticket_json() {
    let recording_id = "550e8400-e29b-41d4-a716-446655440000";
    let result = RecordingSessionCommandResult {
        protocol_version: 1,
        request_id: request_id(),
        accepted: true,
        recording_id: Some(recording_id.into()),
        reason_code: None,
        detail: None,
    };
    assert_eq!(
        serde_json::to_value(result).unwrap(),
        json!({
            "protocol_version": 1, "request_id": request_id(), "accepted": true,
            "recording_id": recording_id, "reason_code": null, "detail": null
        })
    );
    let query = RecordingClipQuery {
        protocol_version: 1,
        request_id: request_id(),
        entity_id: Some("rover-a".into()),
        relative_directory: Some("missions".into()),
    };
    assert_eq!(
        serde_json::to_value(query).unwrap(),
        json!({
            "protocol_version": 1, "request_id": request_id(), "entity_id": "rover-a",
            "relative_directory": "missions"
        })
    );
    let status = RecordingSessionStatus {
        protocol_version: 1,
        request_id: request_id(),
        recording_id: recording_id.into(),
        entity_id: "rover-a".into(),
        state: RecordingSessionState::Recording,
        started_at_ms: Some(1),
        duration_ms: 2,
        bytes_written: 3,
        reason_code: None,
    };
    assert_eq!(
        serde_json::to_value(status).unwrap(),
        json!({
            "protocol_version": 1, "request_id": request_id(), "recording_id": recording_id,
            "entity_id": "rover-a", "state": "recording", "started_at_ms": 1,
            "duration_ms": 2, "bytes_written": 3, "reason_code": null
        })
    );
    let list = RecordingClipListResult {
        protocol_version: 1,
        request_id: request_id(),
        clips: vec![RecordingClip {
            recording_id: recording_id.into(),
            entity_id: "rover-a".into(),
            relative_path: "missions/a.mp4".into(),
            started_at_ms: 1,
            duration_ms: 2,
            bytes_written: 3,
            video_codec: RecordingVideoCodec::H264,
            audio_codec: RecordingAudioCodec::Aac,
        }],
    };
    assert_eq!(
        serde_json::to_value(list).unwrap(),
        json!({
            "protocol_version": 1, "request_id": request_id(), "clips": [{
                "recording_id": recording_id, "entity_id": "rover-a", "relative_path": "missions/a.mp4",
                "started_at_ms": 1, "duration_ms": 2, "bytes_written": 3,
                "video_codec": "h264", "audio_codec": "aac"
            }]
        })
    );
    let ticket_request = RecordingPlaybackTicketRequest {
        protocol_version: 1,
        request_id: request_id(),
        recording_id: recording_id.into(),
    };
    assert_eq!(
        serde_json::to_value(ticket_request).unwrap(),
        json!({
            "protocol_version": 1, "request_id": request_id(), "recording_id": recording_id
        })
    );
    let ticket = RecordingPlaybackTicketResult {
        protocol_version: 1,
        request_id: request_id(),
        recording_id: recording_id.into(),
        ticket: "opaque.ticket".into(),
        url: "/recordings/playback/opaque.ticket".into(),
        expires_at_ms: 2,
    };
    assert_eq!(
        serde_json::to_value(ticket).unwrap(),
        json!({
            "protocol_version": 1, "request_id": request_id(), "recording_id": recording_id,
            "ticket": "opaque.ticket", "url": "/recordings/playback/opaque.ticket", "expires_at_ms": 2
        })
    );
}
