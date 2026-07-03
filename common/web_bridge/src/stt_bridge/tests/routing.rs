use super::*;

#[test]
fn status_cache_and_request_are_authoritative() {
    let bridge = bridge(16);
    assert!(bridge.cached_status().is_none());
    bridge.request_status();
    assert_eq!(deliver_next(&bridge), Some(SttDoraMessage::StatusRequest));
    let status = SttStatus {
        state: SttState::Ready,
        profile: SttProfile::EnVadOffline,
        language: "en".into(),
        timestamp: 1,
        error: None,
    };
    bridge.cache_status(status.clone());
    assert_eq!(bridge.cached_status(), Some(status));
}

#[test]
fn browser_is_private_and_rover_is_broadcast() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    let browser = browser_transcription(id);
    assert_eq!(
        bridge.route_transcription(&browser),
        TranscriptRoute::Browser {
            socket_id: "socket-a".into()
        }
    );

    let mut rover = browser;
    rover.source_kind = SttSourceKind::Rover;
    rover.entity_id = Some("rover-b".into());
    rover.target_entity_id = "rover-b".into();
    assert_eq!(
        bridge.route_transcription(&rover),
        TranscriptRoute::RoverBroadcast
    );
}

#[test]
fn closing_mapping_routes_multiple_final_results() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    deliver_next(&bridge);
    bridge
        .handle_control(
            "socket-a",
            VoiceCommandControl::Stop { stream_id: id },
            None,
            false,
        )
        .unwrap();
    let transcription = browser_transcription(id);

    for _ in 0..2 {
        assert_eq!(
            bridge.route_transcription(&transcription),
            TranscriptRoute::Browser {
                socket_id: "socket-a".into()
            }
        );
    }
}

fn browser_transcription(id: Uuid) -> SpeechTranscription {
    SpeechTranscription::new_browser(
        "stop".into(),
        None,
        400,
        id.to_string(),
        "rover-a".into(),
        SttProfile::EnVadOffline,
    )
}
