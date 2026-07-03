use super::*;

#[test]
fn disconnect_enqueues_stop_after_start_was_delivered() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    deliver_next(&bridge);
    assert_eq!(bridge.close_owner("socket-a"), 1);
    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Control(BrowserControlOutput::Stop { stream_id })) if stream_id == id
    ));
}

#[test]
fn delivery_failure_retries_same_message_before_following_audio() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    bridge.handle_audio("socket-a", audio(id, 0)).unwrap();

    let start_message = bridge.pop_message().unwrap();
    bridge.retry_delivery(start_message.clone());
    assert_eq!(bridge.pop_message(), Some(start_message));
    bridge.complete_delivery();
    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Audio { .. })
    ));
}

#[test]
fn explicit_stop_preserves_start_audio_stop_order() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    bridge.handle_audio("socket-a", audio(id, 0)).unwrap();
    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Control(BrowserControlOutput::Start { .. }))
    ));
    bridge
        .handle_control(
            "socket-a",
            VoiceCommandControl::Stop { stream_id: id },
            None,
            false,
        )
        .unwrap();

    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Audio { .. })
    ));
    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Control(BrowserControlOutput::Stop { .. }))
    ));
}

#[test]
fn stop_before_start_delivery_cancels_stream_atomically() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    bridge.handle_audio("socket-a", audio(id, 0)).unwrap();
    bridge
        .handle_control(
            "socket-a",
            VoiceCommandControl::Stop { stream_id: id },
            None,
            false,
        )
        .unwrap();

    assert_eq!(deliver_next(&bridge), None);
    assert!(bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .is_ok());
}

#[test]
fn total_closing_ownership_is_bounded() {
    let bridge = bridge(8);
    for _ in 0..8 {
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
        deliver_next(&bridge);
    }
    assert!(bridge
        .handle_control("socket-a", start(Uuid::new_v4()), Some("rover-a"), true)
        .is_err());
}
