use super::*;

#[test]
fn start_snapshots_target_and_rejects_duplicate_uuid() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    assert!(bridge
        .handle_control("socket-b", start(id), Some("rover-b"), true)
        .is_err());
    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Control(BrowserControlOutput::Start {
            target_entity_id,
            ..
        })) if target_entity_id == "rover-a"
    ));
    bridge.handle_audio("socket-a", audio(id, 0)).unwrap();
    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Audio {
            target_entity_id,
            ..
        }) if target_entity_id == "rover-a"
    ));
}

#[test]
fn ownership_spoof_is_rejected_without_terminating_owner() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    deliver_next(&bridge);
    assert!(bridge.handle_audio("socket-b", audio(id, 0)).is_err());
    assert!(bridge.handle_audio("socket-a", audio(id, 0)).is_ok());
}

#[test]
fn sequence_gap_terminates_stream_and_enqueues_stop() {
    let bridge = bridge(16);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    deliver_next(&bridge);
    assert!(bridge.handle_audio("socket-a", audio(id, 1)).is_err());
    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Control(BrowserControlOutput::Stop { stream_id })) if stream_id == id
    ));
}

#[test]
fn full_queue_drops_newest_and_terminates_affected_stream() {
    let bridge = bridge(4);
    let id = Uuid::new_v4();
    bridge
        .handle_control("socket-a", start(id), Some("rover-a"), true)
        .unwrap();
    deliver_next(&bridge);
    for frame_id in 0..3 {
        bridge
            .handle_audio("socket-a", audio(id, frame_id))
            .unwrap();
    }
    assert!(bridge.handle_audio("socket-a", audio(id, 3)).is_err());
    assert_eq!(bridge.metrics().queue_drops, 1);
    assert!(matches!(
        deliver_next(&bridge),
        Some(SttDoraMessage::Audio { .. })
    ));
    let remaining = std::iter::from_fn(|| deliver_next(&bridge)).collect::<Vec<_>>();
    assert!(matches!(
        remaining.last(),
        Some(SttDoraMessage::Control(BrowserControlOutput::Stop { stream_id })) if *stream_id == id
    ));
}

#[test]
fn start_requires_an_active_selected_target() {
    let bridge = bridge(16);
    assert!(bridge
        .handle_control("socket-a", start(Uuid::new_v4()), None, false)
        .is_err());
    assert!(bridge
        .handle_control("socket-a", start(Uuid::new_v4()), Some("rover-a"), false)
        .is_err());
}
