use super::*;

fn frame(stream_id: Uuid, frame_id: u64) -> VoiceCommandAudioFrame {
    VoiceCommandAudioFrame {
        stream_id,
        frame_id,
        sample_rate: 48_000,
        channels: 1,
        sample_count: 2,
        audio_data: vec![0.0, 0.1],
    }
}

#[test]
fn ownership_sequence_and_target_are_stable() {
    let now = Instant::now();
    let id = Uuid::new_v4();
    let mut registry = StreamRegistry::default();
    registry
        .start(id, "socket-a".into(), "rover-a".into(), 48_000, 1, now)
        .unwrap();

    assert_eq!(
        registry
            .accept_frame("socket-a", &frame(id, 0), now)
            .unwrap(),
        "rover-a"
    );
    assert_eq!(
        registry
            .accept_frame("socket-b", &frame(id, 1), now)
            .unwrap_err()
            .kind,
        FrameErrorKind::NotOwner
    );
    assert_eq!(
        registry
            .accept_frame("socket-a", &frame(id, 2), now)
            .unwrap_err()
            .kind,
        FrameErrorKind::Sequence
    );
}

#[test]
fn closing_results_route_until_mapping_expires() {
    let now = Instant::now();
    let id = Uuid::new_v4();
    let mut registry = StreamRegistry::default();
    registry
        .start(id, "socket-a".into(), "rover-a".into(), 48_000, 1, now)
        .unwrap();
    assert!(registry
        .close(id, "socket-a", now, Duration::from_secs(10))
        .unwrap());
    assert_eq!(
        registry.route_browser_result(id, "rover-a").as_deref(),
        Some("socket-a")
    );
    assert_eq!(
        registry.route_browser_result(id, "rover-a").as_deref(),
        Some("socket-a")
    );
    assert!(registry.contains(id));
}

#[test]
fn sweep_stops_idle_stream_then_expires_closing_owner() {
    let now = Instant::now();
    let id = Uuid::new_v4();
    let mut registry = StreamRegistry::default();
    registry
        .start(id, "socket-a".into(), "rover-a".into(), 48_000, 1, now)
        .unwrap();

    let first = registry.sweep(
        now + Duration::from_secs(11),
        Duration::from_secs(10),
        Duration::from_secs(5),
    );
    assert_eq!(first.stop_streams, vec![id]);
    assert!(registry.contains(id));

    let second = registry.sweep(
        now + Duration::from_secs(17),
        Duration::from_secs(10),
        Duration::from_secs(5),
    );
    assert_eq!(second.expired_streams, 1);
    assert!(!registry.contains(id));
}
