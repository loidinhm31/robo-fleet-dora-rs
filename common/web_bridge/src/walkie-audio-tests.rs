use super::*;
use dora_node_api::Parameter;
use protocol::frame_duration;

fn wire(stream_id: Uuid, frame_id: u64, sample_rate: u32) -> WalkieAudioFrameMetadata {
    WalkieAudioFrameMetadata {
        protocol_version: 1,
        stream_id: stream_id.to_string(),
        frame_id,
        capture_timestamp_ms: 1,
        sample_rate,
        channels: 1,
        sample_count: ((u64::from(sample_rate) * 20 + 500) / 1_000) as u32,
        format: "f32le".into(),
    }
}

fn payload(samples: usize) -> Vec<Vec<u8>> {
    vec![vec![0_u8; samples * 4]]
}

#[test]
fn preserves_multirate_metadata_and_pcm() {
    for rate in [16_000, 44_100, 48_000] {
        let mut ingress = WalkieIngress::default();
        let id = Uuid::new_v4();
        ingress
            .admit(
                "socket",
                wire(id, 0, rate),
                payload(((rate as u64 * 20 + 500) / 1_000) as usize),
                Instant::now(),
            )
            .unwrap();
        let frame = ingress.pop_front().unwrap();
        assert_eq!(frame.metadata.stream_id, id);
        assert_eq!(frame.metadata.sample_rate, rate);
        assert_eq!(frame_duration(frame.metadata), Duration::from_millis(20));
        assert_eq!(frame.parameters()["frame_id"], Parameter::Integer(0));
    }
}

#[test]
fn rejects_legacy_shape_attachments_and_non_finite_samples() {
    assert!(serde_json::from_value::<WalkieAudioFrameMetadata>(
        serde_json::json!({"audio_data": [0.0]})
    )
    .is_err());
    let mut ingress = WalkieIngress::default();
    let id = Uuid::new_v4();
    assert!(ingress
        .admit("socket", wire(id, 0, 16_000), vec![], Instant::now())
        .is_err());
    let mut nan = payload(320);
    nan[0][0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(ingress
        .admit("socket", wire(id, 0, 16_000), nan, Instant::now())
        .is_err());
    assert_eq!(ingress.metrics().invalid_frames, 2);
}

#[test]
fn counts_duplicates_and_gaps_without_rewriting_identity() {
    let mut ingress = WalkieIngress::default();
    let id = Uuid::new_v4();
    let now = Instant::now();
    ingress
        .admit("socket", wire(id, 0, 16_000), payload(320), now)
        .unwrap();
    assert!(ingress
        .admit("socket", wire(id, 0, 16_000), payload(320), now)
        .is_err());
    ingress
        .admit("socket", wire(id, 3, 16_000), payload(320), now)
        .unwrap();
    let metrics = ingress.metrics();
    assert_eq!(metrics.duplicate_frames, 1);
    assert_eq!(metrics.gap_events, 1);
    assert_eq!(metrics.missing_frames, 2);
}

#[test]
fn drops_oldest_until_queue_is_within_forty_milliseconds() {
    let mut ingress = WalkieIngress::default();
    let id = Uuid::new_v4();
    let now = Instant::now();
    for frame_id in 0..3 {
        ingress
            .admit("socket", wire(id, frame_id, 16_000), payload(320), now)
            .unwrap();
    }
    let metrics = ingress.metrics();
    assert_eq!(metrics.queue_frames, 2);
    assert_eq!(metrics.queue_duration_ms, 40.0);
    assert_eq!(metrics.queue_high_water_ms, 40.0);
    assert_eq!(metrics.overflow_dropped_frames, 1);
    assert_eq!(metrics.overflow_dropped_samples, 320);
    assert_eq!(ingress.pop_front().unwrap().metadata.frame_id, 1);
}

#[test]
fn expires_stream_validation_after_authority_window() {
    let mut ingress = WalkieIngress::default();
    let now = Instant::now();
    let first = Uuid::new_v4();
    ingress
        .admit("socket", wire(first, 0, 16_000), payload(320), now)
        .unwrap();
    let later = now + STREAM_TTL + Duration::from_millis(1);
    assert!(ingress
        .admit("socket", wire(first, 1, 16_000), payload(320), later)
        .is_err());
    ingress
        .admit(
            "socket",
            wire(Uuid::new_v4(), 0, 16_000),
            payload(320),
            later,
        )
        .unwrap();
}
