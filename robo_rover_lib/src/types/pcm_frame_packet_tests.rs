use super::*;
use uuid::Uuid;

fn metadata() -> AudioFrameMetadata {
    AudioFrameMetadata {
        stream_id: Uuid::from_u128(0x00112233445566778899aabbccddeeff),
        frame_id: 42,
        capture_timestamp_ms: 1_717_000_000_000,
        sample_rate: 16_000,
        channels: 1,
        sample_count: 800,
        format: PcmSampleFormat::S16Le,
    }
}

#[test]
fn pcm_packet_round_trips_identity_dimensions_and_payload() {
    let payload = vec![0x5a; 1_600];
    let encoded = PcmFramePacket {
        metadata: metadata(),
        payload: &payload,
    }
    .encode()
    .unwrap();
    let decoded = PcmFramePacket::decode(&encoded).unwrap();
    assert_eq!(decoded.metadata, metadata());
    assert_eq!(decoded.payload, payload);
}

#[test]
fn pcm_packet_rejects_truncated_and_unknown_headers() {
    assert!(PcmFramePacket::decode(b"PCMF").is_err());
    let payload = vec![0; 1_600];
    let mut encoded = PcmFramePacket {
        metadata: metadata(),
        payload: &payload,
    }
    .encode()
    .unwrap();
    encoded[4] = 2;
    assert!(PcmFramePacket::decode(&encoded).is_err());
    encoded[4] = 1;
    encoded[5] = 99;
    assert!(PcmFramePacket::decode(&encoded).is_err());

    let mut encoded = PcmFramePacket {
        metadata: metadata(),
        payload: &payload,
    }
    .encode()
    .unwrap();
    encoded[48..52].copy_from_slice(&1_599u32.to_le_bytes());
    assert!(PcmFramePacket::decode(&encoded).is_err());
}

#[test]
fn pcm_packet_rejects_invalid_dimensions_and_payload_mismatch() {
    let payload = vec![0; 1_600];
    let mut invalid = metadata();
    invalid.channels = 0;
    assert!(PcmFramePacket {
        metadata: invalid,
        payload: &payload,
    }
    .encode()
    .is_err());

    invalid = metadata();
    invalid.channels = 2;
    invalid.sample_count = 801;
    assert!(PcmFramePacket {
        metadata: invalid,
        payload: &payload,
    }
    .encode()
    .is_err());

    invalid = metadata();
    invalid.sample_rate = 7_999;
    assert!(PcmFramePacket {
        metadata: invalid,
        payload: &payload,
    }
    .encode()
    .is_err());

    invalid = metadata();
    invalid.sample_count = 0;
    assert!(PcmFramePacket {
        metadata: invalid,
        payload: &[],
    }
    .encode()
    .is_err());
    assert!(PcmFramePacket {
        metadata: metadata(),
        payload: &payload[..1_599],
    }
    .encode()
    .is_err());

    let mut encoded = PcmFramePacket {
        metadata: metadata(),
        payload: &payload,
    }
    .encode()
    .unwrap();
    encoded.pop();
    assert!(PcmFramePacket::decode(&encoded).is_err());
}

#[test]
fn pcm_packet_accepts_maximum_frame_and_rejects_oversized_frame() {
    let mut maximum = metadata();
    maximum.sample_rate = 192_000;
    maximum.channels = 8;
    maximum.sample_count = 192_000 * 8;
    let payload = vec![0; maximum.expected_payload_len().unwrap()];
    assert!(PcmFramePacket {
        metadata: maximum,
        payload: &payload,
    }
    .encode()
    .is_ok());
    maximum.sample_count += 8;
    assert!(maximum.expected_payload_len().is_err());
}

#[test]
fn sequence_tracker_resets_on_stream_change_and_reports_gaps() {
    let mut tracker = AudioFrameSequenceTracker::default();
    assert!(tracker.observe(metadata()).unwrap().stream_changed);
    let mut next = metadata();
    next.frame_id = 45;
    assert_eq!(tracker.observe(next).unwrap().missing_frames, 2);
    assert!(tracker.observe(next).is_err());
    next.stream_id = Uuid::from_u128(2);
    next.frame_id = 0;
    assert!(tracker.observe(next).unwrap().stream_changed);
}
