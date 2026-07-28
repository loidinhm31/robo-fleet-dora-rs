use crate::buffers::{PlaybackBuffers, SOURCE_WAKE_ACK};

#[test]
fn walkie_drops_oldest_to_preserve_live_audio() {
    let buffers = PlaybackBuffers::new(4, 2);
    buffers.enqueue_walkie(&[1.0, 2.0, 3.0]);
    buffers.preempt_tts();

    assert_eq!(buffers.pop_for_output().unwrap().1.value, 2.0);
    assert_eq!(buffers.pop_for_output().unwrap().1.value, 3.0);
    assert_eq!(buffers.dropped_counts(), (0, 1));
}

#[test]
fn walkie_preemption_clears_queued_tts() {
    let buffers = PlaybackBuffers::new(4, 4);
    assert!(buffers.try_enqueue_tts_frame(&[0.1, 0.2], 7));

    buffers.preempt_tts();

    assert!(buffers.tts_is_empty());
    assert!(buffers.walkie_is_active());
}

#[test]
fn tts_buffer_admits_complete_frames_only() {
    let buffers = PlaybackBuffers::new(2, 2);
    assert!(!buffers.try_enqueue_tts_frame(&[1.0, 2.0, 3.0], 1));
    assert!(buffers.tts_is_empty());
    assert_eq!(buffers.dropped_counts(), (0, 0));
    assert!(buffers.try_enqueue_tts_frame(&[1.0, 2.0], 1));
    assert!(!buffers.try_enqueue_tts_frame(&[3.0], 1));
}

#[test]
fn wake_ack_is_bounded_and_has_its_own_consumption_source() {
    let buffers = PlaybackBuffers::new(4, 2);
    assert!(buffers.try_enqueue_wake_ack(&[0.1, 0.2, 0.3]));
    assert_eq!(buffers.pop_for_output().unwrap().0, SOURCE_WAKE_ACK);
    assert!(buffers.wake_ack_is_active());
    while buffers.pop_for_output().is_some() {}
    assert!(buffers.wake_ack_is_empty());
}
