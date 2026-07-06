use crate::arbiter::SourceArbiter;
use crate::buffers::PlaybackBuffers;
use crate::playback_event::ArbiterEvent;
use crate::protocol::AudioSource;
use crate::protocol::SourceFrame;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

fn frame(source: AudioSource, command_id: Option<String>, samples: Vec<f32>) -> SourceFrame {
    frame_with_id(source, command_id, 0, samples)
}

fn frame_with_id(
    source: AudioSource,
    command_id: Option<String>,
    frame_id: u64,
    samples: Vec<f32>,
) -> SourceFrame {
    SourceFrame {
        source,
        command_id,
        frame_id,
        sample_rate: 48_000,
        samples,
        normalized_samples: 0,
    }
}

#[test]
fn walkie_preempts_tts_and_rejects_new_tts_until_deadline() {
    let buffers = Arc::new(PlaybackBuffers::new(256, 256));
    let mut arbiter = SourceArbiter::new(48_000, buffers, Duration::from_millis(60));
    let now = Instant::now();
    let id = Uuid::new_v4().to_string();
    arbiter
        .accept(frame(AudioSource::Tts, Some(id.clone()), vec![0.5]), now)
        .unwrap();

    let event = arbiter
        .accept(frame(AudioSource::Walkie, None, vec![0.7]), now)
        .unwrap();

    assert_eq!(
        event,
        Some(ArbiterEvent::WalkieStarted {
            interrupted: Some(id)
        })
    );
    assert_eq!(
        arbiter
            .accept(
                frame(
                    AudioSource::Tts,
                    Some(Uuid::new_v4().to_string()),
                    vec![0.2]
                ),
                now
            )
            .unwrap(),
        Some(ArbiterEvent::TtsRejectedWhileWalkie)
    );
}

#[test]
fn one_hundred_sequential_short_utterances_do_not_overrun() {
    let buffers = Arc::new(PlaybackBuffers::new(4_096, 64));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    for _ in 0..100 {
        let id = Uuid::new_v4().to_string();
        arbiter
            .accept(
                frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 16]),
                Instant::now(),
            )
            .unwrap();
        assert_eq!(arbiter.finish_tts(&id, Instant::now()).unwrap(), None);
        while buffers.pop_for_output().is_some() {}
        assert_eq!(
            arbiter.tick(Instant::now()),
            Some(ArbiterEvent::TtsPlaybackCompleted { command_id: id })
        );
        arbiter.prune_command_ids();
    }
    assert_eq!(buffers.dropped_counts(), (0, 0));
    assert!(arbiter.command_ids().is_empty());
}

#[test]
fn preemption_preserves_command_id_before_first_resampled_sample() {
    let buffers = Arc::new(PlaybackBuffers::new(4_096, 256));
    let mut arbiter = SourceArbiter::new(48_000, buffers, Duration::from_millis(60));
    let now = Instant::now();
    let id = Uuid::new_v4().to_string();
    let mut tts = frame(AudioSource::Tts, Some(id.clone()), vec![0.5; 8]);
    tts.sample_rate = 44_100;
    arbiter.accept(tts, now).unwrap();

    let event = arbiter
        .accept(frame(AudioSource::Walkie, None, vec![0.7]), now)
        .unwrap();

    assert_eq!(
        event,
        Some(ArbiterEvent::WalkieStarted {
            interrupted: Some(id)
        })
    );
}

#[test]
fn walkie_authority_ends_exactly_at_hold_and_drops_backlog() {
    let buffers = Arc::new(PlaybackBuffers::new(64, 64));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let now = Instant::now();
    arbiter
        .accept(frame(AudioSource::Walkie, None, vec![0.7]), now)
        .unwrap();

    assert_eq!(arbiter.tick(now + Duration::from_millis(249)), None);
    assert_eq!(
        arbiter.tick(now + Duration::from_millis(250)),
        Some(ArbiterEvent::WalkieEnded)
    );
    assert!(!buffers.walkie_is_active());
    assert!(buffers.walkie_is_empty());
}

#[test]
fn completed_result_flushes_exact_resampler_tail() {
    let buffers = Arc::new(PlaybackBuffers::new(4_096, 64));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let id = Uuid::new_v4().to_string();
    let mut tts = frame(AudioSource::Tts, Some(id.clone()), vec![0.5; 441]);
    tts.sample_rate = 44_100;
    arbiter.accept(tts, Instant::now()).unwrap();

    assert_eq!(arbiter.finish_tts(&id, Instant::now()).unwrap(), None);

    let mut samples = 0;
    while buffers.pop_for_output().is_some() {
        samples += 1;
    }
    assert_eq!(samples, 480);
    assert_eq!(
        arbiter.tick(Instant::now()),
        Some(ArbiterEvent::TtsPlaybackCompleted { command_id: id })
    );
}

#[test]
fn valid_silent_walkie_frame_starts_and_refreshes_authority() {
    let buffers = Arc::new(PlaybackBuffers::new(64, 64));
    let mut arbiter = SourceArbiter::new(48_000, buffers, Duration::from_millis(60));
    let now = Instant::now();

    assert!(matches!(
        arbiter
            .accept(frame(AudioSource::Walkie, None, vec![0.0]), now)
            .unwrap(),
        Some(ArbiterEvent::WalkieStarted { .. })
    ));
    assert_eq!(
        arbiter
            .accept(
                frame(AudioSource::Walkie, None, vec![0.0]),
                now + Duration::from_millis(200)
            )
            .unwrap(),
        None
    );
    assert_eq!(arbiter.tick(now + Duration::from_millis(251)), None);
}

#[test]
fn temporary_tts_fullness_retries_without_clearing_audio() {
    let buffers = Arc::new(PlaybackBuffers::new(8, 8));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let now = Instant::now();
    let id = Uuid::new_v4().to_string();

    assert_eq!(
        arbiter
            .accept(frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 8]), now)
            .unwrap(),
        None
    );
    let event = arbiter
        .accept(
            frame_with_id(AudioSource::Tts, Some(id.clone()), 1, vec![0.2; 8]),
            now,
        )
        .unwrap();

    assert_eq!(event, None);
    assert_eq!(buffers.playback_counts().tts_depth, 8);
    assert_eq!(arbiter.tts_stats().pending_frames, 1);
    assert_eq!(buffers.dropped_counts(), (0, 0));

    while buffers.pop_for_output().is_some() {}
    assert_eq!(arbiter.tick(now + Duration::from_millis(20)), None);
    assert_eq!(buffers.playback_counts().tts_depth, 8);
    assert_eq!(arbiter.tts_stats().pending_frames, 0);
}

#[test]
fn pending_tts_frames_cannot_be_bypassed_by_later_smaller_frame() {
    let buffers = Arc::new(PlaybackBuffers::new(10, 8));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let now = Instant::now();
    let id = Uuid::new_v4().to_string();

    arbiter
        .accept(
            frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 10]),
            now,
        )
        .unwrap();
    arbiter
        .accept(
            frame_with_id(AudioSource::Tts, Some(id.clone()), 1, vec![0.2; 10]),
            now,
        )
        .unwrap();

    for _ in 0..4 {
        buffers.pop_for_output();
    }

    arbiter
        .accept(
            frame_with_id(AudioSource::Tts, Some(id), 2, vec![0.3; 4]),
            now,
        )
        .unwrap();

    assert_eq!(buffers.playback_counts().tts_depth, 6);
    assert_eq!(arbiter.tts_stats().pending_frames, 2);

    while buffers.pop_for_output().is_some() {}
    assert_eq!(arbiter.tick(now + Duration::from_millis(20)), None);
    let older_frame: Vec<f32> = (0..10)
        .map(|_| buffers.pop_for_output().unwrap().1.value)
        .collect();
    assert_eq!(older_frame, vec![0.2; 10]);

    assert_eq!(arbiter.tick(now + Duration::from_millis(40)), None);
    let later_frame: Vec<f32> = (0..4)
        .map(|_| buffers.pop_for_output().unwrap().1.value)
        .collect();
    assert_eq!(later_frame, vec![0.3; 4]);
}

#[test]
fn tts_stall_deadline_is_explicit_failure_and_clears_audio() {
    let buffers = Arc::new(PlaybackBuffers::new(8, 8));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let now = Instant::now();
    let id = Uuid::new_v4().to_string();

    arbiter
        .accept(
            frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 16]),
            now,
        )
        .unwrap();

    assert_eq!(
        arbiter.tick(now + Duration::from_millis(60)),
        Some(ArbiterEvent::TtsPlaybackFailed { command_id: id })
    );
    assert!(buffers.tts_is_empty());
    assert_eq!(buffers.dropped_counts(), (0, 0));
    assert_eq!(arbiter.tts_stats().stall_failures, 1);
    assert_eq!(arbiter.tts_stats().pending_frames_cleared, 1);
    assert_eq!(arbiter.tts_stats().pending_samples_cleared, 16);
}

#[test]
fn fourth_pending_tts_frame_fails_and_clears_pending_audio() {
    let buffers = Arc::new(PlaybackBuffers::new(4, 8));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let now = Instant::now();
    let id = Uuid::new_v4().to_string();

    arbiter
        .accept(frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 4]), now)
        .unwrap();
    for frame_id in 1..=3 {
        assert_eq!(
            arbiter
                .accept(
                    frame_with_id(AudioSource::Tts, Some(id.clone()), frame_id, vec![0.2; 4]),
                    now,
                )
                .unwrap(),
            None
        );
    }

    assert_eq!(
        arbiter
            .accept(
                frame_with_id(AudioSource::Tts, Some(id.clone()), 4, vec![0.3; 4]),
                now,
            )
            .unwrap(),
        Some(ArbiterEvent::TtsPlaybackFailed { command_id: id })
    );
    assert!(buffers.tts_is_empty());
    assert_eq!(arbiter.tts_stats().pending_overflows, 1);
    assert_eq!(arbiter.tts_stats().pending_frames_cleared, 3);
    assert_eq!(arbiter.tts_stats().pending_samples_cleared, 12);
}

#[test]
fn tts_frame_sequence_gap_fails_current_command() {
    let buffers = Arc::new(PlaybackBuffers::new(64, 8));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let now = Instant::now();
    let id = Uuid::new_v4().to_string();

    arbiter
        .accept(frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 4]), now)
        .unwrap();

    assert_eq!(
        arbiter
            .accept(
                frame_with_id(AudioSource::Tts, Some(id.clone()), 2, vec![0.2; 4]),
                now
            )
            .unwrap(),
        Some(ArbiterEvent::TtsPlaybackFailed { command_id: id })
    );
    assert!(buffers.tts_is_empty());
    assert_eq!(arbiter.tts_stats().sequence_failures, 1);
}

#[test]
fn playback_failure_clears_all_queued_sources() {
    let buffers = Arc::new(PlaybackBuffers::new(64, 64));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let id = Uuid::new_v4().to_string();
    arbiter
        .accept(
            frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 8]),
            Instant::now(),
        )
        .unwrap();

    assert_eq!(
        arbiter.fail_playback(),
        Some(ArbiterEvent::TtsPlaybackFailed { command_id: id })
    );
    assert!(buffers.tts_is_empty());
    assert!(buffers.walkie_is_empty());
}

#[test]
fn synthesis_failure_aborts_matching_partial_audio() {
    let buffers = Arc::new(PlaybackBuffers::new(64, 64));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let id = Uuid::new_v4().to_string();
    arbiter
        .accept(
            frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 8]),
            Instant::now(),
        )
        .unwrap();

    arbiter.abort_tts(&id);

    assert!(buffers.tts_is_empty());
    assert!(arbiter.command_ids().values().all(|value| value != &id));
}

#[test]
fn command_after_preemption_can_complete() {
    let buffers = Arc::new(PlaybackBuffers::new(64, 64));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone(), Duration::from_millis(60));
    let interrupted = Uuid::new_v4().to_string();
    arbiter
        .accept(
            frame(AudioSource::Tts, Some(interrupted.clone()), vec![0.1; 8]),
            Instant::now(),
        )
        .unwrap();
    arbiter.abort_tts(&interrupted);

    let next = Uuid::new_v4().to_string();
    arbiter
        .accept(
            frame(AudioSource::Tts, Some(next.clone()), vec![0.2; 8]),
            Instant::now(),
        )
        .unwrap();
    arbiter.finish_tts(&next, Instant::now()).unwrap();
    while buffers.pop_for_output().is_some() {}

    assert_eq!(
        arbiter.tick(Instant::now()),
        Some(ArbiterEvent::TtsPlaybackCompleted { command_id: next })
    );
}
