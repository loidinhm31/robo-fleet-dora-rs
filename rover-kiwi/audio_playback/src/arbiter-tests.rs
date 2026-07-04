use crate::arbiter::SourceArbiter;
use crate::buffers::PlaybackBuffers;
use crate::playback_event::ArbiterEvent;
use crate::protocol::AudioSource;
use crate::protocol::SourceFrame;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

fn frame(source: AudioSource, command_id: Option<String>, samples: Vec<f32>) -> SourceFrame {
    SourceFrame {
        source,
        command_id,
        sample_rate: 48_000,
        samples,
        normalized_samples: 0,
    }
}

#[test]
fn walkie_preempts_tts_and_rejects_new_tts_until_deadline() {
    let buffers = Arc::new(PlaybackBuffers::new(256, 256));
    let mut arbiter = SourceArbiter::new(48_000, buffers);
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
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone());
    for _ in 0..100 {
        let id = Uuid::new_v4().to_string();
        arbiter
            .accept(
                frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 16]),
                Instant::now(),
            )
            .unwrap();
        assert_eq!(arbiter.finish_tts(&id).unwrap(), None);
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
    let mut arbiter = SourceArbiter::new(48_000, buffers);
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
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone());
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
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone());
    let id = Uuid::new_v4().to_string();
    let mut tts = frame(AudioSource::Tts, Some(id.clone()), vec![0.5; 441]);
    tts.sample_rate = 44_100;
    arbiter.accept(tts, Instant::now()).unwrap();

    assert_eq!(arbiter.finish_tts(&id).unwrap(), None);

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
    let mut arbiter = SourceArbiter::new(48_000, buffers);
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
fn tts_overflow_is_explicit_failure_and_clears_audio() {
    let buffers = Arc::new(PlaybackBuffers::new(8, 8));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone());
    let id = Uuid::new_v4().to_string();

    let event = arbiter
        .accept(
            frame(AudioSource::Tts, Some(id.clone()), vec![0.1; 16]),
            Instant::now(),
        )
        .unwrap();

    assert_eq!(
        event,
        Some(ArbiterEvent::TtsPlaybackFailed { command_id: id })
    );
    assert!(buffers.tts_is_empty());
    assert_eq!(buffers.dropped_counts(), (8, 0));
}

#[test]
fn playback_failure_clears_all_queued_sources() {
    let buffers = Arc::new(PlaybackBuffers::new(64, 64));
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone());
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
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone());
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
    let mut arbiter = SourceArbiter::new(48_000, buffers.clone());
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
    arbiter.finish_tts(&next).unwrap();
    while buffers.pop_for_output().is_some() {}

    assert_eq!(
        arbiter.tick(Instant::now()),
        Some(ArbiterEvent::TtsPlaybackCompleted { command_id: next })
    );
}
