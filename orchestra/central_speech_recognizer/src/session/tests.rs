use super::*;
use crate::config::VAD_WINDOW_SIZE;
use crate::segmenter::Segmenter;
use std::sync::Arc;

struct HoldingSegmenter {
    buffered: Vec<f32>,
}

impl Segmenter for HoldingSegmenter {
    fn accept(&mut self, samples: &[f32]) {
        self.buffered.extend_from_slice(samples);
    }

    fn drain(&mut self) -> Vec<Vec<f32>> {
        Vec::new()
    }

    fn flush(&mut self) -> Vec<Vec<f32>> {
        if self.buffered.is_empty() {
            Vec::new()
        } else {
            vec![std::mem::take(&mut self.buffered)]
        }
    }

    fn reset(&mut self) {
        self.buffered.clear();
    }
}

fn manager() -> SessionManager {
    SessionManager::new(Arc::new(|| {
        Ok(Box::new(HoldingSegmenter {
            buffered: Vec::new(),
        }))
    }))
}

fn browser(stream_id: Uuid, target: &str, frame_id: u64, samples: Vec<f32>) -> AudioInput {
    AudioInput {
        identity: SourceIdentity {
            stream_id,
            source_kind: SttSourceKind::Browser,
            entity_id: None,
            target_entity_id: target.into(),
        },
        frame_id,
        sample_rate: 16_000,
        samples,
    }
}

fn rover(entity: &str, stream_id: Uuid, samples: Vec<f32>) -> AudioInput {
    AudioInput {
        identity: SourceIdentity {
            stream_id,
            source_kind: SttSourceKind::Rover,
            entity_id: Some(entity.into()),
            target_entity_id: entity.into(),
        },
        frame_id: 0,
        sample_rate: 16_000,
        samples,
    }
}

#[test]
fn browser_buffers_audio_until_start_and_flushes_padded_final_window() {
    let stream_id = Uuid::new_v4();
    let mut sessions = manager();
    let frame = browser(stream_id, "rover-a", 0, vec![1.0; 100]);
    assert_eq!(
        sessions.accept_browser(frame.clone()).unwrap(),
        FrameOutcome::default()
    );
    let start_outcome = sessions
        .start_browser(frame.identity.clone(), frame.sample_rate)
        .unwrap();
    assert_eq!(start_outcome, FrameOutcome::default());
    let jobs = sessions.stop_browser(stream_id).unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].samples.len(), VAD_WINDOW_SIZE);
    assert_eq!(jobs[0].identity.target_entity_id, "rover-a");
}

#[test]
fn browser_prestart_buffer_is_bounded_and_ordered() {
    let stream_id = Uuid::new_v4();
    let mut sessions = manager();
    for frame_id in 0..MAX_PRESTART_FRAMES as u64 {
        sessions
            .accept_browser(browser(stream_id, "rover-a", frame_id, vec![1.0; 512]))
            .unwrap();
    }
    assert!(sessions
        .accept_browser(browser(
            stream_id,
            "rover-a",
            MAX_PRESTART_FRAMES as u64,
            vec![1.0; 512]
        ))
        .is_err());
    sessions
        .start_browser(
            browser(stream_id, "rover-a", 0, Vec::new()).identity,
            16_000,
        )
        .unwrap();
    let jobs = sessions.stop_browser(stream_id).unwrap();
    assert_eq!(jobs[0].samples.len(), MAX_PRESTART_FRAMES * 512);
}

#[test]
fn browser_prestart_metadata_must_match_start() {
    let stream_id = Uuid::new_v4();
    let mut sessions = manager();
    sessions
        .accept_browser(browser(stream_id, "rover-a", 0, vec![1.0; 512]))
        .unwrap();
    assert!(sessions
        .start_browser(
            browser(stream_id, "rover-b", 0, Vec::new()).identity,
            16_000,
        )
        .is_err());
}

#[test]
fn browser_prestart_frames_survive_session_factory_failure() {
    let stream_id = Uuid::new_v4();
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let factory_attempts = attempts.clone();
    let mut sessions = SessionManager::new(Arc::new(move || {
        if factory_attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            return Err(eyre!("injected segmenter failure"));
        }
        Ok(Box::new(HoldingSegmenter {
            buffered: Vec::new(),
        }))
    }));
    let frame = browser(stream_id, "rover-a", 0, vec![1.0; 512]);
    sessions.accept_browser(frame.clone()).unwrap();

    assert!(sessions
        .start_browser(frame.identity.clone(), frame.sample_rate)
        .is_err());
    assert_eq!(sessions.pending_browsers.len(), 1);
    sessions
        .start_browser(frame.identity, frame.sample_rate)
        .unwrap();
    assert_eq!(
        sessions.stop_browser(stream_id).unwrap()[0].samples.len(),
        512
    );
}

#[test]
fn browser_stop_before_start_discards_pending_frames() {
    let stream_id = Uuid::new_v4();
    let mut sessions = manager();
    sessions
        .accept_browser(browser(stream_id, "rover-a", 0, vec![1.0; 512]))
        .unwrap();
    assert!(sessions.stop_browser(stream_id).unwrap().is_empty());
    assert!(sessions.pending_browsers.is_empty());
}

#[test]
fn sequence_gap_discards_only_the_current_utterance() {
    let stream_id = Uuid::new_v4();
    let mut sessions = manager();
    let first = browser(stream_id, "rover-a", 4, vec![1.0; VAD_WINDOW_SIZE]);
    sessions
        .start_browser(first.identity.clone(), first.sample_rate)
        .unwrap();
    sessions.accept_browser(first).unwrap();
    let outcome = sessions
        .accept_browser(browser(stream_id, "rover-a", 6, vec![2.0; VAD_WINDOW_SIZE]))
        .unwrap();
    assert!(outcome.sequence_reset);
    let jobs = sessions.stop_browser(stream_id).unwrap();
    assert_eq!(jobs.len(), 1);
    assert!(jobs[0].samples.iter().all(|sample| *sample == 2.0));
}

#[test]
fn duplicate_and_regressed_frames_reset_the_utterance() {
    for next_frame in [4, 3] {
        let stream_id = Uuid::new_v4();
        let mut sessions = manager();
        let first = browser(stream_id, "rover-a", 4, vec![1.0; VAD_WINDOW_SIZE]);
        sessions
            .start_browser(first.identity.clone(), first.sample_rate)
            .unwrap();
        sessions.accept_browser(first).unwrap();
        let outcome = sessions
            .accept_browser(browser(
                stream_id,
                "rover-a",
                next_frame,
                vec![2.0; VAD_WINDOW_SIZE],
            ))
            .unwrap();
        assert!(outcome.sequence_reset);
        let jobs = sessions.stop_browser(stream_id).unwrap();
        assert!(jobs[0].samples.iter().all(|sample| *sample == 2.0));
    }
}

#[test]
fn rover_entities_keep_independent_sessions() {
    let mut sessions = manager();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    sessions
        .accept_rover(rover("rover-a", a, vec![1.0; 512]))
        .unwrap();
    sessions
        .accept_rover(rover("rover-b", b, vec![2.0; 512]))
        .unwrap();
    let browser_id = Uuid::new_v4();
    let browser = browser(browser_id, "rover-a", 0, vec![3.0; 512]);
    sessions
        .start_browser(browser.identity.clone(), 16_000)
        .unwrap();
    sessions.accept_browser(browser).unwrap();

    let jobs: Vec<_> = sessions
        .sessions
        .values_mut()
        .flat_map(|session| session.flush())
        .collect();
    assert_eq!(jobs.len(), 3);
    for job in jobs {
        let expected = match job.identity.source_kind {
            SttSourceKind::Browser => 3.0,
            SttSourceKind::Rover if job.identity.entity_id.as_deref() == Some("rover-a") => 1.0,
            SttSourceKind::Rover => 2.0,
        };
        assert!(job.samples.iter().all(|sample| *sample == expected));
    }
}

#[test]
fn rover_stream_replacement_discards_old_session() {
    let mut sessions = manager();
    sessions
        .accept_rover(rover("rover-a", Uuid::new_v4(), vec![1.0; 512]))
        .unwrap();
    let outcome = sessions
        .accept_rover(rover("rover-a", Uuid::new_v4(), vec![2.0; 512]))
        .unwrap();
    assert!(outcome.sequence_reset);
    assert_eq!(sessions.sessions.len(), 1);
    let jobs: Vec<_> = sessions
        .sessions
        .values_mut()
        .flat_map(|session| session.flush())
        .collect();
    assert_eq!(jobs.len(), 1);
    assert!(jobs[0].samples.iter().all(|sample| *sample == 2.0));
}

#[test]
fn flushing_browser_input_retires_all_browser_sessions() {
    let mut sessions = manager();
    for value in [1.0, 2.0] {
        let stream_id = Uuid::new_v4();
        let frame = browser(stream_id, "rover-a", 0, vec![value; 512]);
        sessions
            .start_browser(frame.identity.clone(), frame.sample_rate)
            .unwrap();
        sessions.accept_browser(frame).unwrap();
    }
    let jobs = sessions.flush_all_browsers();
    assert_eq!(jobs.len(), 2);
    assert!(sessions.sessions.is_empty());
}

#[test]
fn browser_resampler_preserves_state_across_frames_and_flushes() {
    let stream_id = Uuid::new_v4();
    let mut sessions = manager();
    let mut first = browser(stream_id, "rover-a", 0, vec![0.25; 2_400]);
    first.sample_rate = 48_000;
    sessions
        .start_browser(first.identity.clone(), first.sample_rate)
        .unwrap();
    sessions.accept_browser(first).unwrap();
    let mut second = browser(stream_id, "rover-a", 1, vec![0.25; 2_400]);
    second.sample_rate = 48_000;
    sessions.accept_browser(second).unwrap();
    let jobs = sessions.stop_browser(stream_id).unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].samples.len() % VAD_WINDOW_SIZE, 0);
    assert!(jobs[0].samples.len() < 4_800);
}

#[test]
fn browser_stream_rejects_target_changes() {
    let stream_id = Uuid::new_v4();
    let mut sessions = manager();
    let frame = browser(stream_id, "rover-a", 0, vec![0.0; 512]);
    sessions
        .start_browser(frame.identity.clone(), 16_000)
        .unwrap();
    assert!(sessions
        .accept_browser(browser(stream_id, "rover-b", 0, vec![0.0; 512]))
        .is_err());
}
