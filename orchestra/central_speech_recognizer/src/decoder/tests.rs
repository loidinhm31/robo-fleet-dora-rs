use super::*;
use crate::audio_input::SourceIdentity;
use robo_rover_lib::SttSourceKind;

fn job() -> DecodeJob {
    DecodeJob {
        identity: SourceIdentity {
            stream_id: Uuid::new_v4(),
            source_kind: SttSourceKind::Browser,
            entity_id: None,
            target_entity_id: "rover-a".into(),
        },
        samples: vec![0.0; VAD_SAMPLE_RATE as usize],
    }
}

#[test]
fn bounded_submitter_drops_new_job_without_blocking() {
    let (sender, _receiver) = mpsc::sync_channel(1);
    let submitter = DecodeSubmitter {
        sender,
        admission: Arc::new(RwLock::new(true)),
    };
    assert_eq!(submitter.try_submit(job()), SubmitResult::Submitted);
    assert_eq!(submitter.try_submit(job()), SubmitResult::Full);
}

#[test]
fn submitter_reports_disconnected_worker() {
    let (sender, receiver) = mpsc::sync_channel(1);
    drop(receiver);
    let submitter = DecodeSubmitter {
        sender,
        admission: Arc::new(RwLock::new(true)),
    };
    assert_eq!(submitter.try_submit(job()), SubmitResult::Disconnected);
}

#[test]
fn lifecycle_close_rejects_new_decode_jobs() {
    let (sender, _receiver) = mpsc::sync_channel(1);
    let submitter = DecodeSubmitter {
        sender,
        admission: Arc::new(RwLock::new(true)),
    };
    submitter.close_admission();
    assert_eq!(submitter.try_submit(job()), SubmitResult::Disconnected);
}

#[test]
fn transcription_has_no_synthetic_confidence() {
    let transcription = transcription(job(), "move forward".into(), SttProfile::EnVadOffline);
    assert_eq!(transcription.confidence, None);
    assert_eq!(transcription.duration_ms, 1_000);
    assert_eq!(transcription.source_kind, SttSourceKind::Browser);
    assert_eq!(transcription.target_entity_id, "rover-a");
}

struct FakeDecoder(Option<String>);

impl DecoderBackend for FakeDecoder {
    fn decode(&mut self, _samples: &[f32]) -> Option<String> {
        self.0.take()
    }
}

#[test]
fn fake_decoder_boundary_maps_only_nonempty_results() {
    let mut decoder = FakeDecoder(Some("turn left".into()));
    let result = decode_job(&mut decoder, job(), SttProfile::EnVadOffline).unwrap();
    assert_eq!(result.text, "turn left");

    let mut empty = FakeDecoder(None);
    assert!(decode_job(&mut empty, job(), SttProfile::EnVadOffline).is_none());
}
