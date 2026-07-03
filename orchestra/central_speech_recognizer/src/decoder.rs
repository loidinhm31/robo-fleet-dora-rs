use crate::config::VAD_SAMPLE_RATE;
use crate::session::DecodeJob;
use dora_node_api::{arrow::array::BinaryArray, dora_core::config::DataId, DoraNode};
use robo_rover_lib::{SpeechTranscription, SttProfile};
use sherpa_onnx::OfflineRecognizer;
use std::sync::{
    mpsc::{self, Receiver, SyncSender, TrySendError},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use uuid::Uuid;

pub type SharedNode = Arc<Mutex<DoraNode>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitResult {
    Submitted,
    Full,
    Disconnected,
}

pub trait DecoderBackend: Send + 'static {
    fn decode(&mut self, samples: &[f32]) -> Option<String>;
}

pub struct SherpaDecoder {
    recognizer: OfflineRecognizer,
}

impl SherpaDecoder {
    pub fn new(recognizer: OfflineRecognizer) -> Self {
        Self { recognizer }
    }
}

impl DecoderBackend for SherpaDecoder {
    fn decode(&mut self, samples: &[f32]) -> Option<String> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(VAD_SAMPLE_RATE, samples);
        self.recognizer.decode(&stream);
        stream
            .get_result()
            .map(|result| result.text.trim().to_owned())
            .filter(|text| !text.is_empty())
    }
}

#[derive(Clone)]
pub struct DecodeSubmitter {
    sender: SyncSender<DecodeJob>,
}

impl DecodeSubmitter {
    pub fn try_submit(&self, job: DecodeJob) -> SubmitResult {
        match self.sender.try_send(job) {
            Ok(()) => SubmitResult::Submitted,
            Err(TrySendError::Full(_)) => SubmitResult::Full,
            Err(TrySendError::Disconnected(_)) => SubmitResult::Disconnected,
        }
    }
}

pub fn spawn(
    capacity: usize,
    decoder: Box<dyn DecoderBackend>,
    profile: SttProfile,
    node: SharedNode,
) -> std::io::Result<(DecodeSubmitter, JoinHandle<()>)> {
    let (sender, receiver) = mpsc::sync_channel(capacity);
    let handle = thread::Builder::new()
        .name("sherpa-offline-decode".into())
        .spawn(move || worker_loop(receiver, decoder, profile, node))?;
    Ok((DecodeSubmitter { sender }, handle))
}

fn worker_loop(
    receiver: Receiver<DecodeJob>,
    mut decoder: Box<dyn DecoderBackend>,
    profile: SttProfile,
    node: SharedNode,
) {
    let mut decode_count = 0u64;
    let mut empty_count = 0u64;
    let mut latencies = Vec::new();
    while let Ok(job) = receiver.recv() {
        let started = Instant::now();
        let audio_seconds = job.samples.len() as f64 / f64::from(VAD_SAMPLE_RATE);
        let transcription = decode_job(decoder.as_mut(), job, profile);
        let elapsed = started.elapsed();
        decode_count += 1;
        latencies.push(elapsed.as_millis() as u64);
        tracing::debug!(
            decode_ms = elapsed.as_millis(),
            rtf = elapsed.as_secs_f64() / audio_seconds.max(f64::EPSILON),
            "offline speech decode completed"
        );
        match transcription {
            Some(transcription) => send_transcription(&node, transcription),
            None => {
                empty_count += 1;
                tracing::debug!("offline speech decode produced an empty result");
            }
        }
        if decode_count % 32 == 0 {
            log_decode_metrics(decode_count, empty_count, &latencies);
            latencies.clear();
        }
    }
    log_decode_metrics(decode_count, empty_count, &latencies);
}

fn decode_job(
    decoder: &mut dyn DecoderBackend,
    job: DecodeJob,
    profile: SttProfile,
) -> Option<SpeechTranscription> {
    decoder
        .decode(&job.samples)
        .map(|text| transcription(job, text, profile))
}

fn transcription(job: DecodeJob, text: String, profile: SttProfile) -> SpeechTranscription {
    SpeechTranscription {
        text,
        confidence: None,
        language: profile.language_code().into(),
        duration_ms: (job.samples.len() as u64 * 1_000) / VAD_SAMPLE_RATE as u64,
        timestamp: current_timestamp_ms(),
        utterance_id: Uuid::new_v4().to_string(),
        stream_id: job.identity.stream_id.to_string(),
        source_kind: job.identity.source_kind,
        entity_id: job.identity.entity_id,
        target_entity_id: job.identity.target_entity_id,
        profile,
    }
}

fn send_transcription(node: &SharedNode, transcription: SpeechTranscription) {
    let result = serde_json::to_vec(&transcription)
        .map_err(eyre::Report::from)
        .and_then(|json| {
            let array = BinaryArray::from_vec(vec![json.as_slice()]);
            node.lock()
                .map_err(|_| eyre::eyre!("Dora node lock poisoned"))?
                .send_output(
                    DataId::from("transcription".to_owned()),
                    Default::default(),
                    array,
                )
        });
    if let Err(error) = result {
        tracing::error!(%error, "failed to emit speech transcription");
    }
}

fn log_decode_metrics(count: u64, empty: u64, latencies: &[u64]) {
    let mut values = latencies.to_vec();
    values.sort_unstable();
    let percentile = |p: usize| {
        values
            .get(values.len().saturating_sub(1) * p / 100)
            .copied()
            .unwrap_or(0)
    };
    tracing::info!(
        decode_count = count,
        empty_results = empty,
        p50_ms = percentile(50),
        p95_ms = percentile(95),
        p99_ms = percentile(99),
        "speech decode metrics"
    );
}

fn current_timestamp_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
