use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use eyre::{eyre, Result};
use robo_rover_lib::{TtsCommand, TtsLanguage, TtsPriority, TtsRuntimeConfig, VoiceReasonCode};
use serde_json::Value;
use sherpa_onnx::GenerationConfig;

use crate::{
    config::DeploymentConfig,
    model::create_engine,
    protocol::{current_time_ms, sanitized_error},
    text::sanitize_text,
};

const CHUNK_SAMPLES_20_MS: usize = 882;
const WORKER_COMMAND_CAPACITY: usize = 2;
const WORKER_EVENT_CAPACITY: usize = 256;

#[derive(Debug)]
pub enum WorkerCommand {
    Synthesize {
        command: TtsCommand,
        config: TtsRuntimeConfig,
        revision: u64,
    },
    Stop,
}

#[derive(Debug)]
pub enum WorkerEvent {
    Loaded {
        sample_rate: i32,
        speakers: i32,
    },
    LoadFailed {
        detail: String,
    },
    Started {
        command_id: String,
        revision: u64,
        config: TtsRuntimeConfig,
    },
    AudioChunk {
        command_id: String,
        priority: TtsPriority,
        frame_id: u64,
        timestamp_ms: u64,
        samples: Vec<f32>,
    },
    Completed {
        command_id: String,
        elapsed_ms: u64,
        samples: usize,
    },
    Interrupted {
        command_id: String,
        reason: VoiceReasonCode,
    },
    Failed {
        command_id: String,
        reason: VoiceReasonCode,
        detail: String,
    },
    Stopped,
}

#[derive(Clone)]
pub struct WorkerHandle {
    tx: SyncSender<WorkerCommand>,
    cancel_requested: Arc<AtomicBool>,
    cancel_reason: Arc<Mutex<Option<VoiceReasonCode>>>,
}

impl WorkerHandle {
    pub fn synthesize(
        &self,
        command: TtsCommand,
        config: TtsRuntimeConfig,
        revision: u64,
    ) -> Result<()> {
        self.cancel_requested.store(false, Ordering::SeqCst);
        if let Ok(mut reason) = self.cancel_reason.lock() {
            *reason = None;
        }
        self.tx
            .send(WorkerCommand::Synthesize {
                command,
                config,
                revision,
            })
            .map_err(|error| eyre!("edge voice worker unavailable: {error}"))
    }

    pub fn cancel(&self, reason: VoiceReasonCode) {
        self.cancel_requested.store(true, Ordering::SeqCst);
        if let Ok(mut slot) = self.cancel_reason.lock() {
            *slot = Some(reason);
        }
    }

    pub fn stop(&self) {
        self.cancel(VoiceReasonCode::Cancelled);
        let _ = self.tx.send(WorkerCommand::Stop);
    }
}

pub fn spawn(config: DeploymentConfig) -> (WorkerHandle, Receiver<WorkerEvent>) {
    let (command_tx, command_rx) = mpsc::sync_channel(WORKER_COMMAND_CAPACITY);
    let (event_tx, event_rx) = mpsc::sync_channel(WORKER_EVENT_CAPACITY);
    let cancel_requested = Arc::new(AtomicBool::new(false));
    let cancel_reason = Arc::new(Mutex::new(None));

    let worker_cancel = cancel_requested.clone();
    let worker_reason = cancel_reason.clone();
    thread::spawn(move || run_worker(config, command_rx, event_tx, worker_cancel, worker_reason));

    (
        WorkerHandle {
            tx: command_tx,
            cancel_requested,
            cancel_reason,
        },
        event_rx,
    )
}

fn run_worker(
    config: DeploymentConfig,
    command_rx: Receiver<WorkerCommand>,
    event_tx: SyncSender<WorkerEvent>,
    cancel_requested: Arc<AtomicBool>,
    cancel_reason: Arc<Mutex<Option<VoiceReasonCode>>>,
) {
    let engine = match create_engine(&config) {
        Ok(engine) => {
            let _ = event_tx.send(WorkerEvent::Loaded {
                sample_rate: engine.sample_rate(),
                speakers: engine.num_speakers(),
            });
            engine
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::LoadFailed {
                detail: sanitized_error(error),
            });
            drain_until_stop(command_rx);
            let _ = event_tx.send(WorkerEvent::Stopped);
            return;
        }
    };

    while let Ok(command) = command_rx.recv() {
        match command {
            WorkerCommand::Synthesize {
                command,
                config,
                revision,
            } => synthesize(
                &engine,
                command,
                config,
                revision,
                &event_tx,
                &cancel_requested,
                &cancel_reason,
            ),
            WorkerCommand::Stop => break,
        }
    }
    let _ = event_tx.send(WorkerEvent::Stopped);
}

fn synthesize(
    engine: &sherpa_onnx::OfflineTts,
    command: TtsCommand,
    config: TtsRuntimeConfig,
    revision: u64,
    event_tx: &SyncSender<WorkerEvent>,
    cancel_requested: &Arc<AtomicBool>,
    cancel_reason: &Arc<Mutex<Option<VoiceReasonCode>>>,
) {
    let command_id = command.command_id.clone();
    let text = match sanitize_text(&command.text) {
        Ok(text) => text,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::Failed {
                command_id,
                reason: VoiceReasonCode::InvalidCommand,
                detail: error,
            });
            return;
        }
    };
    let options = match synthesis_options(&config) {
        Ok(options) => options,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::Failed {
                command_id,
                reason: VoiceReasonCode::InvalidConfig,
                detail: error.to_string(),
            });
            return;
        }
    };

    let _ = event_tx.send(WorkerEvent::Started {
        command_id: command.command_id.clone(),
        revision,
        config: config.clone(),
    });

    let started = Instant::now();
    let event_tx_callback = event_tx.clone();
    let command_id_callback = command.command_id.clone();
    let priority = command.priority;
    let cancel_flag = cancel_requested.clone();
    let callback_error = Arc::new(Mutex::new(None));
    let callback_error_writer = callback_error.clone();
    let chunker = Arc::new(Mutex::new(PcmChunker::new(
        CHUNK_SAMPLES_20_MS,
        options.volume,
    )));
    let callback_chunker = chunker.clone();
    let audio = engine.generate_with_config(
        &text,
        &options.generation,
        Some(move |samples: &[f32], _progress: f32| {
            if let Ok(mut chunker) = callback_chunker.lock() {
                let result = chunker.ingest_cumulative(samples, |frame_id, chunk| {
                    let _ = event_tx_callback.send(WorkerEvent::AudioChunk {
                        command_id: command_id_callback.clone(),
                        priority,
                        frame_id,
                        timestamp_ms: current_time_ms(),
                        samples: chunk,
                    });
                });
                if let Err(error) = result {
                    if let Ok(mut slot) = callback_error_writer.lock() {
                        *slot = Some(error.to_string());
                    }
                    return false;
                }
            }
            !cancel_flag.load(Ordering::SeqCst)
        }),
    );

    let Some(audio) = audio else {
        if let Some(detail) = callback_error.lock().ok().and_then(|guard| guard.clone()) {
            let _ = event_tx.send(WorkerEvent::Failed {
                command_id: command.command_id,
                reason: VoiceReasonCode::SynthesisFailed,
                detail,
            });
            return;
        }
        let reason = cancel_reason
            .lock()
            .ok()
            .and_then(|guard| *guard)
            .unwrap_or(VoiceReasonCode::Cancelled);
        let _ = event_tx.send(WorkerEvent::Interrupted {
            command_id: command.command_id,
            reason,
        });
        return;
    };

    let total_samples = audio.samples().len();
    if cancel_requested.load(Ordering::SeqCst) {
        let reason = cancel_reason
            .lock()
            .ok()
            .and_then(|guard| *guard)
            .unwrap_or(VoiceReasonCode::Cancelled);
        let _ = event_tx.send(WorkerEvent::Interrupted {
            command_id: command.command_id,
            reason,
        });
        return;
    }

    if let Ok(mut chunker) = chunker.lock() {
        if let Err(error) = chunker.ingest_cumulative(audio.samples(), |frame_id, chunk| {
            let _ = event_tx.send(WorkerEvent::AudioChunk {
                command_id: command.command_id.clone(),
                priority: command.priority,
                frame_id,
                timestamp_ms: current_time_ms(),
                samples: chunk,
            });
        }) {
            let _ = event_tx.send(WorkerEvent::Failed {
                command_id: command.command_id,
                reason: VoiceReasonCode::SynthesisFailed,
                detail: error.to_string(),
            });
            return;
        }
        chunker.flush(|frame_id, chunk| {
            let _ = event_tx.send(WorkerEvent::AudioChunk {
                command_id: command.command_id.clone(),
                priority: command.priority,
                frame_id,
                timestamp_ms: current_time_ms(),
                samples: chunk,
            });
        });
    }

    let _ = event_tx.send(WorkerEvent::Completed {
        command_id: command.command_id,
        elapsed_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        samples: total_samples,
    });
}

fn drain_until_stop(command_rx: Receiver<WorkerCommand>) {
    while let Ok(command) = command_rx.recv() {
        if matches!(command, WorkerCommand::Stop) {
            break;
        }
    }
}

#[derive(Debug)]
pub struct SynthesisOptions {
    pub generation: GenerationConfig,
    pub volume: f32,
}

pub fn synthesis_options(config: &TtsRuntimeConfig) -> Result<SynthesisOptions> {
    config.validate().map_err(eyre::Report::msg)?;
    let mut extra = HashMap::new();
    extra.insert(
        "lang".to_string(),
        Value::String(match config.language {
            TtsLanguage::En => "en".to_string(),
            TtsLanguage::Vi => "vi".to_string(),
        }),
    );
    Ok(SynthesisOptions {
        generation: GenerationConfig {
            silence_scale: 0.2,
            speed: config.speed,
            sid: i32::from(config.speaker_id),
            num_steps: i32::from(config.num_steps),
            extra: Some(extra),
            ..Default::default()
        },
        volume: config.volume,
    })
}

#[derive(Debug)]
pub struct PcmChunker {
    chunk_samples: usize,
    pending: Vec<f32>,
    emitted_index: usize,
    next_frame_id: u64,
    volume: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonMonotonicSamples {
    previous: usize,
    current: usize,
}

impl std::fmt::Display for NonMonotonicSamples {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "non-monotonic cumulative TTS samples: previous={}, current={}",
            self.previous, self.current
        )
    }
}

impl std::error::Error for NonMonotonicSamples {}

impl PcmChunker {
    pub fn new(chunk_samples: usize, volume: f32) -> Self {
        Self {
            chunk_samples,
            pending: Vec::with_capacity(chunk_samples),
            emitted_index: 0,
            next_frame_id: 0,
            volume,
        }
    }

    pub fn ingest_cumulative<F>(
        &mut self,
        samples: &[f32],
        mut emit: F,
    ) -> Result<(), NonMonotonicSamples>
    where
        F: FnMut(u64, Vec<f32>),
    {
        if samples.len() < self.emitted_index {
            return Err(NonMonotonicSamples {
                previous: self.emitted_index,
                current: samples.len(),
            });
        }
        if samples.len() == self.emitted_index {
            return Ok(());
        }
        let delta = &samples[self.emitted_index..];
        self.emitted_index = samples.len();
        self.ingest_delta(delta, &mut emit);
        Ok(())
    }

    pub fn flush<F>(&mut self, mut emit: F)
    where
        F: FnMut(u64, Vec<f32>),
    {
        if !self.pending.is_empty() {
            let chunk = std::mem::take(&mut self.pending);
            let frame_id = self.next_frame_id;
            self.next_frame_id = self.next_frame_id.saturating_add(1);
            emit(frame_id, chunk);
        }
    }

    fn ingest_delta<F>(&mut self, samples: &[f32], emit: &mut F)
    where
        F: FnMut(u64, Vec<f32>),
    {
        for sample in samples {
            self.pending.push(scale_sample(*sample, self.volume));
            if self.pending.len() == self.chunk_samples {
                let chunk = std::mem::take(&mut self.pending);
                let frame_id = self.next_frame_id;
                self.next_frame_id = self.next_frame_id.saturating_add(1);
                emit(frame_id, chunk);
                self.pending.reserve(self.chunk_samples);
            }
        }
    }
}

fn scale_sample(sample: f32, volume: f32) -> f32 {
    if !sample.is_finite() {
        return 0.0;
    }
    (sample * volume).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{TtsLanguage, MAX_TTS_SPEAKER_ID};

    #[test]
    fn maps_runtime_config_to_generation_options() {
        let config = TtsRuntimeConfig {
            language: TtsLanguage::Vi,
            speaker_id: MAX_TTS_SPEAKER_ID,
            speed: 1.25,
            num_steps: 12,
            volume: 0.5,
        };
        let options = synthesis_options(&config).unwrap();
        assert_eq!(options.generation.sid, 9);
        assert_eq!(options.generation.speed, 1.25);
        assert_eq!(options.generation.num_steps, 12);
        assert_eq!(options.volume, 0.5);
        assert_eq!(
            options.generation.extra.unwrap().get("lang").unwrap(),
            &Value::String("vi".to_string())
        );
    }

    #[test]
    fn chunker_emits_twenty_ms_chunks_and_final_partial_without_duplicates() {
        let mut chunker = PcmChunker::new(4, 0.5);
        let mut emitted = Vec::new();
        chunker
            .ingest_cumulative(&[1.0, 2.0, 3.0], |id, chunk| emitted.push((id, chunk)))
            .unwrap();
        chunker
            .ingest_cumulative(&[1.0, 2.0, 3.0, 4.0, 5.0], |id, chunk| {
                emitted.push((id, chunk))
            })
            .unwrap();
        chunker.flush(|id, chunk| emitted.push((id, chunk)));
        assert_eq!(emitted.len(), 2);
        assert_eq!(emitted[0], (0, vec![0.5, 1.0, 1.0, 1.0]));
        assert_eq!(emitted[1], (1, vec![1.0]));
    }

    #[test]
    fn chunker_rejects_non_monotonic_cumulative_samples() {
        let mut chunker = PcmChunker::new(4, 1.0);
        chunker
            .ingest_cumulative(&[1.0, 2.0, 3.0], |_, _| {})
            .unwrap();
        let error = chunker
            .ingest_cumulative(&[1.0, 2.0], |_, _| {})
            .unwrap_err();
        assert_eq!(
            error,
            NonMonotonicSamples {
                previous: 3,
                current: 2
            }
        );
    }

    #[test]
    fn rejects_invalid_generation_config() {
        let config = TtsRuntimeConfig {
            volume: f32::NAN,
            ..Default::default()
        };
        assert!(synthesis_options(&config).is_err());
    }
}
