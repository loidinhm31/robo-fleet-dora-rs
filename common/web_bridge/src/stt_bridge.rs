use crate::stt_ingress::{enqueue_marked_close, handle_audio, handle_control};
use crate::stt_protocol::{SttDoraMessage, VoiceCommandAudioFrame, VoiceCommandControl};
use crate::stt_stream_registry::StreamRegistry;
pub use crate::stt_stream_state::SttBridgeMetrics;
use crate::stt_transcript_routing::route_transcription;
pub use crate::stt_transcript_routing::TranscriptRoute;
use robo_rover_lib::{SpeechTranscription, SttStatus};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct SttBridgeConfig {
    pub queue_capacity: usize,
    pub stream_idle_ttl: Duration,
    pub closing_ttl: Duration,
}

#[derive(Debug)]
pub(crate) struct BridgeInner {
    pub streams: StreamRegistry,
    pub queue: VecDeque<SttDoraMessage>,
    pub status: Option<SttStatus>,
    pub status_request_pending: bool,
    pub delivery_in_flight: bool,
    pub metrics: SttBridgeMetrics,
}

#[derive(Debug)]
pub struct SttBridge {
    config: SttBridgeConfig,
    inner: Mutex<BridgeInner>,
}

impl SttBridge {
    pub fn new(config: SttBridgeConfig) -> Self {
        assert!(config.queue_capacity >= 4);
        Self {
            config,
            inner: Mutex::new(BridgeInner {
                streams: StreamRegistry::default(),
                queue: VecDeque::with_capacity(config.queue_capacity),
                status: None,
                status_request_pending: false,
                delivery_in_flight: false,
                metrics: SttBridgeMetrics::default(),
            }),
        }
    }

    pub fn handle_control(
        &self,
        owner_socket: &str,
        control: VoiceCommandControl,
        selected_target: Option<&str>,
        target_is_active: bool,
    ) -> Result<(), String> {
        let mut inner = self.inner.lock().map_err(|_| "STT bridge lock poisoned")?;
        handle_control(
            &mut inner,
            self.config,
            owner_socket,
            control,
            selected_target,
            target_is_active,
        )
    }

    pub fn handle_audio(
        &self,
        owner_socket: &str,
        frame: VoiceCommandAudioFrame,
    ) -> Result<(), String> {
        let mut inner = self.inner.lock().map_err(|_| "STT bridge lock poisoned")?;
        handle_audio(&mut inner, self.config, owner_socket, frame)
    }

    pub fn close_owner(&self, owner_socket: &str) -> usize {
        let Ok(mut inner) = self.inner.lock() else {
            return 0;
        };
        let ids = inner
            .streams
            .close_owner(owner_socket, Instant::now(), self.config.closing_ttl);
        let count = ids.len();
        for id in ids {
            enqueue_marked_close(&mut inner, id);
        }
        count
    }

    pub fn sweep(&self) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let outcome = inner.streams.sweep(
            Instant::now(),
            self.config.stream_idle_ttl,
            self.config.closing_ttl,
        );
        for id in outcome.stop_streams {
            enqueue_marked_close(&mut inner, id);
        }
        inner.metrics.expired_streams = inner
            .metrics
            .expired_streams
            .saturating_add(outcome.expired_streams as u64);
        if outcome.expired_streams > 0 {
            tracing::info!(
                metric = "browser_stt_expired_owners",
                expired = outcome.expired_streams,
                expired_total = inner.metrics.expired_streams,
                "Expired closing browser speech ownership"
            );
        }
    }

    pub fn pop_message(&self) -> Option<SttDoraMessage> {
        let mut inner = self.inner.lock().ok()?;
        if inner.delivery_in_flight {
            return None;
        }
        if inner.status_request_pending {
            inner.status_request_pending = false;
            inner.delivery_in_flight = true;
            return Some(SttDoraMessage::StatusRequest);
        }
        let message = inner.queue.pop_front()?;
        inner.delivery_in_flight = true;
        Some(message)
    }

    pub fn complete_delivery(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.delivery_in_flight = false;
        }
    }

    pub fn retry_delivery(&self, message: SttDoraMessage) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.delivery_in_flight = false;
            match message {
                SttDoraMessage::StatusRequest => inner.status_request_pending = true,
                message => inner.queue.push_front(message),
            }
        }
    }

    pub fn request_status(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.status_request_pending = true;
            inner.metrics.status_requests = inner.metrics.status_requests.saturating_add(1);
        }
    }

    pub fn cache_status(&self, status: SttStatus) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.status = Some(status);
        }
    }

    pub fn cached_status(&self) -> Option<SttStatus> {
        self.inner.lock().ok()?.status.clone()
    }

    pub fn route_transcription(&self, transcription: &SpeechTranscription) -> TranscriptRoute {
        let Ok(mut inner) = self.inner.lock() else {
            return TranscriptRoute::Drop("STT bridge lock poisoned".into());
        };
        route_transcription(&mut inner, transcription)
    }

    pub fn metrics(&self) -> SttBridgeMetrics {
        self.inner
            .lock()
            .map(|inner| inner.metrics)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests;
