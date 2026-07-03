use std::time::{Duration, Instant};

pub struct RuntimeMetrics {
    started: Instant,
    last_log: Instant,
    pub frames: u64,
    pub validation_errors: u64,
    pub sequence_resets: u64,
    pub speech_segments: u64,
    pub queue_drops: u64,
}

impl RuntimeMetrics {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            last_log: now,
            frames: 0,
            validation_errors: 0,
            sequence_resets: 0,
            speech_segments: 0,
            queue_drops: 0,
        }
    }

    pub fn log_if_due(&mut self) {
        if self.last_log.elapsed() >= Duration::from_secs(5) {
            self.log("interval");
            self.last_log = Instant::now();
        }
    }

    pub fn log_shutdown(&self) {
        self.log("shutdown");
    }

    fn log(&self, period: &str) {
        tracing::info!(
            metric = "central_stt_runtime",
            period,
            uptime_ms = self.started.elapsed().as_millis(),
            frames = self.frames,
            validation_errors = self.validation_errors,
            sequence_resets = self.sequence_resets,
            speech_segments = self.speech_segments,
            queue_drops = self.queue_drops,
        );
    }
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}
