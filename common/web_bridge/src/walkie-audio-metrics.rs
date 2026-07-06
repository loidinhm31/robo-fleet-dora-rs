#[derive(Clone, Copy, Debug, Default)]
pub struct WalkieMetrics {
    pub received_frames: u64,
    pub invalid_frames: u64,
    pub duplicate_frames: u64,
    pub gap_events: u64,
    pub missing_frames: u64,
    pub overflow_dropped_frames: u64,
    pub overflow_dropped_samples: u64,
    pub forwarded_frames: u64,
    pub send_failures: u64,
    pub queue_frames: usize,
    pub queue_duration_ms: f64,
    pub queue_high_water_ms: f64,
}

pub fn log_walkie_metrics(metrics: WalkieMetrics, stage: &'static str) {
    tracing::info!(
        metric = "walkie_transport_total",
        stage,
        received_frames = metrics.received_frames,
        invalid_frames = metrics.invalid_frames,
        duplicate_frames = metrics.duplicate_frames,
        gap_events = metrics.gap_events,
        missing_frames = metrics.missing_frames,
        overflow_dropped_frames = metrics.overflow_dropped_frames,
        overflow_dropped_samples = metrics.overflow_dropped_samples,
        forwarded_frames = metrics.forwarded_frames,
        send_failures = metrics.send_failures,
        queue_frames = metrics.queue_frames,
        queue_duration_ms = metrics.queue_duration_ms,
        queue_high_water_ms = metrics.queue_high_water_ms,
        "Walkie transport counters"
    );
}
