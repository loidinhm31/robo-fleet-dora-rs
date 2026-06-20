use crate::vision_pipeline::PipelineTimings;
use robo_rover_lib::{MetricSnapshot, MetricWindow};
use std::time::Duration;

pub struct PipelineMetricWindows {
    yolo: MetricWindow,
    reid_total: MetricWindow,
    reid_per_detection: MetricWindow,
    cmc: MetricWindow,
    tracker: MetricWindow,
    serialization: MetricWindow,
}

impl PipelineMetricWindows {
    pub fn new() -> Self {
        Self {
            yolo: window(),
            reid_total: window(),
            reid_per_detection: window(),
            cmc: window(),
            tracker: window(),
            serialization: window(),
        }
    }

    pub fn record(&mut self, frame_id: u64, timings: PipelineTimings) {
        record_nonzero(&mut self.yolo, timings.yolo);
        record_nonzero(&mut self.reid_total, timings.reid);
        if timings.reid_count > 0 {
            let per_detection = timings.reid / timings.reid_count as u32;
            for _ in 0..timings.reid_count {
                self.reid_per_detection.record(per_detection, 0);
            }
        }
        record_nonzero(&mut self.cmc, timings.cmc);
        record_nonzero(&mut self.tracker, timings.tracker);
        record_nonzero(&mut self.serialization, timings.serialization);

        log_snapshot("yolo", frame_id, self.yolo.snapshot_if_due());
        log_snapshot("reid_total", frame_id, self.reid_total.snapshot_if_due());
        log_snapshot(
            "reid_per_detection",
            frame_id,
            self.reid_per_detection.snapshot_if_due(),
        );
        log_snapshot("cmc", frame_id, self.cmc.snapshot_if_due());
        log_snapshot("tracker", frame_id, self.tracker.snapshot_if_due());
        log_snapshot(
            "serialization",
            frame_id,
            self.serialization.snapshot_if_due(),
        );
    }
}

fn window() -> MetricWindow {
    MetricWindow::new(Duration::from_secs(5))
}

fn record_nonzero(window: &mut MetricWindow, duration: Duration) {
    if !duration.is_zero() {
        window.record(duration, 0);
    }
}

fn log_snapshot(stage: &str, frame_id: u64, snapshot: Option<MetricSnapshot>) {
    if let Some(snapshot) = snapshot {
        tracing::info!(
            metric = "video_pipeline",
            stage,
            frame_id,
            count = snapshot.count,
            bytes = snapshot.bytes,
            drops = snapshot.drops,
            errors = snapshot.errors,
            p50_us = snapshot.p50_us,
            p95_us = snapshot.p95_us,
            p99_us = snapshot.p99_us,
            max_us = snapshot.max_us
        );
    }
}
