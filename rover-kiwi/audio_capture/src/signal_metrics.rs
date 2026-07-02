use std::time::{Duration, Instant};

const SILENT_DBFS: f32 = -999.0;

pub(crate) struct SignalSummary {
    pub(crate) sample_count: usize,
    pub(crate) silent_samples: usize,
    pub(crate) sum_squares: f64,
    pub(crate) peak_abs: f32,
    pub(crate) min_sample: f32,
    pub(crate) max_sample: f32,
}

pub(crate) fn analyze_signal(samples: &[f32], silence_threshold: f32) -> SignalSummary {
    let mut summary = SignalSummary {
        sample_count: samples.len(),
        silent_samples: 0,
        sum_squares: 0.0,
        peak_abs: 0.0,
        min_sample: f32::INFINITY,
        max_sample: f32::NEG_INFINITY,
    };

    for &sample in samples {
        let abs = sample.abs();
        summary.sum_squares += f64::from(sample) * f64::from(sample);
        summary.peak_abs = summary.peak_abs.max(abs);
        summary.min_sample = summary.min_sample.min(sample);
        summary.max_sample = summary.max_sample.max(sample);
        if abs < silence_threshold {
            summary.silent_samples += 1;
        }
    }

    summary
}

pub(crate) struct SignalMetricWindow {
    started_at: Instant,
    interval: Duration,
    sum_squares: f64,
    peak_abs: f32,
    silent_samples: usize,
    sample_count: usize,
}

impl SignalMetricWindow {
    pub(crate) fn new(interval: Duration) -> Self {
        Self {
            started_at: Instant::now(),
            interval,
            sum_squares: 0.0,
            peak_abs: 0.0,
            silent_samples: 0,
            sample_count: 0,
        }
    }

    pub(crate) fn record_summary(&mut self, summary: &SignalSummary) {
        self.sum_squares += summary.sum_squares;
        self.peak_abs = self.peak_abs.max(summary.peak_abs);
        self.silent_samples += summary.silent_samples;
        self.sample_count += summary.sample_count;
    }

    pub(crate) fn snapshot(&mut self) -> SignalMetricSnapshot {
        let snapshot = if self.sample_count == 0 {
            SignalMetricSnapshot {
                rms_dbfs: SILENT_DBFS,
                peak_dbfs: SILENT_DBFS,
                silence_pct: 0.0,
            }
        } else {
            SignalMetricSnapshot {
                rms_dbfs: dbfs((self.sum_squares / self.sample_count as f64).sqrt() as f32),
                peak_dbfs: dbfs(self.peak_abs),
                silence_pct: (self.silent_samples as f32 / self.sample_count as f32) * 100.0,
            }
        };
        self.started_at = Instant::now();
        self.sum_squares = 0.0;
        self.peak_abs = 0.0;
        self.silent_samples = 0;
        self.sample_count = 0;
        snapshot
    }

    #[allow(dead_code)]
    pub(crate) fn snapshot_if_due(&mut self) -> Option<SignalMetricSnapshot> {
        (self.started_at.elapsed() >= self.interval).then(|| self.snapshot())
    }
}

pub(crate) struct SignalMetricSnapshot {
    pub(crate) rms_dbfs: f32,
    pub(crate) peak_dbfs: f32,
    pub(crate) silence_pct: f32,
}

pub(crate) struct PreflightSignalProbe {
    target_samples: usize,
    collected_samples: usize,
    sum_squares: f64,
    peak_abs: f32,
    silent_samples: usize,
    emitted: bool,
}

impl PreflightSignalProbe {
    pub(crate) fn new(output_sample_rate: u32, output_channels: u16, duration_ms: u32) -> Self {
        Self {
            target_samples: ((output_sample_rate as u64
                * output_channels as u64
                * duration_ms as u64)
                / 1000) as usize,
            collected_samples: 0,
            sum_squares: 0.0,
            peak_abs: 0.0,
            silent_samples: 0,
            emitted: false,
        }
    }

    pub(crate) fn observe_summary(&mut self, summary: &SignalSummary) {
        if self.emitted {
            return;
        }
        self.sum_squares += summary.sum_squares;
        self.peak_abs = self.peak_abs.max(summary.peak_abs);
        self.collected_samples += summary.sample_count;
        self.silent_samples += summary.silent_samples;
    }

    pub(crate) fn log_if_ready(&mut self) {
        if self.emitted || self.collected_samples < self.target_samples {
            return;
        }
        self.emit("capture_preflight");
    }

    pub(crate) fn log_if_pending(&mut self, stage: &'static str) {
        if self.emitted || self.collected_samples == 0 {
            return;
        }
        self.emit(stage);
    }

    #[cfg(test)]
    pub(crate) fn is_emitted(&self) -> bool {
        self.emitted
    }

    fn emit(&mut self, stage: &'static str) {
        let rms = (self.sum_squares / self.collected_samples as f64).sqrt() as f32;
        let rms_dbfs = dbfs(rms);
        let silence_pct = (self.silent_samples as f32 / self.collected_samples as f32) * 100.0;
        let signal = if self.peak_abs == 0.0 {
            "SILENT"
        } else if rms_dbfs < -40.0 {
            "LOW"
        } else {
            "OK"
        };

        tracing::info!(
            metric = "audio_pipeline",
            stage,
            pre_flight_rms_dbfs = rms_dbfs,
            pre_flight_peak_dbfs = dbfs(self.peak_abs),
            pre_flight_silence_pct = silence_pct,
            observed_samples = self.collected_samples,
            target_samples = self.target_samples,
            signal,
            "Audio capture pre-flight signal analysis"
        );
        self.emitted = true;
    }
}

fn dbfs(level: f32) -> f32 {
    if level > 0.0 {
        20.0 * level.log10()
    } else {
        SILENT_DBFS
    }
}
