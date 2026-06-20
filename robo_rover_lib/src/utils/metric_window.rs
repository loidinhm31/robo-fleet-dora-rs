use std::time::{Duration, Instant};

const MAX_SAMPLES: usize = 4096;

/// Returns wall-clock age, rejecting timestamps that are in the future.
pub fn capture_age_ms(capture_timestamp_ms: u64) -> Option<u64> {
    let now_ms: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis()
        .try_into()
        .ok()?;
    now_ms.checked_sub(capture_timestamp_ms)
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricSnapshot {
    pub count: u64,
    pub bytes: u64,
    pub drops: u64,
    pub errors: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
}

/// Bounded in-memory samples for low-overhead five-second structured summaries.
pub struct MetricWindow {
    started_at: Instant,
    interval: Duration,
    samples_us: Vec<u64>,
    count: u64,
    bytes: u64,
    drops: u64,
    errors: u64,
}

impl MetricWindow {
    pub fn new(interval: Duration) -> Self {
        Self {
            started_at: Instant::now(),
            interval,
            samples_us: Vec::with_capacity(256),
            count: 0,
            bytes: 0,
            drops: 0,
            errors: 0,
        }
    }

    pub fn record(&mut self, duration: Duration, bytes: usize) {
        self.count += 1;
        self.bytes = self.bytes.saturating_add(bytes as u64);
        if self.samples_us.len() < MAX_SAMPLES {
            self.samples_us
                .push(duration.as_micros().min(u64::MAX as u128) as u64);
        } else {
            self.drops += 1;
        }
    }

    pub fn record_drop(&mut self) {
        self.drops += 1;
    }

    pub fn record_drops(&mut self, count: u64) {
        self.drops = self.drops.saturating_add(count);
    }
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    pub fn snapshot_if_due(&mut self) -> Option<MetricSnapshot> {
        if self.started_at.elapsed() < self.interval {
            return None;
        }
        self.samples_us.sort_unstable();
        let snapshot = MetricSnapshot {
            count: self.count,
            bytes: self.bytes,
            drops: self.drops,
            errors: self.errors,
            p50_us: percentile(&self.samples_us, 50),
            p95_us: percentile(&self.samples_us, 95),
            p99_us: percentile(&self.samples_us, 99),
            max_us: self.samples_us.last().copied().unwrap_or(0),
        };
        self.started_at = Instant::now();
        self.samples_us.clear();
        self.count = 0;
        self.bytes = 0;
        self.drops = 0;
        self.errors = 0;
        Some(snapshot)
    }
}

#[derive(Default)]
pub struct FrameSequenceTracker {
    last_frame_id: Option<u64>,
}

impl FrameSequenceTracker {
    /// Returns missing frame count. Duplicate/regressed IDs are errors.
    pub fn observe(&mut self, frame_id: u64) -> Result<u64, ()> {
        let missing = match self.last_frame_id {
            None => 0,
            Some(previous) if frame_id > previous => frame_id - previous - 1,
            Some(previous) if frame_id == previous => return Err(()),
            Some(_) => return Err(()),
        };
        self.last_frame_id = Some(frame_id);
        Ok(missing)
    }
}

fn percentile(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let rank = (samples.len() * percentile).div_ceil(100);
    let index = rank.saturating_sub(1).min(samples.len() - 1);
    samples[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reports_distributions_and_resets_window() {
        let mut window = MetricWindow::new(Duration::ZERO);
        for sample in 1..=100 {
            window.record(Duration::from_micros(sample), sample as usize);
        }
        window.record_drop();
        window.record_error();
        let snapshot = window.snapshot_if_due().unwrap();
        assert_eq!(
            (
                snapshot.count,
                snapshot.bytes,
                snapshot.drops,
                snapshot.errors
            ),
            (100, 5050, 1, 1)
        );
        assert_eq!(
            (
                snapshot.p50_us,
                snapshot.p95_us,
                snapshot.p99_us,
                snapshot.max_us
            ),
            (50, 95, 99, 100)
        );
        assert_eq!(window.snapshot_if_due().unwrap().count, 0);
    }

    #[test]
    fn frame_sequence_reports_gaps_and_rejects_regression() {
        let mut sequence = FrameSequenceTracker::default();
        assert_eq!(sequence.observe(10), Ok(0));
        assert_eq!(sequence.observe(13), Ok(2));
        assert_eq!(sequence.observe(13), Err(()));
        assert_eq!(sequence.observe(12), Err(()));
        assert_eq!(sequence.observe(13), Err(()));
        assert_eq!(sequence.observe(14), Ok(0));
    }

    #[test]
    fn capture_age_rejects_future_timestamp() {
        assert_eq!(capture_age_ms(u64::MAX), None);
        assert!(capture_age_ms(0).is_some());
    }
}
