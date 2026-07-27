use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

pub const DEFAULT_BOOTSTRAP_MS: i64 = 30_000;
pub const DEFAULT_SAFETY_MARGIN_MS: i64 = 5_000;
pub const DEFAULT_SAMPLE_FLOOR: usize = 5;
const MAX_SAMPLES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrewarmEstimate {
    pub sample_count: usize,
    pub estimate_ms: i64,
    pub bootstrap_active: bool,
}

/// A bounded rolling estimator. The scheduler owns timing evidence only; it
/// never decides which workloads the coordinator must start.
#[derive(Debug, Clone)]
pub struct PrewarmEstimator {
    bootstrap_ms: i64,
    safety_margin_ms: i64,
    sample_floor: usize,
    samples: VecDeque<i64>,
    miss_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedPrewarmEstimator {
    pub entity_id: String,
    pub samples: Vec<i64>,
    #[serde(default)]
    pub miss_count: usize,
}

/// Operator-visible, per-reservation prewarm telemetry. It is persisted with
/// the group/estimator and emitted on the scheduler's Dora output whenever a
/// reservation reaches a new power state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrewarmMetrics {
    pub entity_id: String,
    pub reservation_id: String,
    pub sample_count: usize,
    pub estimate_ms: i64,
    pub actual_ready_ms: Option<i64>,
    pub bootstrap_active: bool,
    pub missed: bool,
    pub miss_count: usize,
}

impl Default for PrewarmEstimator {
    fn default() -> Self {
        Self::new(
            DEFAULT_BOOTSTRAP_MS,
            DEFAULT_SAFETY_MARGIN_MS,
            DEFAULT_SAMPLE_FLOOR,
        )
    }
}

impl PrewarmEstimator {
    pub fn new(bootstrap_ms: i64, safety_margin_ms: i64, sample_floor: usize) -> Self {
        Self {
            bootstrap_ms: bootstrap_ms.max(0),
            safety_margin_ms: safety_margin_ms.max(0),
            sample_floor: sample_floor.max(1),
            samples: VecDeque::new(),
            miss_count: 0,
        }
    }

    pub fn observe(&mut self, elapsed_ms: i64, missed: bool) {
        if elapsed_ms < 0 {
            return;
        }
        if self.samples.len() == MAX_SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(elapsed_ms);
        if missed {
            self.miss_count = self.miss_count.saturating_add(1);
        }
    }

    pub fn estimate(&self) -> PrewarmEstimate {
        let sample_count = self.samples.len();
        let bootstrap_active = sample_count < self.sample_floor;
        let rolling_p95 = (!bootstrap_active).then(|| {
            let mut values = self.samples.iter().copied().collect::<Vec<_>>();
            values.sort_unstable();
            values[((values.len() * 95).saturating_add(99) / 100).saturating_sub(1)]
        });
        PrewarmEstimate {
            sample_count,
            estimate_ms: self
                .bootstrap_ms
                .max(rolling_p95.unwrap_or(self.bootstrap_ms))
                .saturating_add(self.safety_margin_ms),
            bootstrap_active,
        }
    }

    pub fn prewarm_at(&self, planned_start_ms: i64) -> i64 {
        planned_start_ms.saturating_sub(self.estimate().estimate_ms)
    }

    pub fn from_samples(samples: impl IntoIterator<Item = i64>, miss_count: usize) -> Self {
        let mut estimator = Self::default();
        for sample in samples {
            estimator.observe(sample, false);
        }
        estimator.miss_count = miss_count;
        estimator
    }

    pub fn samples(&self) -> Vec<i64> {
        self.samples.iter().copied().collect()
    }

    pub fn miss_count(&self) -> usize {
        self.miss_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_is_conservative_until_the_sample_floor() {
        let mut estimator = PrewarmEstimator::new(30_000, 5_000, 3);
        estimator.observe(1_000, false);
        assert_eq!(estimator.estimate().estimate_ms, 35_000);
        estimator.observe(40_000, false);
        estimator.observe(20_000, false);
        assert_eq!(estimator.estimate().estimate_ms, 45_000);
        assert_eq!(estimator.prewarm_at(100_000), 55_000);
    }
}
