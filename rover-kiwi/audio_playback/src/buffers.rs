use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_queue::ArrayQueue;

pub const SOURCE_IDLE: u8 = 0;
pub const SOURCE_TTS: u8 = 1;
pub const SOURCE_WALKIE: u8 = 2;
const SOURCE_BITS: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct BufferedSample {
    pub value: f32,
    pub token: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumptionEvent {
    pub source: u8,
    pub token: u64,
}

pub struct PlaybackBuffers {
    tts: ArrayQueue<BufferedSample>,
    walkie: ArrayQueue<BufferedSample>,
    walkie_active: AtomicBool,
    current_consumption: AtomicU64,
    interval_activity: AtomicU64,
    dropped_tts: AtomicU64,
    dropped_walkie: AtomicU64,
    cleared_tts: AtomicU64,
    walkie_enqueued: AtomicU64,
    tts_enqueued: AtomicU64,
    tts_retired: AtomicU64,
    stream_errors: AtomicU64,
}

impl PlaybackBuffers {
    pub fn new(tts_capacity: usize, walkie_capacity: usize) -> Self {
        Self {
            tts: ArrayQueue::new(tts_capacity),
            walkie: ArrayQueue::new(walkie_capacity),
            walkie_active: AtomicBool::new(false),
            current_consumption: AtomicU64::new(0),
            interval_activity: AtomicU64::new(0),
            dropped_tts: AtomicU64::new(0),
            dropped_walkie: AtomicU64::new(0),
            cleared_tts: AtomicU64::new(0),
            walkie_enqueued: AtomicU64::new(0),
            tts_enqueued: AtomicU64::new(0),
            tts_retired: AtomicU64::new(0),
            stream_errors: AtomicU64::new(0),
        }
    }

    pub fn try_enqueue_tts_frame(&self, samples: &[f32], token: u64) -> bool {
        if samples.is_empty() {
            return true;
        }
        if self.tts.len().saturating_add(samples.len()) > self.tts.capacity() {
            return false;
        }
        for &value in samples {
            if self.tts.push(BufferedSample { value, token }).is_err() {
                return false;
            }
        }
        self.tts_enqueued
            .fetch_add(samples.len() as u64, Ordering::Relaxed);
        true
    }

    pub fn enqueue_walkie(&self, samples: &[f32]) {
        for &value in samples {
            self.walkie_enqueued.fetch_add(1, Ordering::Relaxed);
            if self
                .walkie
                .force_push(BufferedSample { value, token: 0 })
                .is_some()
            {
                self.dropped_walkie.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn preempt_tts(&self) {
        self.walkie_active.store(true, Ordering::Release);
        self.clear_tts();
    }

    pub fn clear_tts(&self) {
        while self.tts.pop().is_some() {
            self.cleared_tts.fetch_add(1, Ordering::Relaxed);
            self.tts_retired.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn finish_walkie(&self) {
        self.clear_walkie();
        self.walkie_active.store(false, Ordering::Release);
    }

    pub fn clear_walkie(&self) {
        while self.walkie.pop().is_some() {}
    }

    pub fn clear_all(&self) {
        self.clear_tts();
        self.clear_walkie();
        self.walkie_active.store(false, Ordering::Release);
    }

    pub fn pop_for_output(&self) -> Option<(u8, BufferedSample)> {
        if self.walkie_active.load(Ordering::Acquire) {
            self.walkie.pop().map(|sample| (SOURCE_WALKIE, sample))
        } else {
            self.tts.pop().map(|sample| {
                self.tts_retired.fetch_add(1, Ordering::Relaxed);
                (SOURCE_TTS, sample)
            })
        }
    }

    pub fn publish_consumption(&self, source: u8, token: u64) {
        let packed = pack_consumption(source, token);
        self.current_consumption.store(packed, Ordering::Release);
        if source != SOURCE_IDLE {
            self.interval_activity.store(packed, Ordering::Release);
        }
    }

    pub fn active_consumption(&self) -> (u8, u64) {
        unpack_consumption(self.current_consumption.load(Ordering::Acquire))
    }

    pub fn take_interval_consumption(&self) -> ConsumptionEvent {
        let activity = self.interval_activity.swap(0, Ordering::AcqRel);
        let packed = if activity == 0 {
            self.current_consumption.load(Ordering::Acquire)
        } else {
            activity
        };
        let (source, token) = unpack_consumption(packed);
        ConsumptionEvent { source, token }
    }

    pub fn may_report_tts_activity(&self) -> bool {
        let interval = self.interval_activity.load(Ordering::Acquire);
        if unpack_consumption(interval).0 == SOURCE_TTS {
            return true;
        }
        let current = self.current_consumption.load(Ordering::Acquire);
        unpack_consumption(current).0 == SOURCE_TTS
    }

    pub fn walkie_is_active(&self) -> bool {
        self.walkie_active.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub fn walkie_is_empty(&self) -> bool {
        self.walkie.is_empty()
    }

    pub fn tts_is_empty(&self) -> bool {
        self.tts.is_empty()
    }

    pub fn tts_enqueued_total(&self) -> u64 {
        self.tts_enqueued.load(Ordering::Acquire)
    }

    pub fn tts_retired_total(&self) -> u64 {
        self.tts_retired.load(Ordering::Acquire)
    }

    pub fn dropped_counts(&self) -> (u64, u64) {
        (
            self.dropped_tts.load(Ordering::Relaxed),
            self.dropped_walkie.load(Ordering::Relaxed),
        )
    }

    pub fn playback_counts(&self) -> PlaybackCounts {
        PlaybackCounts {
            tts_enqueued: self.tts_enqueued.load(Ordering::Relaxed),
            tts_retired: self.tts_retired.load(Ordering::Relaxed),
            tts_cleared: self.cleared_tts.load(Ordering::Relaxed),
            walkie_enqueued: self.walkie_enqueued.load(Ordering::Relaxed),
            dropped_tts: self.dropped_tts.load(Ordering::Relaxed),
            dropped_walkie: self.dropped_walkie.load(Ordering::Relaxed),
            stream_errors: self.stream_errors.load(Ordering::Relaxed),
            tts_depth: self.tts.len(),
            walkie_depth: self.walkie.len(),
        }
    }

    pub fn record_stream_error(&self) {
        self.stream_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stream_errors(&self) -> u64 {
        self.stream_errors.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaybackCounts {
    pub tts_enqueued: u64,
    pub tts_retired: u64,
    pub tts_cleared: u64,
    pub walkie_enqueued: u64,
    pub dropped_tts: u64,
    pub dropped_walkie: u64,
    pub stream_errors: u64,
    pub tts_depth: usize,
    pub walkie_depth: usize,
}

fn pack_consumption(source: u8, token: u64) -> u64 {
    (token << SOURCE_BITS) | u64::from(source)
}

fn unpack_consumption(packed: u64) -> (u8, u64) {
    ((packed & 0b11) as u8, packed >> SOURCE_BITS)
}
