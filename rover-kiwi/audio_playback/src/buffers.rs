use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_queue::ArrayQueue;

pub const SOURCE_IDLE: u8 = 0;
pub const SOURCE_TTS: u8 = 1;
pub const SOURCE_WALKIE: u8 = 2;
const CONSUMPTION_EVENT_CAPACITY: usize = 1_024;
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
    packed_consumption: AtomicU64,
    consumption_events: ArrayQueue<ConsumptionEvent>,
    consumption_event_overflows: AtomicU64,
    dropped_tts: AtomicU64,
    dropped_walkie: AtomicU64,
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
            packed_consumption: AtomicU64::new(0),
            consumption_events: ArrayQueue::new(CONSUMPTION_EVENT_CAPACITY),
            consumption_event_overflows: AtomicU64::new(0),
            dropped_tts: AtomicU64::new(0),
            dropped_walkie: AtomicU64::new(0),
            tts_enqueued: AtomicU64::new(0),
            tts_retired: AtomicU64::new(0),
            stream_errors: AtomicU64::new(0),
        }
    }

    pub fn enqueue_tts(&self, samples: &[f32], token: u64) -> usize {
        let mut written = 0;
        for &value in samples {
            if self.tts.push(BufferedSample { value, token }).is_err() {
                self.dropped_tts.fetch_add(1, Ordering::Relaxed);
            } else {
                written += 1;
                self.tts_enqueued.fetch_add(1, Ordering::Relaxed);
            }
        }
        written
    }

    pub fn enqueue_walkie(&self, samples: &[f32]) {
        for &value in samples {
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
        let packed = (token << SOURCE_BITS) | u64::from(source);
        let previous = self.packed_consumption.swap(packed, Ordering::AcqRel);
        if previous != packed
            && self
                .consumption_events
                .push(ConsumptionEvent { source, token })
                .is_err()
        {
            self.consumption_event_overflows
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn active_consumption(&self) -> (u8, u64) {
        let packed = self.packed_consumption.load(Ordering::Acquire);
        ((packed & 0b11) as u8, packed >> SOURCE_BITS)
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

    pub fn record_stream_error(&self) {
        self.stream_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stream_errors(&self) -> u64 {
        self.stream_errors.load(Ordering::Relaxed)
    }

    pub fn pop_consumption_event(&self) -> Option<ConsumptionEvent> {
        self.consumption_events.pop()
    }

    pub fn consumption_event_overflows(&self) -> u64 {
        self.consumption_event_overflows.load(Ordering::Relaxed)
    }
}
