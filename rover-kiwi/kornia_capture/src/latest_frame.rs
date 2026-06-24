use std::{
    sync::{Arc, Condvar, Mutex},
    time::Instant,
};

#[derive(Debug)]
pub struct CapturedFrame {
    pub frame_id: u64,
    pub captured_at: Instant,
    pub capture_timestamp_ms: u64,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl CapturedFrame {
    pub fn new(
        frame_id: u64,
        captured_at: Instant,
        capture_timestamp_ms: u64,
        width: u32,
        height: u32,
        rgb: Vec<u8>,
    ) -> Self {
        Self {
            frame_id,
            captured_at,
            capture_timestamp_ms,
            width,
            height,
            rgb,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LatestFrameCounters {
    pub submitted: u64,
    pub replaced: u64,
    pub taken: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TakeOutcome {
    Frame,
    Woken,
    Closed,
}

#[derive(Debug, Default)]
struct LatestFrameState {
    frame: Option<CapturedFrame>,
    closed: bool,
    wake_generation: u64,
    counters: LatestFrameCounters,
}

#[derive(Debug, Default)]
struct SharedLatestFrame {
    state: Mutex<LatestFrameState>,
    changed: Condvar,
}

#[derive(Debug, Clone, Default)]
pub struct LatestFrameSlot {
    shared: Arc<SharedLatestFrame>,
}

impl LatestFrameSlot {
    pub fn submit(&self, frame: CapturedFrame) -> bool {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("latest frame lock poisoned");
        let replaced = state.frame.replace(frame).is_some();
        state.counters.submitted = state.counters.submitted.saturating_add(1);
        if replaced {
            state.counters.replaced = state.counters.replaced.saturating_add(1);
        }
        self.shared.changed.notify_one();
        replaced
    }

    pub fn wake(&self) {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("latest frame lock poisoned");
        state.wake_generation = state.wake_generation.saturating_add(1);
        self.shared.changed.notify_one();
    }

    pub fn take_next(
        &self,
        seen_wake_generation: &mut u64,
    ) -> (TakeOutcome, Option<CapturedFrame>) {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("latest frame lock poisoned");
        loop {
            if let Some(frame) = state.frame.take() {
                state.counters.taken = state.counters.taken.saturating_add(1);
                return (TakeOutcome::Frame, Some(frame));
            }
            if state.closed {
                return (TakeOutcome::Closed, None);
            }
            if state.wake_generation != *seen_wake_generation {
                *seen_wake_generation = state.wake_generation;
                return (TakeOutcome::Woken, None);
            }

            state = self
                .shared
                .changed
                .wait(state)
                .expect("latest frame lock poisoned while waiting");
        }
    }

    pub fn close(&self) {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("latest frame lock poisoned");
        state.closed = true;
        state.frame = None;
        self.shared.changed.notify_all();
    }

    pub fn counters(&self) -> LatestFrameCounters {
        self.shared
            .state
            .lock()
            .expect("latest frame lock poisoned")
            .counters
    }
}

#[cfg(test)]
mod tests {
    use super::{CapturedFrame, LatestFrameSlot, TakeOutcome};
    use std::time::Instant;

    fn frame(frame_id: u64) -> CapturedFrame {
        CapturedFrame::new(frame_id, Instant::now(), frame_id, 2, 2, vec![0; 12])
    }

    #[test]
    fn latest_frame_replaces_unprocessed_input() {
        let slot = LatestFrameSlot::default();

        assert!(!slot.submit(frame(1)));
        assert!(slot.submit(frame(2)));

        let mut wake = 0;
        let (outcome, frame) = slot.take_next(&mut wake);
        assert_eq!(outcome, TakeOutcome::Frame);
        assert_eq!(frame.unwrap().frame_id, 2);

        let counters = slot.counters();
        assert_eq!(counters.submitted, 2);
        assert_eq!(counters.replaced, 1);
        assert_eq!(counters.taken, 1);
    }

    #[test]
    fn latest_frame_keeps_only_last_of_many_submissions() {
        let slot = LatestFrameSlot::default();

        assert!(!slot.submit(frame(1)));
        assert!(slot.submit(frame(2)));
        assert!(slot.submit(frame(3)));
        assert!(slot.submit(frame(4)));

        let mut wake = 0;
        let (outcome, frame) = slot.take_next(&mut wake);
        assert_eq!(outcome, TakeOutcome::Frame);
        assert_eq!(frame.unwrap().frame_id, 4);

        let counters = slot.counters();
        assert_eq!(counters.submitted, 4);
        assert_eq!(counters.replaced, 3);
        assert_eq!(counters.taken, 1);
    }

    #[test]
    fn latest_frame_close_unblocks_waiter() {
        let slot = LatestFrameSlot::default();
        let waiter = slot.clone();
        let handle = std::thread::spawn(move || {
            let mut wake = 0;
            waiter.take_next(&mut wake).0
        });

        slot.close();

        assert_eq!(handle.join().unwrap(), TakeOutcome::Closed);
    }
}
