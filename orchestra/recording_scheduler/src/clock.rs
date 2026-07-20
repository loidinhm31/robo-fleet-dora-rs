use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub trait Clock: Send + Sync {
    fn now_ms(&self) -> i64;
}

#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis()
            .try_into()
            .unwrap_or(i64::MAX)
    }
}

#[derive(Debug, Clone)]
pub struct FakeClock(Arc<Mutex<i64>>);

impl FakeClock {
    pub fn new(now_ms: i64) -> Self {
        Self(Arc::new(Mutex::new(now_ms)))
    }

    pub fn advance_ms(&self, elapsed_ms: i64) {
        *self.0.lock().expect("fake clock lock") += elapsed_ms;
    }
}

impl Clock for FakeClock {
    fn now_ms(&self) -> i64 {
        *self.0.lock().expect("fake clock lock")
    }
}
