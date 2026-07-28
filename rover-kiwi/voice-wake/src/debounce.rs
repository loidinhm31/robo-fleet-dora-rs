pub const COOLDOWN_MS: u64 = 10_000;

#[derive(Debug, Default)]
pub struct WakeDebounce {
    last_triggered_at_ms: Option<u64>,
}

impl WakeDebounce {
    pub fn accept(&mut self, now_ms: u64) -> bool {
        if self
            .last_triggered_at_ms
            .is_some_and(|last| now_ms.saturating_sub(last) < COOLDOWN_MS)
        {
            return false;
        }
        self.last_triggered_at_ms = Some(now_ms);
        true
    }

    pub fn reset(&mut self) {
        self.last_triggered_at_ms = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooldown_coalesces_duplicate_wakes() {
        let mut debounce = WakeDebounce::default();
        assert!(debounce.accept(1));
        assert!(!debounce.accept(COOLDOWN_MS));
        assert!(debounce.accept(COOLDOWN_MS + 1));
    }
}
