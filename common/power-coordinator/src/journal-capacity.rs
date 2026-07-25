use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalCapacity {
    pub bytes: u64,
    pub records: usize,
    pub unsafe_for_sleep: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalHealth {
    pub bytes: u64,
    pub unacknowledged_records: usize,
    pub unsafe_for_sleep: bool,
    pub recovered_torn_tail: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalAppendClass {
    Normal,
    WakeToSafer,
}

#[derive(Debug, Clone)]
pub struct JournalConfig {
    pub directory: PathBuf,
    pub max_bytes: u64,
    pub max_records: usize,
    pub wake_reserve_bytes: u64,
    pub wake_reserve_records: usize,
}
