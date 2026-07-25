use crate::{
    journal_storage::{
        io, read_meta, recover, sync_directory, write_frame, write_frames, write_meta, JournalMeta,
        HEADER_BYTES, MAX_RECORD_BYTES,
    },
    JournalAppendClass, JournalCapacity, JournalConfig, JournalHealth, JournalIntent,
    JournalRecord, JOURNAL_VERSION,
};
use std::{
    fs::{self, File, OpenOptions},
    io::{Seek, SeekFrom},
};

pub struct EventJournal {
    config: JournalConfig,
    log: File,
    meta: JournalMeta,
    records: Vec<JournalRecord>,
    log_bytes: u64,
    recovered_torn_tail: bool,
}

impl EventJournal {
    pub fn open(config: JournalConfig) -> Result<Self, String> {
        if config.max_bytes == 0
            || config.max_records == 0
            || config.wake_reserve_bytes >= config.max_bytes
            || config.wake_reserve_records >= config.max_records
        {
            return Err("invalid journal capacity configuration".into());
        }
        fs::create_dir_all(&config.directory).map_err(io)?;
        let meta_path = config.directory.join("power-journal.meta");
        let mut meta = read_meta(&meta_path)?;
        let log_path = config.directory.join("power-events.log");
        let mut log = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&log_path)
            .map_err(io)?;
        let (records, valid_bytes) = recover(&mut log)?;
        let original = log.metadata().map_err(io)?.len();
        if valid_bytes < original {
            log.set_len(valid_bytes).map_err(io)?;
            log.sync_all().map_err(io)?;
        }
        meta.next_sequence = meta
            .next_sequence
            .max(
                records
                    .iter()
                    .map(|item| item.sequence)
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1),
            )
            .max(1);
        meta.last_epoch = meta.last_epoch.max(
            records
                .iter()
                .map(|item| item.event.authority.epoch)
                .max()
                .unwrap_or(0),
        );
        write_meta(&meta_path, &meta)?;
        Ok(Self {
            config,
            log,
            meta,
            records,
            log_bytes: valid_bytes,
            recovered_torn_tail: valid_bytes < original,
        })
    }

    pub fn next_epoch(&self) -> u64 {
        self.meta.last_epoch.saturating_add(1).max(1)
    }
    pub fn pending(&self) -> impl Iterator<Item = &JournalRecord> {
        self.records.iter().filter(|item| {
            !self
                .meta
                .acknowledged
                .iter()
                .any(|id| id == &item.event.event_id)
        })
    }
    pub fn has_pending_intent(&self, intent: JournalIntent) -> bool {
        self.pending().any(|record| record.intent == intent)
    }
    pub fn capacity(&self) -> JournalCapacity {
        JournalCapacity {
            bytes: self.log_bytes,
            records: self.pending().count(),
            unsafe_for_sleep: self.log_bytes
                >= self
                    .config
                    .max_bytes
                    .saturating_sub(self.config.wake_reserve_bytes)
                || self.pending().count()
                    >= self.config.max_records - self.config.wake_reserve_records,
        }
    }
    pub fn health(&self) -> JournalHealth {
        let capacity = self.capacity();
        JournalHealth {
            bytes: capacity.bytes,
            unacknowledged_records: capacity.records,
            unsafe_for_sleep: capacity.unsafe_for_sleep,
            recovered_torn_tail: self.recovered_torn_tail,
        }
    }

    pub fn append(
        &mut self,
        mut record: JournalRecord,
        class: JournalAppendClass,
    ) -> Result<(), String> {
        record.format_version = JOURNAL_VERSION;
        record.sequence = self.meta.next_sequence;
        record.validate()?;
        let payload = serde_json::to_vec(&record).map_err(|error| error.to_string())?;
        if payload.len() > MAX_RECORD_BYTES {
            return Err("journal record exceeds maximum size".into());
        }
        let byte_limit = match class {
            JournalAppendClass::Normal => self.config.max_bytes - self.config.wake_reserve_bytes,
            JournalAppendClass::WakeToSafer => self.config.max_bytes,
        };
        let record_limit = match class {
            JournalAppendClass::Normal => {
                self.config.max_records - self.config.wake_reserve_records
            }
            JournalAppendClass::WakeToSafer => self.config.max_records,
        };
        if self.pending().count() >= record_limit
            || self
                .log_bytes
                .saturating_add((HEADER_BYTES + payload.len()) as u64)
                > byte_limit
        {
            return Err("journal CapacityExceeded".into());
        }
        self.log.seek(SeekFrom::End(0)).map_err(io)?;
        write_frame(&mut self.log, &payload)?;
        self.log.sync_data().map_err(io)?;
        self.log_bytes = self
            .log_bytes
            .saturating_add((HEADER_BYTES + payload.len()) as u64);
        self.meta.next_sequence = self.meta.next_sequence.saturating_add(1);
        self.meta.last_epoch = self.meta.last_epoch.max(record.event.authority.epoch);
        write_meta(
            &self.config.directory.join("power-journal.meta"),
            &self.meta,
        )?;
        self.records.push(record);
        Ok(())
    }

    /// A new boot epoch supersedes an unreplicated earlier boot intent. Other
    /// unacknowledged transition records remain intact.
    pub fn replace_boot_intent(&mut self, record: JournalRecord) -> Result<(), String> {
        if self
            .records
            .iter()
            .any(|item| item.intent == JournalIntent::BootAwake)
        {
            self.records
                .retain(|item| item.intent != JournalIntent::BootAwake);
            self.compact()?;
        }
        self.append(record, JournalAppendClass::WakeToSafer)
    }

    pub fn acknowledge(&mut self, event_id: &str) -> Result<(), String> {
        if self
            .records
            .iter()
            .all(|item| item.event.event_id != event_id)
        {
            return Ok(());
        }
        if !self.meta.acknowledged.iter().any(|item| item == event_id) {
            self.meta.acknowledged.push(event_id.into());
            write_meta(
                &self.config.directory.join("power-journal.meta"),
                &self.meta,
            )?;
        }
        Ok(())
    }

    pub fn compact(&mut self) -> Result<(), String> {
        let retained = self.pending().cloned().collect::<Vec<_>>();
        let temporary = self.config.directory.join("power-events.compacting");
        write_frames(&temporary, &retained)?;
        fs::rename(&temporary, self.config.directory.join("power-events.log")).map_err(io)?;
        sync_directory(&self.config.directory)?;
        self.log = OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.config.directory.join("power-events.log"))
            .map_err(io)?;
        self.log_bytes = self.log.metadata().map_err(io)?.len();
        self.records = retained;
        self.meta.acknowledged.clear();
        write_meta(
            &self.config.directory.join("power-journal.meta"),
            &self.meta,
        )
    }
}
