use crate::{JournalRecord, JOURNAL_VERSION};
use crc32fast::hash;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

pub const MAGIC: &[u8; 4] = b"RPCJ";
pub const HEADER_BYTES: usize = 13;
pub const MAX_RECORD_BYTES: usize = 64 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub struct JournalMeta {
    pub format_version: u8,
    pub last_epoch: u64,
    pub next_sequence: u64,
    pub acknowledged: Vec<String>,
}

pub fn recover(log: &mut File) -> Result<(Vec<JournalRecord>, u64), String> {
    log.seek(SeekFrom::Start(0)).map_err(io)?;
    let mut bytes = Vec::new();
    log.read_to_end(&mut bytes).map_err(io)?;
    let mut offset = 0;
    let mut records = Vec::new();
    while offset < bytes.len() {
        if bytes.len() - offset < HEADER_BYTES {
            break;
        }
        if &bytes[offset..offset + 4] != MAGIC || bytes[offset + 4] != JOURNAL_VERSION {
            return Err("journal corruption before final record".into());
        }
        let length = u32::from_le_bytes(bytes[offset + 5..offset + 9].try_into().unwrap()) as usize;
        if length > MAX_RECORD_BYTES {
            return Err("journal record length is invalid".into());
        }
        let end = offset + HEADER_BYTES + length;
        if end > bytes.len() {
            break;
        }
        let checksum = u32::from_le_bytes(bytes[offset + 9..offset + 13].try_into().unwrap());
        let payload = &bytes[offset + HEADER_BYTES..end];
        if hash(payload) != checksum && end < bytes.len() {
            return Err("journal checksum mismatch".into());
        }
        if hash(payload) != checksum {
            break;
        }
        let record: JournalRecord = match serde_json::from_slice(payload) {
            Ok(record) => record,
            Err(_error) if end == bytes.len() => break,
            Err(error) => return Err(format!("journal JSON: {error}")),
        };
        record.validate()?;
        records.push(record);
        offset = end;
    }
    Ok((records, offset as u64))
}

pub fn read_meta(path: &Path) -> Result<JournalMeta, String> {
    if !path.exists() {
        return Ok(JournalMeta {
            format_version: JOURNAL_VERSION,
            last_epoch: 0,
            next_sequence: 1,
            acknowledged: vec![],
        });
    }
    let meta: JournalMeta =
        serde_json::from_slice(&fs::read(path).map_err(io)?).map_err(|error| error.to_string())?;
    (meta.format_version == JOURNAL_VERSION)
        .then_some(meta)
        .ok_or_else(|| "unsupported journal metadata version".into())
}

pub fn write_meta(path: &Path, meta: &JournalMeta) -> Result<(), String> {
    let temporary = path.with_extension("meta.tmp");
    let mut file = File::create(&temporary).map_err(io)?;
    file.write_all(&serde_json::to_vec(meta).map_err(|error| error.to_string())?)
        .map_err(io)?;
    file.sync_all().map_err(io)?;
    fs::rename(&temporary, path).map_err(io)?;
    sync_directory(path.parent().ok_or("journal metadata has no parent")?)
}

pub fn write_frames(path: &Path, records: &[JournalRecord]) -> Result<(), String> {
    let mut file = File::create(path).map_err(io)?;
    for record in records {
        let payload = serde_json::to_vec(record).map_err(|error| error.to_string())?;
        write_frame(&mut file, &payload)?;
    }
    file.sync_all().map_err(io)
}

pub fn write_frame(file: &mut File, payload: &[u8]) -> Result<(), String> {
    file.write_all(MAGIC)
        .and_then(|_| file.write_all(&[JOURNAL_VERSION]))
        .and_then(|_| file.write_all(&(payload.len() as u32).to_le_bytes()))
        .and_then(|_| file.write_all(&hash(payload).to_le_bytes()))
        .and_then(|_| file.write_all(payload))
        .map_err(io)
}
pub fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path).map_err(io)?.sync_all().map_err(io)
}
pub fn io(error: std::io::Error) -> String {
    error.to_string()
}
