//! Streaming WAV (IEEE float32) dumper for debugging the captured audio
//! pipeline.
//!
//! Opt-in via the `AUDIO_DUMP_FILE` env var (a path to write). Disabled by
//! default. `AUDIO_DUMP_MAX_SECS` caps the file size (default 30s) so a long
//! run cannot fill the disk.
//!
//! The dumped audio is *exactly* what `audio_capture` sends downstream (post
//! downmix/resample, e.g. 16 kHz mono f32) — so playing the file verifies the
//! full in-pipeline capture path, complementing the OS-level
//! `scripts/check-usb-microphone.py` check.
//!
//! No external crates: the WAV container (RIFF/WAVE/fmt /data) is written by
//! hand. The header is emitted up-front with the *cap* byte size so a
//! killed node still leaves a playable file; on graceful close the RIFF and
//! data chunk sizes are patched to the real byte count.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

const IEEE_FLOAT_FORMAT: u16 = 3;
const BITS_PER_F32_SAMPLE: u16 = 32;
const FMT_CHUNK_SIZE: u32 = 16;
const DEFAULT_MAX_SECS: u32 = 30;
const BYTES_PER_F32: u64 = 4;

/// Streaming WAV (IEEE float32) dumper. A disabled dumper (no env path) drops
/// all writes cheaply.
pub(crate) struct AudioDumper {
    writer: Option<std::io::BufWriter<File>>,
    sample_rate: u32,
    channels: u16,
    max_data_bytes: u64,
    written_data_bytes: u64,
    closed: bool,
}

impl AudioDumper {
    /// Reads `AUDIO_DUMP_FILE` (path) and `AUDIO_DUMP_MAX_SECS` (cap, default
    /// 30s). If the path is unset/empty, returns a disabled dumper.
    pub(crate) fn from_env(sample_rate: u32, channels: u16) -> Self {
        let path = match std::env::var("AUDIO_DUMP_FILE") {
            Ok(value) if !value.trim().is_empty() => value,
            _ => return Self::disabled(),
        };
        let max_secs = std::env::var("AUDIO_DUMP_MAX_SECS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&secs| secs > 0)
            .unwrap_or(DEFAULT_MAX_SECS);
        match Self::with_max_secs(&path, sample_rate, channels, max_secs) {
            Ok(dumper) => dumper,
            Err(error) => {
                tracing::warn!(
                    path = %path,
                    %error,
                    "AUDIO_DUMP_FILE set but could not open file; audio dump disabled"
                );
                Self::disabled()
            }
        }
    }

    /// Disabled no-op dumper.
    fn disabled() -> Self {
        Self {
            writer: None,
            sample_rate: 0,
            channels: 0,
            max_data_bytes: 0,
            written_data_bytes: 0,
            closed: true,
        }
    }

    /// Opens `path` and writes the WAV header. `max_secs` bounds the data
    /// payload so the file cannot exceed roughly
    /// `sample_rate * channels * 4 * max_secs` bytes.
    fn with_max_secs(
        path: &str,
        sample_rate: u32,
        channels: u16,
        max_secs: u32,
    ) -> std::io::Result<Self> {
        let max_samples = (sample_rate as u64) * (channels as u64) * (max_secs as u64);
        let max_data_bytes = max_samples
            .saturating_mul(BYTES_PER_F32)
            .min(u32::MAX as u64);

        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        let mut writer = std::io::BufWriter::new(File::create(path)?);
        write_wav_header(&mut writer, sample_rate, channels, max_data_bytes as u32)?;
        tracing::info!(
            path = %path,
            sample_rate,
            channels,
            max_secs,
            max_bytes = max_data_bytes,
            "Audio dump enabled (WAV float32); writing captured frames to file"
        );
        Ok(Self {
            writer: Some(writer),
            sample_rate,
            channels,
            max_data_bytes,
            written_data_bytes: 0,
            closed: false,
        })
    }

    /// Appends `samples` (f32, output-domain) to the WAV file, up to the cap.
    /// Once the cap is reached the file is finalized and further writes are
    /// dropped.
    pub(crate) fn write_chunk(&mut self, samples: &[f32]) {
        if self.closed || self.writer.is_none() || self.written_data_bytes >= self.max_data_bytes {
            return;
        }
        let remaining = self.max_data_bytes - self.written_data_bytes;
        let max_samples_now = (remaining / BYTES_PER_F32) as usize;
        let take = samples.len().min(max_samples_now);
        if take == 0 {
            return;
        }

        let mut bytes = Vec::with_capacity(take * BYTES_PER_F32 as usize);
        for &sample in &samples[..take] {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        let failed = match self.writer.as_mut() {
            Some(writer) => writer.write_all(&bytes).is_err(),
            None => true,
        };
        if failed {
            tracing::warn!("audio dump write failed; disabling dump");
            self.closed = true;
            self.writer = None;
            return;
        }

        self.written_data_bytes += bytes.len() as u64;
        if self.written_data_bytes >= self.max_data_bytes {
            tracing::info!(
                written_bytes = self.written_data_bytes,
                "Audio dump cap reached; finalizing file"
            );
            self.close();
        }
    }

    /// Finalizes the WAV: flushes and patches the RIFF/data sizes to the real
    /// byte count. Idempotent.
    pub(crate) fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        let Some(mut writer) = self.writer.take() else {
            return;
        };
        if let Err(error) = writer.flush() {
            tracing::warn!(%error, "audio dump flush failed; header sizes left at cap");
            return;
        }
        if let Err(error) = patch_wav_sizes(&mut writer, self.written_data_bytes as u32) {
            tracing::warn!(%error, "audio dump header patch failed; sizes left at cap");
            return;
        }
        tracing::info!(
            written_bytes = self.written_data_bytes,
            sample_rate = self.sample_rate,
            channels = self.channels,
            "Audio dump finalized"
        );
    }
}

impl Drop for AudioDumper {
    fn drop(&mut self) {
        self.close();
    }
}

fn write_wav_header<W: Write + Seek>(
    writer: &mut W,
    sample_rate: u32,
    channels: u16,
    data_size: u32,
) -> std::io::Result<()> {
    let byte_rate = sample_rate * channels as u32 * (BITS_PER_F32_SAMPLE as u32 / 8);
    let block_align = channels * (BITS_PER_F32_SAMPLE / 8);
    let riff_size = 36u32.saturating_add(data_size);
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&FMT_CHUNK_SIZE.to_le_bytes())?;
    writer.write_all(&IEEE_FLOAT_FORMAT.to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&BITS_PER_F32_SAMPLE.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;
    Ok(())
}

fn patch_wav_sizes<W: Seek + Write>(writer: &mut W, data_size: u32) -> std::io::Result<()> {
    let riff_size = 36u32.saturating_add(data_size);
    writer.seek(SeekFrom::Start(4))?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.seek(SeekFrom::Start(40))?;
    writer.write_all(&data_size.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn tmp_wav_path(label: &str) -> String {
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir()
            .join(format!(
                "audio_dump_{label}_{}_{}.wav",
                std::process::id(),
                n
            ))
            .to_string_lossy()
            .into_owned()
    }

    fn u32_le(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }
    fn u16_le(b: &[u8], off: usize) -> u16 {
        u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
    }

    #[test]
    fn wav_header_is_valid_ieee_float_and_sizes_patch_on_close() {
        let path = tmp_wav_path("hdr");
        {
            let mut dumper = AudioDumper::with_max_secs(&path, 16_000, 1, 30).unwrap();
            dumper.write_chunk(&[0.0, 0.5, -0.5, 1.0]);
            dumper.close();
        }
        let b = std::fs::read(&path).unwrap();
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(&b[8..12], b"WAVE");
        assert_eq!(&b[12..16], b"fmt ");
        assert_eq!(u32_le(&b, 16), 16); // fmt chunk size
        assert_eq!(u16_le(&b, 20), 3); // IEEE float
        assert_eq!(u16_le(&b, 22), 1); // channels
        assert_eq!(u32_le(&b, 24), 16_000); // sample rate
        assert_eq!(u32_le(&b, 28), 16_000 * 4); // byte rate = sr * ch(1) * 4
        assert_eq!(u16_le(&b, 32), 4); // block align
        assert_eq!(u16_le(&b, 34), 32); // bits per sample
        assert_eq!(&b[36..40], b"data");
        assert_eq!(u32_le(&b, 40), 16); // data size patched to actual (4 samples * 4 bytes)
        assert_eq!(u32_le(&b, 4), 36 + 16); // riff size patched
        assert_eq!(b.len(), 44 + 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_chunk_caps_at_max_secs() {
        // sample_rate=4, channels=1, max_secs=1 -> cap = 4 samples = 16 bytes
        let path = tmp_wav_path("cap");
        {
            let mut dumper = AudioDumper::with_max_secs(&path, 4, 1, 1).unwrap();
            dumper.write_chunk(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]); // 6 samples, only 4 fit
            dumper.write_chunk(&[0.6, 0.7]); // already capped -> dropped
            dumper.close();
        }
        let b = std::fs::read(&path).unwrap();
        assert_eq!(u32_le(&b, 40), 16); // only 4 samples of data
        assert_eq!(b.len(), 44 + 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn disabled_dumper_is_a_no_op() {
        let mut dumper = AudioDumper::disabled();
        dumper.write_chunk(&[1.0, 2.0]); // must not panic or write
        dumper.close();
        assert!(dumper.writer.is_none());
    }

    #[test]
    fn stereo_header_uses_correct_byte_rate_and_block_align() {
        let path = tmp_wav_path("stereo");
        {
            let mut dumper = AudioDumper::with_max_secs(&path, 48_000, 2, 1).unwrap();
            dumper.write_chunk(&[0.0, 0.0, 0.5, -0.5]); // 2 stereo frames
            dumper.close();
        }
        let b = std::fs::read(&path).unwrap();
        assert_eq!(u16_le(&b, 22), 2); // channels
        assert_eq!(u32_le(&b, 24), 48_000); // sample rate
        assert_eq!(u32_le(&b, 28), 48_000 * 2 * 4); // byte rate
        assert_eq!(u16_le(&b, 32), 2 * 4); // block align
        assert_eq!(u32_le(&b, 40), 16); // 4 samples * 4 bytes
        std::fs::remove_file(&path).ok();
    }
}
