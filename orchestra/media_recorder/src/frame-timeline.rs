use robo_rover_lib::{AudioFrameMetadata, PcmSampleFormat, VideoFrameMetadata};

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub metadata: VideoFrameMetadata,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub metadata: AudioFrameMetadata,
    pub payload: Vec<u8>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TimelineCounters {
    pub dropped_video: u64,
    pub audio_gaps: u64,
    pub timestamp_regressions: u64,
    pub silence_samples: u64,
}

#[derive(Debug, Default)]
pub struct FrameTimeline {
    origin_ms: Option<u64>,
    last_video_ms: Option<u64>,
    last_audio_ms: Option<u64>,
    audio_end_ms: u64,
    pub counters: TimelineCounters,
}

impl FrameTimeline {
    pub fn set_origin(&mut self, timestamp_ms: u64) {
        self.origin_ms.get_or_insert(timestamp_ms);
    }

    pub fn video_pts(&mut self, metadata: VideoFrameMetadata) -> Result<u64, String> {
        let origin = *self.origin_ms.get_or_insert(metadata.capture_timestamp_ms);
        if self
            .last_video_ms
            .is_some_and(|last| metadata.capture_timestamp_ms < last)
        {
            self.counters.timestamp_regressions += 1;
            return Err("video capture timestamp regressed".into());
        }
        self.last_video_ms = Some(metadata.capture_timestamp_ms);
        Ok(metadata.capture_timestamp_ms.saturating_sub(origin))
    }

    pub fn audio_prefix_silence(&mut self, metadata: AudioFrameMetadata) -> Result<u64, String> {
        if metadata.format != PcmSampleFormat::S16Le {
            return Err("recorder requires s16le audio".into());
        }
        let expected = metadata.expected_payload_len()?;
        if expected == 0 {
            return Err("audio frame has no samples".into());
        }
        let origin = *self.origin_ms.get_or_insert(metadata.capture_timestamp_ms);
        if self
            .last_audio_ms
            .is_some_and(|last| metadata.capture_timestamp_ms < last)
        {
            self.counters.timestamp_regressions += 1;
            return Err("audio capture timestamp regressed".into());
        }
        let timestamp_ms = metadata.capture_timestamp_ms.saturating_sub(origin);
        let gap_ms = timestamp_ms.saturating_sub(self.audio_end_ms);
        let silence_samples = gap_ms
            .saturating_mul(u64::from(metadata.sample_rate))
            .saturating_mul(u64::from(metadata.channels))
            / 1000;
        self.counters.audio_gaps += u64::from(gap_ms > 2);
        self.counters.silence_samples = self
            .counters
            .silence_samples
            .saturating_add(silence_samples);
        let frame_ms = u64::from(metadata.sample_count) * 1000 / u64::from(metadata.sample_rate);
        self.audio_end_ms = timestamp_ms.saturating_add(frame_ms);
        self.last_audio_ms = Some(metadata.capture_timestamp_ms);
        Ok(silence_samples.saturating_mul(2))
    }

    pub fn finish_silence(&mut self, end_pts_ms: u64, sample_rate: u32, channels: u16) -> u64 {
        let gap_ms = end_pts_ms.saturating_sub(self.audio_end_ms);
        let samples = gap_ms
            .saturating_mul(u64::from(sample_rate))
            .saturating_mul(u64::from(channels))
            / 1000;
        self.counters.silence_samples = self.counters.silence_samples.saturating_add(samples);
        samples.saturating_mul(2)
    }

    pub fn duration_ms(&self) -> u64 {
        self.last_video_ms
            .zip(self.origin_ms)
            .map(|(last, origin)| last.saturating_sub(origin))
            .unwrap_or(self.audio_end_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn audio(timestamp: u64, samples: u32) -> AudioFrameMetadata {
        AudioFrameMetadata {
            stream_id: Uuid::new_v4(),
            frame_id: 1,
            capture_timestamp_ms: timestamp,
            sample_rate: 8000,
            channels: 1,
            sample_count: samples,
            format: PcmSampleFormat::S16Le,
        }
    }

    #[test]
    fn inserts_silence_for_audio_gap_and_rejects_regression() {
        let mut timeline = FrameTimeline::default();
        assert_eq!(timeline.audio_prefix_silence(audio(1000, 800)).unwrap(), 0);
        assert_eq!(
            timeline.audio_prefix_silence(audio(1200, 800)).unwrap(),
            1600
        );
        assert!(timeline.audio_prefix_silence(audio(1100, 800)).is_err());
        assert_eq!(timeline.counters.audio_gaps, 1);
        assert_eq!(timeline.counters.timestamp_regressions, 1);
    }
}
