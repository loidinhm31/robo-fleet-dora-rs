use std::time::{SystemTime, UNIX_EPOCH};

use dora_node_api::{MetadataParameters, Parameter};
use eyre::Result;
use uuid::Uuid;

const WALKIE_SAMPLE_RATE: u32 = 16_000;
const MAX_WALKIE_SAMPLES: usize = 16_384;

pub struct WalkieMetadataSequence {
    stream_id: Uuid,
    next_frame_id: u64,
}

impl WalkieMetadataSequence {
    pub fn new() -> Self {
        Self {
            stream_id: Uuid::new_v4(),
            next_frame_id: 0,
        }
    }

    pub fn next(&mut self, sample_count: usize) -> Result<MetadataParameters> {
        if sample_count == 0 || sample_count > MAX_WALKIE_SAMPLES {
            return Err(eyre::eyre!("walkie frame sample count is out of bounds"));
        }
        let sample_count = u32::try_from(sample_count)?;
        let frame_id = self.next_frame_id;
        self.next_frame_id = self.next_frame_id.saturating_add(1);
        Ok(MetadataParameters::from([
            ("source_kind".into(), Parameter::String("walkie".into())),
            (
                "stream_id".into(),
                Parameter::String(self.stream_id.to_string()),
            ),
            ("frame_id".into(), Parameter::Integer(frame_id.try_into()?)),
            (
                "capture_timestamp_ms".into(),
                Parameter::Integer(current_time_ms()?.try_into()?),
            ),
            (
                "sample_rate".into(),
                Parameter::Integer(i64::from(WALKIE_SAMPLE_RATE)),
            ),
            ("channels".into(), Parameter::Integer(1)),
            (
                "sample_count".into(),
                Parameter::Integer(i64::from(sample_count)),
            ),
            ("format".into(), Parameter::String("f32le".into())),
            ("priority".into(), Parameter::String("high".into())),
        ]))
    }
}

fn current_time_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_stable_stream_and_monotonic_frame_metadata() {
        let mut sequence = WalkieMetadataSequence::new();
        let first = sequence.next(320).unwrap();
        let second = sequence.next(320).unwrap();

        assert_eq!(first["stream_id"], second["stream_id"]);
        assert_eq!(first["frame_id"], Parameter::Integer(0));
        assert_eq!(second["frame_id"], Parameter::Integer(1));
        assert_eq!(first["sample_rate"], Parameter::Integer(16_000));
        assert!(sequence.next(0).is_err());
    }
}
