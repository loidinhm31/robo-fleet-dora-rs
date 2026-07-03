use eyre::{bail, eyre, Result, WrapErr};
use robo_rover_lib::{AudioFrameMetadata, PcmFramePacket, PcmSampleFormat};
use std::env;
use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const SAMPLE_RATE: u32 = 16_000;
const CHANNELS: u16 = 1;
const FRAME_SAMPLES: usize = 800;
const TRAILING_SILENCE_FRAMES: usize = 40;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let entity_id = args.next().ok_or_else(|| eyre!("missing entity ID"))?;
    let wav_path = args.next().ok_or_else(|| eyre!("missing WAV path"))?;
    let repetitions = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .wrap_err("repetitions must be a positive integer")?
        .unwrap_or(1);
    if repetitions == 0 {
        bail!("repetitions must be positive");
    }

    let pcm = read_pcm16_wav(&wav_path)?;
    let mut config = zenoh::Config::default();
    config
        .insert_json5("connect/endpoints", r#"["tcp/127.0.0.1:7447"]"#)
        .map_err(|error| eyre!(error.to_string()))?;
    let session = zenoh::open(config)
        .await
        .map_err(|error| eyre!(error.to_string()))?;
    let topic = format!("rover/{entity_id}/audio/raw");
    let publisher = session
        .declare_publisher(&topic)
        .await
        .map_err(|error| eyre!(error.to_string()))?;
    tokio::time::sleep(Duration::from_secs(1)).await;

    let stream_id = Uuid::new_v4();
    let mut next_frame_id = 0u64;
    for repetition in 0..repetitions {
        publish_once(&publisher, &pcm, stream_id, &mut next_frame_id).await?;
        println!(
            "entity={entity_id} repetition={}/{}",
            repetition + 1,
            repetitions
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Ok(())
}

async fn publish_once(
    publisher: &zenoh::pubsub::Publisher<'_>,
    pcm: &[u8],
    stream_id: Uuid,
    next_frame_id: &mut u64,
) -> Result<()> {
    let silence = vec![0u8; FRAME_SAMPLES * 2];
    let frames = pcm
        .chunks(FRAME_SAMPLES * 2)
        .chain(std::iter::repeat(silence.as_slice()).take(TRAILING_SILENCE_FRAMES));
    for payload in frames {
        let metadata = AudioFrameMetadata {
            stream_id,
            frame_id: *next_frame_id,
            capture_timestamp_ms: now_ms()?,
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
            sample_count: (payload.len() / 2) as u32,
            format: PcmSampleFormat::S16Le,
        };
        let packet = PcmFramePacket { metadata, payload }
            .encode()
            .map_err(|error| eyre!(error))?;
        publisher
            .put(packet)
            .await
            .map_err(|error| eyre!(error.to_string()))?;
        *next_frame_id = next_frame_id.saturating_add(1);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Ok(())
}

fn read_pcm16_wav(path: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path).wrap_err_with(|| format!("failed to read {path}"))?;
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        bail!("fixture must be a RIFF/WAVE file");
    }

    let mut offset = 12usize;
    let mut valid_format = false;
    let mut pcm = None;
    while offset + 8 <= bytes.len() {
        let chunk = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        let start = offset + 8;
        let end = start
            .checked_add(size)
            .ok_or_else(|| eyre!("WAV chunk overflow"))?;
        if end > bytes.len() {
            bail!("truncated WAV chunk");
        }
        if chunk == b"fmt " && size >= 16 {
            let format = u16::from_le_bytes(bytes[start..start + 2].try_into()?);
            let channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into()?);
            let sample_rate = u32::from_le_bytes(bytes[start + 4..start + 8].try_into()?);
            let bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into()?);
            valid_format =
                format == 1 && channels == CHANNELS && sample_rate == SAMPLE_RATE && bits == 16;
        } else if chunk == b"data" {
            pcm = Some(bytes[start..end].to_vec());
        }
        offset = end + (size % 2);
    }

    if !valid_format {
        bail!("fixture must be mono 16 kHz PCM16");
    }
    pcm.filter(|payload| !payload.is_empty())
        .ok_or_else(|| eyre!("fixture has no PCM data"))
}

fn now_ms() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64)
}
