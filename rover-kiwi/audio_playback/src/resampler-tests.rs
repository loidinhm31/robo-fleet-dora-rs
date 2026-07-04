use super::*;
use std::f32::consts::TAU;

fn tone(rate: u32, seconds: f32, frequency: f32) -> Vec<f32> {
    (0..(rate as f32 * seconds) as usize)
        .map(|index| (TAU * frequency * index as f32 / rate as f32).sin())
        .collect()
}

fn resample_all(input_rate: u32, output_rate: u32) -> Vec<f32> {
    let input = tone(input_rate, 1.0, 440.0);
    let mut resampler = SourceResampler::new(input_rate, output_rate).unwrap();
    let mut output = Vec::new();
    for chunk in input.chunks(882) {
        output.extend(resampler.process(chunk).unwrap());
    }
    output.extend(resampler.flush().unwrap());
    output
}

#[test]
fn preserves_duration_for_tts_and_walkie_rates() {
    for input_rate in [16_000, 44_100] {
        let output = resample_all(input_rate, 48_000);
        assert!(output.len().abs_diff(48_000) <= 2, "{}", output.len());
        let positive_crossings = output
            .windows(2)
            .filter(|pair| pair[0] <= 0.0 && pair[1] > 0.0)
            .count();
        let frequency = positive_crossings as f32 * 48_000.0 / output.len() as f32;
        assert!((frequency - 440.0).abs() < 2.0, "{frequency}");
    }
}

#[test]
fn passthrough_preserves_samples_and_resets() {
    let mut resampler = SourceResampler::new(48_000, 48_000).unwrap();
    assert_eq!(resampler.process(&[0.1, -0.2]).unwrap(), vec![0.1, -0.2]);
    assert!(resampler.flush().unwrap().is_empty());
    assert_eq!(resampler.process(&[0.3]).unwrap(), vec![0.3]);
    assert!(resampler.flush().unwrap().is_empty());
}

#[test]
fn duration_is_independent_of_input_chunk_partition() {
    let input = tone(44_100, 0.25, 440.0);
    for chunk_size in [137, 882, 4_096] {
        let mut resampler = SourceResampler::new(44_100, 48_000).unwrap();
        let mut output = Vec::new();
        for chunk in input.chunks(chunk_size) {
            output.extend(resampler.process(chunk).unwrap());
        }
        output.extend(resampler.flush().unwrap());
        assert_eq!(output.len(), 12_000, "chunk size {chunk_size}");
        assert!(resampler.flush().unwrap().is_empty());
    }
}

#[test]
fn short_tail_uses_exact_rational_ceiling() {
    let mut resampler = SourceResampler::new(44_100, 48_000).unwrap();
    let mut output = resampler.process(&[0.5; 7]).unwrap();
    output.extend(resampler.flush().unwrap());
    assert_eq!(output.len(), 8);
}
