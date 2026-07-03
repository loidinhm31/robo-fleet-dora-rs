use central_speech_recognizer::config::SttConfig;
use central_speech_recognizer::native::{load_models, NativeModels};
use robo_rover_lib::SttProfile;
use sherpa_onnx::Wave;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[test]
#[ignore = "requires downloaded Sherpa ASR models"]
fn selected_profile_loads_native_models() {
    let config = SttConfig::from_env().expect("profile configuration should validate");
    let _models = load_models(&config).expect("native Sherpa models should load");
}

#[test]
#[ignore = "requires downloaded Sherpa ASR models"]
fn selected_profile_decodes_bundled_fixture_through_vad() {
    let config = SttConfig::from_env().expect("profile configuration should validate");
    let models = load_models(&config).expect("native Sherpa models should load");
    let fixture = fixture_path(&config);
    let wave = read_wave(&fixture);
    assert_eq!(wave.sample_rate(), 16_000, "fixture must match VAD rate");

    let started = Instant::now();
    let transcript = decode_through_vad(&models, wave.samples());
    let decode_seconds = started.elapsed().as_secs_f64();
    let audio_seconds = wave.samples().len() as f64 / wave.sample_rate() as f64;
    let rtf = decode_seconds / audio_seconds;

    assert!(
        !transcript.is_empty(),
        "recognizer should decode VAD speech"
    );
    if config.models.profile == SttProfile::EnVadOffline {
        assert_eq!(
            normalize(&transcript),
            "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS"
        );
    }
    assert!(rtf < 1.0, "decode RTF {rtf:.3} must remain below 1.0");
    println!(
        "profile={:?} fixture={} audio_seconds={audio_seconds:.3} decode_seconds={decode_seconds:.3} rtf={rtf:.3} transcript={transcript:?}",
        config.models.profile,
        fixture.display()
    );
}

#[test]
#[ignore = "requires downloaded Sherpa ASR models"]
fn selected_profile_f32_and_s16le_transcripts_match() {
    let config = SttConfig::from_env().expect("profile configuration should validate");
    let fixture = fixture_path(&config);
    let wave = read_wave(&fixture);
    let transported = wave
        .samples()
        .iter()
        .map(|sample| {
            let encoded = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
            f32::from(encoded) / 32_768.0
        })
        .collect::<Vec<_>>();

    let f32_text = decode_through_vad(
        &load_models(&config).expect("F32 model load should succeed"),
        wave.samples(),
    );
    let s16le_text = decode_through_vad(
        &load_models(&config).expect("S16LE model load should succeed"),
        &transported,
    );

    assert_eq!(normalize(&f32_text), normalize(&s16le_text));
    println!(
        "profile={:?} f32_transcript={f32_text:?} s16le_transcript={s16le_text:?}",
        config.models.profile
    );
}

fn decode_through_vad(models: &NativeModels, samples: &[f32]) -> String {
    models.vad.reset();
    for samples in samples.chunks(512) {
        models.vad.accept_waveform(samples);
    }
    models.vad.flush();

    let mut transcripts = Vec::new();
    while let Some(segment) = models.vad.front() {
        let stream = models.recognizer.create_stream();
        stream.accept_waveform(16_000, segment.samples());
        models.recognizer.decode(&stream);
        if let Some(result) = stream.get_result() {
            let text = result.text.trim();
            if !text.is_empty() {
                transcripts.push(text.to_owned());
            }
        }
        models.vad.pop();
    }
    assert!(!transcripts.is_empty(), "VAD should detect bundled speech");
    transcripts.join(" ")
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn fixture_path(config: &SttConfig) -> PathBuf {
    let model_dir = config
        .models
        .encoder
        .parent()
        .expect("encoder path should have a model directory");
    match config.models.profile {
        SttProfile::EnVadOffline => parent(model_dir).join("test_wavs/1089-134686-0001.wav"),
        SttProfile::ViVadOffline => model_dir.join("test_wavs/0.wav"),
    }
}

fn read_wave(path: &Path) -> Wave {
    Wave::read(path.to_str().expect("fixture path should be valid UTF-8"))
        .expect("bundled fixture should be a readable WAV")
}

fn parent(path: &Path) -> &Path {
    path.parent().expect("English model directory should exist")
}
