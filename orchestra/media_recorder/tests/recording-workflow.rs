#![cfg(unix)]

use media_recorder::clip_catalog::{ClipCatalog, RecordingManifest};
use media_recorder::config::RecorderConfig;
use media_recorder::ffmpeg_session::{ffprobe_json, validate_mp4, FfmpegSession, FfmpegSpec};
use media_recorder::frame_timeline::{AudioFrame, VideoFrame};
use media_recorder::path_resolver::PathResolver;
use media_recorder::session_manager::{SessionManager, StartRequest};
use robo_rover_lib::{AudioFrameMetadata, PcmSampleFormat, VideoFrameMetadata};
use robo_rover_lib::{RecordingAudioCodec, RecordingClip, RecordingVideoCodec};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;
use uuid::Uuid;

fn jpeg_fixture_at(size: &str) -> Vec<u8> {
    let child = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=s={size}"),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "3",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg fixture");
    let output = wait_for_fixture(child);
    assert!(output.status.success(), "fixture ffmpeg failed");
    output.stdout
}

fn wait_for_fixture(mut child: std::process::Child) -> Output {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().expect("inspect ffmpeg fixture").is_some() {
            return child.wait_with_output().expect("collect ffmpeg fixture");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().expect("collect timed-out fixture");
            panic!(
                "ffmpeg fixture exceeded five seconds: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn jpeg_fixture() -> Vec<u8> {
    jpeg_fixture_at("32x24")
}

fn audio_fixture(frame_id: u64, capture_timestamp_ms: u64) -> AudioFrame {
    let sample_count = 528;
    AudioFrame {
        metadata: AudioFrameMetadata {
            stream_id: Uuid::nil(),
            frame_id,
            capture_timestamp_ms,
            sample_rate: 16_000,
            channels: 1,
            sample_count,
            format: PcmSampleFormat::S16Le,
        },
        payload: vec![0; sample_count as usize * 2],
    }
}

fn mp4_duration_ms(path: &std::path::Path) -> u64 {
    let duration = ffprobe_json(std::path::Path::new("/usr/bin/ffprobe"), path)
        .unwrap()
        .pointer("/format/duration")
        .and_then(|value| value.as_str())
        .and_then(|value| value.parse::<f64>().ok())
        .expect("ffprobe duration");
    (duration * 1_000.0).round() as u64
}

#[test]
fn synthetic_jpeg_pcm_is_atomically_published_probeable_and_deletable() {
    let root = tempdir().unwrap();
    let resolver = PathResolver::new(root.path()).unwrap();
    let directory = resolver.directory("rover-a/session").unwrap();
    let recording_id = Uuid::new_v4().to_string();
    let partial = resolver
        .partial()
        .join(format!("{recording_id}.mp4.partial"));
    let spec = FfmpegSpec {
        executable: PathBuf::from("/usr/bin/ffmpeg"),
        width: 32,
        height: 24,
        fps: 30,
        sample_rate: 16_000,
        channels: 1,
        output: partial.clone(),
    };
    let mut session = FfmpegSession::spawn(&spec).unwrap();
    let jpeg = jpeg_fixture();
    session.write_video(&jpeg).unwrap();
    session.write_audio(&vec![0u8; 32_000]).unwrap();
    let outcome = session.wait(std::time::Duration::from_secs(10)).unwrap();
    assert!(outcome.success, "ffmpeg failed: {}", outcome.stderr);
    validate_mp4(PathBuf::from("/usr/bin/ffprobe").as_path(), &partial).unwrap();

    let bytes = std::fs::metadata(&partial).unwrap().len();
    let manifest = RecordingManifest {
        clip: RecordingClip {
            recording_id: recording_id.clone(),
            entity_id: "rover-a".into(),
            relative_path: format!("rover-a/session/{recording_id}.mp4"),
            started_at_ms: 100,
            duration_ms: 1000,
            bytes_written: bytes,
            video_codec: RecordingVideoCodec::H264,
            audio_codec: RecordingAudioCodec::Aac,
        },
        dropped_video: 0,
        audio_gaps: 1,
        silence_samples: 16_000,
        timestamp_regressions: 0,
    };
    let catalog = ClipCatalog::new(resolver.clone(), PathBuf::from("/usr/bin/ffprobe"));
    catalog.publish(&partial, &directory, &manifest).unwrap();
    assert!(!partial.exists());
    assert!(directory.join(format!("{recording_id}.mp4")).is_file());
    assert!(directory
        .join(format!("{recording_id}.manifest.json"))
        .is_file());
    let clips = catalog
        .list(Some("rover-a"), Some("rover-a/session"))
        .unwrap();
    assert_eq!(clips.len(), 1);
    assert!(!root
        .path()
        .join("rover-a")
        .to_string_lossy()
        .contains(".jpg"));

    catalog.delete(&recording_id).unwrap();
    assert!(!directory.join(format!("{recording_id}.mp4")).exists());
    assert!(!directory
        .join(format!("{recording_id}.manifest.json"))
        .exists());
    assert!(catalog.lookup(&recording_id).is_err());
    assert!(catalog.delete(&recording_id).is_err());
}

#[test]
fn audio_backpressure_does_not_starve_the_video_pipe() {
    let root = tempdir().unwrap();
    let output = root.path().join("dual-pipe.mp4.partial");
    let spec = FfmpegSpec {
        executable: PathBuf::from("/usr/bin/ffmpeg"),
        width: 640,
        height: 480,
        fps: 30,
        sample_rate: 16_000,
        channels: 1,
        output: output.clone(),
    };
    let mut session = FfmpegSession::spawn(&spec).unwrap();

    // FFmpeg cannot complete stream probing from PCM alone. The video write
    // must still be scheduled while the audio pipe is under backpressure.
    // The recorder can accumulate a bounded burst of PCM before the first
    // video frame arrives. Each handoff must accept that burst without
    // preventing the video pump from scheduling FFmpeg's probe frame.
    for _ in 0..8 {
        session.write_audio(&vec![0; 32_000]).unwrap();
    }
    let jpeg = jpeg_fixture_at("640x480");
    for _ in 0..120 {
        session.write_video(&jpeg).unwrap();
    }

    let outcome = session.wait(std::time::Duration::from_secs(10)).unwrap();
    assert!(outcome.success, "ffmpeg failed: {}", outcome.stderr);
    validate_mp4(PathBuf::from("/usr/bin/ffprobe").as_path(), &output).unwrap();
}

#[test]
fn failed_owned_process_is_reported_without_shell_interpolation() {
    let root = tempdir().unwrap();
    let output = root.path().join("failed.mp4.partial");
    let spec = FfmpegSpec {
        executable: PathBuf::from("/bin/false"),
        width: 32,
        height: 24,
        fps: 30,
        sample_rate: 16_000,
        channels: 1,
        output,
    };
    let session = FfmpegSession::spawn(&spec).unwrap();
    let outcome = session.wait(std::time::Duration::from_secs(2)).unwrap();
    assert!(!outcome.success);
}

#[test]
fn owned_process_timeout_is_killed_and_reaped() {
    let root = tempdir().unwrap();
    let executable = root.path().join("hang-fixture");
    std::fs::write(&executable, "#!/bin/sh\nwhile :; do :; done\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let spec = FfmpegSpec {
        executable,
        width: 32,
        height: 24,
        fps: 30,
        sample_rate: 16_000,
        channels: 1,
        output: root.path().join("hung.mp4.partial"),
    };
    let session = FfmpegSession::spawn(&spec).unwrap();
    let outcome = session.wait(std::time::Duration::from_millis(100)).unwrap();
    assert!(!outcome.success);
    assert!(outcome.stderr.contains("timeout"));
}

#[test]
fn blocked_input_pumps_respect_the_session_timeout() {
    let root = tempdir().unwrap();
    let executable = root.path().join("hang-fixture");
    std::fs::write(&executable, "#!/bin/sh\nwhile :; do :; done\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let spec = FfmpegSpec {
        executable,
        width: 32,
        height: 24,
        fps: 30,
        sample_rate: 16_000,
        channels: 1,
        output: root.path().join("hung-input.mp4.partial"),
    };
    let mut session = FfmpegSession::spawn(&spec).unwrap();
    session.write_audio(&vec![0; 128_000]).unwrap();
    let outcome = session.wait(std::time::Duration::from_millis(100)).unwrap();
    assert!(!outcome.success);
    assert!(outcome.stderr.contains("timeout"));
}

#[test]
fn saturated_video_handoff_does_not_block_recorder_admission() {
    let root = tempdir().unwrap();
    let executable = root.path().join("hang-fixture");
    std::fs::write(&executable, "#!/bin/sh\nwhile :; do :; done\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let spec = FfmpegSpec {
        executable,
        width: 640,
        height: 480,
        fps: 30,
        sample_rate: 16_000,
        channels: 1,
        output: root.path().join("saturated-video.mp4.partial"),
    };
    let mut session = FfmpegSession::spawn(&spec).unwrap();
    let jpeg = jpeg_fixture_at("640x480");
    let stop = Arc::new(AtomicBool::new(false));
    let cancel = Arc::clone(&stop);
    let timer = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        cancel.store(true, Ordering::Release);
    });

    for _ in 0..32 {
        session.write_video_until(&jpeg, Some(&stop)).unwrap();
    }
    timer.join().unwrap();
    let outcome = session.wait(std::time::Duration::from_millis(100)).unwrap();
    assert!(!outcome.success);
}

#[test]
fn saturated_video_handoff_bounds_coalesced_finalization_work() {
    let root = tempdir().unwrap();
    let executable = root.path().join("hang-fixture");
    std::fs::write(&executable, "#!/bin/sh\nwhile :; do :; done\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let spec = FfmpegSpec {
        executable,
        width: 640,
        height: 480,
        fps: 30,
        sample_rate: 16_000,
        channels: 1,
        output: root.path().join("bounded-video.mp4.partial"),
    };
    let mut session = FfmpegSession::spawn(&spec).unwrap();
    let jpeg = jpeg_fixture_at("640x480");

    let error = (0..600)
        .find_map(|_| session.write_video(&jpeg).err())
        .expect("bounded handoff must reject an excessive logical backlog");
    assert!(error.contains("backlog limit"), "unexpected error: {error}");
    let outcome = session.wait(std::time::Duration::from_millis(100)).unwrap();
    assert!(!outcome.success);
}

#[test]
fn sessions_are_independent_and_duplicate_entity_start_is_rejected() {
    let root = tempdir().unwrap();
    let mut manager = SessionManager::new(RecorderConfig {
        recording_root: root.path().to_path_buf(),
        ffmpeg_path: PathBuf::from("/usr/bin/ffmpeg"),
        ffprobe_path: PathBuf::from("/usr/bin/ffprobe"),
        max_concurrent: 2,
        max_duration_ms: 10_000,
        max_output_bytes: 32 * 1024 * 1024,
        startup_timeout_ms: 2_000,
        finalization_timeout_ms: 5_000,
        min_free_bytes: 0,
        queue_capacity: 4,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        video_fps: 30,
    })
    .unwrap();
    let first = manager
        .start(StartRequest {
            request_id: Uuid::new_v4().to_string(),
            entity_id: "rover-a".into(),
            relative_directory: "captures".into(),
        })
        .unwrap();
    assert!(manager
        .start(StartRequest {
            request_id: Uuid::new_v4().to_string(),
            entity_id: "rover-a".into(),
            relative_directory: "captures".into()
        })
        .is_err());
    let second = manager
        .start(StartRequest {
            request_id: Uuid::new_v4().to_string(),
            entity_id: "rover-b".into(),
            relative_directory: "captures".into(),
        })
        .unwrap();
    let jpeg = jpeg_fixture();
    for entity in ["rover-a", "rover-b"] {
        assert!(manager.push_audio(
            entity,
            AudioFrame {
                metadata: AudioFrameMetadata {
                    stream_id: Uuid::new_v4(),
                    frame_id: 1,
                    capture_timestamp_ms: 1_000,
                    sample_rate: 16_000,
                    channels: 1,
                    sample_count: 16_000,
                    format: PcmSampleFormat::S16Le
                },
                payload: vec![0; 32_000],
            }
        ));
        assert!(manager.push_video(
            entity,
            VideoFrame {
                metadata: VideoFrameMetadata {
                    frame_id: 1,
                    capture_timestamp_ms: 1_000,
                    width: 32,
                    height: 24
                },
                payload: jpeg.clone(),
            }
        ));
    }
    manager.stop(&first.recording_id).unwrap();
    manager.stop(&second.recording_id).unwrap();
    // The media bridge continues producing frames after a stop command. Those
    // frames must not keep a stopped session's queue non-empty indefinitely.
    for frame_id in 2..32 {
        assert!(!manager.push_video(
            "rover-a",
            VideoFrame {
                metadata: VideoFrameMetadata {
                    frame_id,
                    capture_timestamp_ms: 1_000 + frame_id,
                    width: 32,
                    height: 24,
                },
                payload: jpeg.clone(),
            },
        ));
    }
    let mut completed = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while completed.len() < 2 && Instant::now() < deadline {
        completed.extend(manager.reap());
        if completed.len() < 2 {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    assert_eq!(completed.len(), 2);
    assert!(
        completed
            .iter()
            .all(|status| status.state == robo_rover_lib::RecordingSessionState::Completed),
        "statuses: {completed:?}"
    );
    let (clips, issues) = manager.catalog().scan();
    assert_eq!(clips.len(), 2);
    assert!(issues.is_empty(), "unexpected catalog issues: {issues:?}");
}

#[test]
fn live_sized_video_without_microphone_finalizes_after_stop() {
    let root = tempdir().unwrap();
    let mut manager = SessionManager::new(RecorderConfig {
        recording_root: root.path().to_path_buf(),
        ffmpeg_path: PathBuf::from("/usr/bin/ffmpeg"),
        ffprobe_path: PathBuf::from("/usr/bin/ffprobe"),
        max_concurrent: 1,
        max_duration_ms: 10_000,
        max_output_bytes: 32 * 1024 * 1024,
        startup_timeout_ms: 2_000,
        finalization_timeout_ms: 5_000,
        min_free_bytes: 0,
        queue_capacity: 8,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        video_fps: 30,
    })
    .unwrap();
    let session = manager
        .start(StartRequest {
            request_id: Uuid::new_v4().to_string(),
            entity_id: "rover-a".into(),
            relative_directory: "captures".into(),
        })
        .unwrap();
    let jpeg = jpeg_fixture_at("640x480");
    assert!(jpeg.len() > 8 * 1024, "fixture must exceed a pipe buffer");
    for frame_id in 1..=4 {
        assert!(manager.push_video(
            "rover-a",
            VideoFrame {
                metadata: VideoFrameMetadata {
                    frame_id,
                    capture_timestamp_ms: 1_000 + u64::from(frame_id - 1) * 33,
                    width: 640,
                    height: 480,
                },
                payload: jpeg.clone(),
            },
        ));
    }
    manager.stop(&session.recording_id).unwrap();
    let mut completed = Vec::new();
    let completion_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < completion_deadline {
        completed.extend(manager.reap());
        if !completed.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(completed.len(), 1, "statuses: {completed:?}");
    assert_eq!(
        completed[0].state,
        robo_rover_lib::RecordingSessionState::Completed,
        "statuses: {completed:?}"
    );
}

#[test]
fn sustained_audio_video_recording_preserves_video_duration() {
    let root = tempdir().unwrap();
    let mut manager = SessionManager::new(RecorderConfig {
        recording_root: root.path().to_path_buf(),
        ffmpeg_path: PathBuf::from("/usr/bin/ffmpeg"),
        ffprobe_path: PathBuf::from("/usr/bin/ffprobe"),
        max_concurrent: 1,
        max_duration_ms: 10_000,
        max_output_bytes: 32 * 1024 * 1024,
        startup_timeout_ms: 2_000,
        finalization_timeout_ms: 5_000,
        min_free_bytes: 0,
        queue_capacity: 4,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        video_fps: 30,
    })
    .unwrap();
    let session = manager
        .start(StartRequest {
            request_id: Uuid::new_v4().to_string(),
            entity_id: "rover-a".into(),
            relative_directory: "captures".into(),
        })
        .unwrap();
    let jpeg = jpeg_fixture();

    for frame_id in 0..4 {
        assert!(manager.push_audio("rover-a", audio_fixture(frame_id, 1_000 + frame_id * 33),));
    }
    let mut video_replacements = 0;
    for frame_id in 0..=60 {
        let timestamp = 1_000 + frame_id * 33;
        let _ = manager.push_audio("rover-a", audio_fixture(frame_id + 4, timestamp));
        if !manager.push_video(
            "rover-a",
            VideoFrame {
                metadata: VideoFrameMetadata {
                    frame_id,
                    capture_timestamp_ms: timestamp,
                    width: 32,
                    height: 24,
                },
                payload: jpeg.clone(),
            },
        ) {
            video_replacements += 1;
        }
    }

    manager.stop(&session.recording_id).unwrap();
    let mut completed = Vec::new();
    let completion_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < completion_deadline {
        completed.extend(manager.reap());
        if !completed.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(completed.len(), 1, "statuses: {completed:?}");
    assert_eq!(
        completed[0].state,
        robo_rover_lib::RecordingSessionState::Completed,
        "statuses: {completed:?}"
    );

    let clips = manager
        .catalog()
        .list(Some("rover-a"), Some("captures"))
        .unwrap();
    assert_eq!(clips.len(), 1);
    let manifest_path = root
        .path()
        .join("captures")
        .join(format!("{}.manifest.json", clips[0].recording_id));
    let manifest: RecordingManifest =
        serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
    assert_eq!(manifest.dropped_video, video_replacements);
    assert_eq!(manifest.clip.duration_ms, clips[0].duration_ms);
    assert!(
        clips[0].duration_ms >= 1_900,
        "video timeline was truncated: {:?}",
        clips[0]
    );
}

#[test]
fn paced_live_sized_audio_video_preserves_capture_duration_on_stop() {
    let root = tempdir().unwrap();
    let mut manager = SessionManager::new(RecorderConfig {
        recording_root: root.path().to_path_buf(),
        ffmpeg_path: PathBuf::from("/usr/bin/ffmpeg"),
        ffprobe_path: PathBuf::from("/usr/bin/ffprobe"),
        max_concurrent: 1,
        max_duration_ms: 10_000,
        max_output_bytes: 32 * 1024 * 1024,
        startup_timeout_ms: 2_000,
        finalization_timeout_ms: 5_000,
        min_free_bytes: 0,
        queue_capacity: 8,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        video_fps: 30,
    })
    .unwrap();
    let session = manager
        .start(StartRequest {
            request_id: Uuid::new_v4().to_string(),
            entity_id: "rover-a".into(),
            relative_directory: "captures".into(),
        })
        .unwrap();
    let jpeg = jpeg_fixture_at("640x480");
    // Dora can deliver a bounded PCM burst before the first JPEG reaches the
    // worker. Keep that ordering in the sustained test because it was the
    // live encoder-path trigger.
    for frame_id in 0..4 {
        assert!(manager.push_audio(
            "rover-a",
            AudioFrame {
                metadata: AudioFrameMetadata {
                    stream_id: Uuid::nil(),
                    frame_id,
                    capture_timestamp_ms: 1_000 + frame_id * 50,
                    sample_rate: 16_000,
                    channels: 1,
                    sample_count: 800,
                    format: PcmSampleFormat::S16Le,
                },
                payload: vec![0; 1_600],
            },
        ));
    }
    let capture_start = std::time::Instant::now();
    let mut video_frame_id = 0u64;
    let mut audio_frame_id = 4u64;
    let mut next_video_ms = 0u64;
    let mut next_audio_ms = 200u64;
    let mut video_admission_drops = 0;

    while capture_start.elapsed() < std::time::Duration::from_millis(6_200) {
        let elapsed_ms = capture_start.elapsed().as_millis() as u64;
        while next_video_ms <= elapsed_ms {
            if !manager.push_video(
                "rover-a",
                VideoFrame {
                    metadata: VideoFrameMetadata {
                        frame_id: video_frame_id,
                        capture_timestamp_ms: 1_000 + next_video_ms,
                        width: 640,
                        height: 480,
                    },
                    payload: jpeg.clone(),
                },
            ) {
                video_admission_drops += 1;
            }
            video_frame_id += 1;
            // The rover's published view stream is 15 FPS. The production
            // recorder duplicates between capture timestamps for its 30-FPS
            // encoder input, which is the pressure pattern from the live bug.
            next_video_ms += 66;
        }
        while next_audio_ms <= elapsed_ms {
            let _ = manager.push_audio(
                "rover-a",
                AudioFrame {
                    metadata: AudioFrameMetadata {
                        stream_id: Uuid::nil(),
                        frame_id: audio_frame_id,
                        capture_timestamp_ms: 1_000 + next_audio_ms,
                        sample_rate: 16_000,
                        channels: 1,
                        sample_count: 800,
                        format: PcmSampleFormat::S16Le,
                    },
                    payload: vec![0; 1_600],
                },
            );
            audio_frame_id += 1;
            next_audio_ms += 50;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    manager.stop(&session.recording_id).unwrap();
    let mut completed = Vec::new();
    let completion_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < completion_deadline {
        completed.extend(manager.reap());
        if !completed.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(completed.len(), 1, "statuses: {completed:?}");
    assert_eq!(
        completed[0].state,
        robo_rover_lib::RecordingSessionState::Completed,
        "statuses: {completed:?}"
    );

    let clips = manager
        .catalog()
        .list(Some("rover-a"), Some("captures"))
        .unwrap();
    assert_eq!(clips.len(), 1);
    let output = root
        .path()
        .join("captures")
        .join(format!("{}.mp4", clips[0].recording_id));
    let actual_duration_ms = mp4_duration_ms(&output);
    let manifest_path = root
        .path()
        .join("captures")
        .join(format!("{}.manifest.json", clips[0].recording_id));
    let manifest: RecordingManifest =
        serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
    assert!(
        actual_duration_ms >= 5_800,
        "encoder truncated the capture interval: actual={actual_duration_ms}ms, clip={:?}",
        clips[0]
    );
    assert!(
        clips[0].duration_ms.abs_diff(actual_duration_ms) <= 100,
        "manifest duration diverged from MP4: manifest={}ms, actual={actual_duration_ms}ms",
        clips[0].duration_ms
    );
    assert_eq!(manifest.dropped_video, video_admission_drops);
    assert_eq!(video_admission_drops, 0, "unexpected video admission drops");
}
