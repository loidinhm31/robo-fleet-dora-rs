#![cfg(unix)]

use media_recorder::clip_catalog::{ClipCatalog, RecordingManifest};
use media_recorder::config::RecorderConfig;
use media_recorder::ffmpeg_session::{validate_mp4, FfmpegSession, FfmpegSpec};
use media_recorder::frame_timeline::{AudioFrame, VideoFrame};
use media_recorder::path_resolver::PathResolver;
use media_recorder::session_manager::{SessionManager, StartRequest};
use robo_rover_lib::{AudioFrameMetadata, PcmSampleFormat, VideoFrameMetadata};
use robo_rover_lib::{RecordingAudioCodec, RecordingClip, RecordingVideoCodec};
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;
use uuid::Uuid;

fn jpeg_fixture() -> Vec<u8> {
    let output = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x24",
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
        .output()
        .expect("ffmpeg fixture");
    assert!(output.status.success(), "fixture ffmpeg failed");
    output.stdout
}

#[test]
fn synthetic_jpeg_pcm_is_atomically_published_and_probeable() {
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
    let mut completed = Vec::new();
    for _ in 0..100 {
        completed.extend(manager.reap());
        if completed.len() == 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
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
