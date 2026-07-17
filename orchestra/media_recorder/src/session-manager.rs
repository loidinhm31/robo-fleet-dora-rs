use crate::clip_catalog::{ClipCatalog, RecordingManifest};
use crate::config::RecorderConfig;
use crate::ffmpeg_session::{FfmpegSession, FfmpegSpec};
use crate::frame_timeline::{AudioFrame, FrameTimeline, VideoFrame};
use crate::path_resolver::PathResolver;
use crossbeam_channel::{unbounded, Receiver, Sender};
use robo_rover_lib::{
    RecordingAudioCodec, RecordingClip, RecordingReasonCode, RecordingSessionState,
    RecordingVideoCodec, RECORDING_PROTOCOL_VERSION,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct StartRequest {
    pub request_id: String,
    pub entity_id: String,
    pub relative_directory: String,
}

#[derive(Debug, Clone)]
pub struct SessionStatus {
    pub request_id: String,
    pub recording_id: String,
    pub entity_id: String,
    pub state: RecordingSessionState,
    pub started_at_ms: Option<u64>,
    pub duration_ms: u64,
    pub bytes_written: u64,
    pub reason_code: Option<RecordingReasonCode>,
}

impl SessionStatus {
    pub fn wire(&self) -> robo_rover_lib::RecordingSessionStatus {
        robo_rover_lib::RecordingSessionStatus {
            protocol_version: RECORDING_PROTOCOL_VERSION,
            request_id: self.request_id.clone(),
            recording_id: self.recording_id.clone(),
            entity_id: self.entity_id.clone(),
            state: self.state,
            started_at_ms: self.started_at_ms,
            duration_ms: self.duration_ms,
            bytes_written: self.bytes_written,
            reason_code: self.reason_code,
        }
    }
}

#[derive(Debug)]
enum Input {
    Video(VideoFrame),
    Audio(AudioFrame),
}

#[derive(Default)]
struct InputQueue {
    items: VecDeque<Input>,
}

struct BoundedInputs {
    inner: Mutex<InputQueue>,
    wake: Condvar,
    capacity: usize,
    dropped_video: std::sync::atomic::AtomicU64,
}

impl BoundedInputs {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(InputQueue::default()),
            wake: Condvar::new(),
            capacity,
            dropped_video: std::sync::atomic::AtomicU64::new(0),
        }
    }
    fn video(&self, frame: VideoFrame) -> bool {
        let mut queue = self.inner.lock().expect("queue lock");
        let dropped = if queue.items.len() >= self.capacity {
            if let Some(index) = queue
                .items
                .iter()
                .position(|item| matches!(item, Input::Video(_)))
            {
                queue.items.remove(index);
                true
            } else {
                self.dropped_video
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return false;
            }
        } else {
            false
        };
        if dropped {
            self.dropped_video
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        queue.items.push_back(Input::Video(frame));
        self.wake.notify_one();
        !dropped
    }
    fn audio(&self, frame: AudioFrame) -> bool {
        let mut queue = self.inner.lock().expect("queue lock");
        if queue.items.len() >= self.capacity {
            return false;
        }
        queue.items.push_back(Input::Audio(frame));
        self.wake.notify_one();
        true
    }
    fn pop(&self, timeout: Duration) -> Option<Input> {
        let mut queue = self.inner.lock().expect("queue lock");
        if let Some(item) = queue.items.pop_front() {
            return Some(item);
        }
        let (mut queue, _) = self.wake.wait_timeout(queue, timeout).expect("queue wait");
        queue.items.pop_front()
    }
    fn wake(&self) {
        self.wake.notify_all();
    }
    fn dropped_video(&self) -> u64 {
        self.dropped_video
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .map(|queue| queue.items.is_empty())
            .unwrap_or(true)
    }
}

struct ActiveSession {
    id: String,
    inputs: Arc<BoundedInputs>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    status: Arc<Mutex<SessionStatus>>,
    join: Option<thread::JoinHandle<()>>,
}

pub struct SessionManager {
    config: RecorderConfig,
    resolver: PathResolver,
    catalog: ClipCatalog,
    sessions: HashMap<String, ActiveSession>,
    finished: HashSet<String>,
    completed_rx: Receiver<SessionStatus>,
    completed_tx: Sender<SessionStatus>,
}

impl SessionManager {
    pub fn new(config: RecorderConfig) -> Result<Self, String> {
        let resolver = PathResolver::new(&config.recording_root)?;
        let catalog = ClipCatalog::new(resolver.clone(), config.ffprobe_path.clone());
        let (_, issues) = catalog.scan();
        for issue in issues {
            tracing::warn!(path = %issue.path.display(), detail = %issue.detail, "recording catalog issue at startup");
        }
        let (completed_tx, completed_rx) = unbounded();
        Ok(Self {
            config,
            resolver,
            catalog,
            sessions: HashMap::new(),
            finished: HashSet::new(),
            completed_rx,
            completed_tx,
        })
    }

    pub fn start(&mut self, request: StartRequest) -> Result<SessionStatus, String> {
        robo_rover_lib::validate_uuid("request_id", &request.request_id)?;
        robo_rover_lib::validate_id("entity_id", &request.entity_id)?;
        robo_rover_lib::validate_relative_directory(&request.relative_directory)?;
        let _ = self.reap();
        if self.sessions.contains_key(&request.entity_id) {
            return Err("rover already has an active recording".into());
        }
        if self.sessions.len() >= self.config.max_concurrent {
            return Err("recording concurrency limit reached".into());
        }
        if available_bytes(&self.config.recording_root) < self.config.min_free_bytes {
            return Err("recording free-space guard rejected start".into());
        }
        let directory = self.resolver.directory(&request.relative_directory)?;
        let id = Uuid::new_v4().to_string();
        let inputs = Arc::new(BoundedInputs::new(self.config.queue_capacity));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let status = SessionStatus {
            request_id: request.request_id.clone(),
            recording_id: id.clone(),
            entity_id: request.entity_id.clone(),
            state: RecordingSessionState::Starting,
            started_at_ms: None,
            duration_ms: 0,
            bytes_written: 0,
            reason_code: None,
        };
        let worker_status = status.clone();
        let config = self.config.clone();
        let resolver = self.resolver.clone();
        let catalog = self.catalog.clone();
        let worker_inputs = Arc::clone(&inputs);
        let worker_stop = Arc::clone(&stop);
        let status_ref = Arc::new(Mutex::new(status.clone()));
        let worker_status_ref = Arc::clone(&status_ref);
        let tx = self.completed_tx.clone();
        let worker_directory = directory.clone();
        let join = thread::Builder::new()
            .name(format!("recording-{id}"))
            .spawn(move || {
                run_worker(
                    worker_status,
                    worker_status_ref,
                    worker_directory,
                    resolver,
                    catalog,
                    config,
                    worker_inputs,
                    worker_stop,
                    tx,
                );
            })
            .map_err(|e| format!("spawn recorder worker: {e}"))?;
        self.sessions.insert(
            request.entity_id.clone(),
            ActiveSession {
                id: id.clone(),
                inputs,
                stop,
                status: status_ref,
                join: Some(join),
            },
        );
        Ok(status)
    }

    pub fn stop(&mut self, recording_id: &str) -> Result<(), String> {
        let _ = self.reap();
        if self.finished.contains(recording_id) {
            return Ok(());
        }
        let session = self
            .sessions
            .values()
            .find(|session| session.id == recording_id)
            .ok_or_else(|| "recording session not found".to_string())?;
        session
            .stop
            .store(true, std::sync::atomic::Ordering::Release);
        if let Ok(mut status) = session.status.lock() {
            status.state = RecordingSessionState::Stopping;
        }
        session.inputs.wake();
        Ok(())
    }

    pub fn push_video(&self, entity_id: &str, frame: VideoFrame) -> bool {
        self.sessions
            .get(entity_id)
            .is_some_and(|session| session.inputs.video(frame))
    }

    pub fn push_audio(&self, entity_id: &str, frame: AudioFrame) -> bool {
        self.sessions
            .get(entity_id)
            .is_some_and(|session| session.inputs.audio(frame))
    }

    pub fn reap(&mut self) -> Vec<SessionStatus> {
        let mut completed = Vec::new();
        while let Ok(status) = self.completed_rx.try_recv() {
            if let Some(mut session) = self.sessions.remove(&status.entity_id) {
                if let Some(join) = session.join.take() {
                    let _ = join.join();
                }
            }
            self.finished.insert(status.recording_id.clone());
            completed.push(status);
        }
        completed
    }

    pub fn statuses(&mut self) -> Vec<SessionStatus> {
        self.sessions
            .values()
            .filter_map(|s| s.status.lock().ok().map(|status| status.clone()))
            .collect()
    }

    pub fn catalog(&self) -> &ClipCatalog {
        &self.catalog
    }

    pub fn shutdown(&mut self) -> Vec<SessionStatus> {
        for session in self.sessions.values() {
            session
                .stop
                .store(true, std::sync::atomic::Ordering::Release);
            session.inputs.wake();
        }
        let sessions = std::mem::take(&mut self.sessions);
        for (_, mut session) in sessions {
            if let Some(join) = session.join.take() {
                let _ = join.join();
            }
        }
        self.reap()
    }
}

fn run_worker(
    mut status: SessionStatus,
    status_ref: Arc<Mutex<SessionStatus>>,
    directory: PathBuf,
    resolver: PathResolver,
    catalog: ClipCatalog,
    config: RecorderConfig,
    inputs: Arc<BoundedInputs>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    done: Sender<SessionStatus>,
) {
    let started = Instant::now();
    let partial = resolver
        .partial()
        .join(format!("{}.mp4.partial", status.recording_id));
    let mut timeline = FrameTimeline::default();
    let mut ffmpeg: Option<FfmpegSession> = None;
    let mut dimensions = None;
    let mut last_video_pts = 0;
    let mut failure = None;
    let mut audio_format: Option<(u32, u16)> = None;
    let mut previous_video: Option<Vec<u8>> = None;
    let mut last_written_pts: Option<u64> = None;
    let mut pending_audio: VecDeque<AudioFrame> = VecDeque::new();
    while !stop.load(std::sync::atomic::Ordering::Acquire) || ffmpeg.is_none() || !inputs.is_empty()
    {
        if stop.load(std::sync::atomic::Ordering::Acquire) && ffmpeg.is_none() && inputs.is_empty()
        {
            failure = Some(RecordingReasonCode::Internal);
            break;
        }
        if ffmpeg.is_none() && started.elapsed() > Duration::from_millis(config.startup_timeout_ms)
        {
            failure = Some(RecordingReasonCode::StartupTimeout);
            break;
        }
        let Some(input) = inputs.pop(Duration::from_millis(50)) else {
            continue;
        };
        match input {
            Input::Video(frame) => {
                if frame.payload.len() < 4
                    || frame.payload[..2] != [0xff, 0xd8]
                    || frame.payload[frame.payload.len() - 2..] != [0xff, 0xd9]
                {
                    continue;
                }
                if ffmpeg.is_none() {
                    if let Some(timestamp) = pending_audio
                        .iter()
                        .map(|pending| pending.metadata.capture_timestamp_ms)
                        .min()
                    {
                        timeline.set_origin(timestamp.min(frame.metadata.capture_timestamp_ms));
                    }
                }
                let pts = match timeline.video_pts(frame.metadata) {
                    Ok(pts) => pts,
                    Err(_) => continue,
                };
                last_video_pts = pts;
                if ffmpeg.is_none() {
                    let (width, height) = (frame.metadata.width, frame.metadata.height);
                    dimensions = Some((width, height));
                    let spec = FfmpegSpec {
                        executable: config.ffmpeg_path.clone(),
                        width,
                        height,
                        fps: config.video_fps,
                        sample_rate: audio_format
                            .map(|format| format.0)
                            .unwrap_or(config.audio_sample_rate),
                        channels: audio_format
                            .map(|format| format.1)
                            .unwrap_or(config.audio_channels),
                        output: partial.clone(),
                    };
                    ffmpeg = match FfmpegSession::spawn(&spec) {
                        Ok(process) => Some(process),
                        Err(_) => {
                            failure = Some(RecordingReasonCode::EncoderFailed);
                            break;
                        }
                    };
                    status.state = RecordingSessionState::Recording;
                    status.started_at_ms = Some(
                        pending_audio
                            .iter()
                            .map(|pending| pending.metadata.capture_timestamp_ms)
                            .chain(std::iter::once(frame.metadata.capture_timestamp_ms))
                            .min()
                            .unwrap_or(frame.metadata.capture_timestamp_ms),
                    );
                    if let Ok(mut shared) = status_ref.lock() {
                        *shared = status.clone();
                    }
                    let mut pending_failed = false;
                    for pending in pending_audio.drain(..) {
                        let silence = match timeline.audio_prefix_silence(pending.metadata) {
                            Ok(silence) => silence,
                            Err(_) => continue,
                        };
                        let Some(process) = ffmpeg.as_mut() else {
                            pending_failed = true;
                            break;
                        };
                        if process.write_audio_silence(silence).is_err()
                            || process.write_audio(&pending.payload).is_err()
                        {
                            pending_failed = true;
                            break;
                        }
                    }
                    if pending_failed {
                        failure = Some(RecordingReasonCode::EncoderFailed);
                        break;
                    }
                } else if dimensions != Some((frame.metadata.width, frame.metadata.height)) {
                    continue;
                }
                if let Some(process) = ffmpeg.as_mut() {
                    let interval = (1000 / u64::from(config.video_fps)).max(1);
                    if let (Some(previous), Some(last_pts)) = (&previous_video, last_written_pts) {
                        let repeats =
                            pts.saturating_sub(last_pts.saturating_add(interval)) / interval;
                        let mut repeat_failed = false;
                        for _ in 0..repeats {
                            if process.write_video(previous).is_err() {
                                repeat_failed = true;
                                break;
                            }
                        }
                        if repeat_failed {
                            failure = Some(RecordingReasonCode::EncoderFailed);
                            break;
                        }
                    }
                    if let Err(error) = process.write_video(&frame.payload) {
                        tracing::warn!(recording_id = %status.recording_id, %error, "video pipe write failed");
                        failure = Some(RecordingReasonCode::EncoderFailed);
                        break;
                    }
                    previous_video = Some(frame.payload);
                    last_written_pts = Some(pts);
                }
            }
            Input::Audio(frame) => {
                if frame.metadata.format != robo_rover_lib::PcmSampleFormat::S16Le
                    || frame
                        .metadata
                        .validate_payload_len(frame.payload.len())
                        .is_err()
                {
                    continue;
                }
                if let Some((sample_rate, channels)) = audio_format {
                    if (sample_rate, channels)
                        != (frame.metadata.sample_rate, frame.metadata.channels)
                    {
                        continue;
                    }
                } else {
                    audio_format = Some((frame.metadata.sample_rate, frame.metadata.channels));
                }
                if ffmpeg.is_none() {
                    if pending_audio.len() >= config.queue_capacity {
                        pending_audio.pop_front();
                    }
                    pending_audio.push_back(frame);
                    continue;
                }
                let silence_bytes = match timeline.audio_prefix_silence(frame.metadata) {
                    Ok(silence_bytes) => silence_bytes,
                    Err(_) => continue,
                };
                if let Some(process) = ffmpeg.as_mut() {
                    if process.write_audio_silence(silence_bytes).is_err()
                        || process.write_audio(&frame.payload).is_err()
                    {
                        failure = Some(RecordingReasonCode::EncoderFailed);
                        break;
                    }
                }
            }
        }
        if ffmpeg.is_some() && last_video_pts >= config.max_duration_ms {
            break;
        }
        if partial
            .metadata()
            .map(|m| m.len() > config.max_output_bytes)
            .unwrap_or(false)
        {
            failure = Some(RecordingReasonCode::ResourceLimit);
            break;
        }
    }
    if ffmpeg.is_none() && failure.is_none() {
        failure = Some(RecordingReasonCode::StartupTimeout);
    }
    if let Some(mut process) = ffmpeg {
        timeline.counters.dropped_video = inputs.dropped_video();
        let silence_bytes = timeline.finish_silence(
            last_video_pts.saturating_add(1000 / u64::from(config.video_fps)),
            audio_format
                .map(|format| format.0)
                .unwrap_or(config.audio_sample_rate),
            audio_format
                .map(|format| format.1)
                .unwrap_or(config.audio_channels),
        );
        if failure.is_none() {
            let _ = process.write_audio_silence(silence_bytes);
        }
        let outcome = process.wait(Duration::from_millis(config.finalization_timeout_ms));
        if outcome.as_ref().map(|o| !o.success).unwrap_or(true) {
            tracing::warn!(recording_id = %status.recording_id, ?outcome, "ffmpeg process failed");
        }
        if failure.is_none() && outcome.as_ref().map(|o| !o.success).unwrap_or(true) {
            failure = Some(RecordingReasonCode::EncoderFailed);
        }
    }
    if failure.is_none() {
        let relative_path = directory
            .strip_prefix(resolver.root())
            .map(|path| {
                path.join(format!("{}.mp4", status.recording_id))
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/")
            })
            .unwrap_or_else(|_| format!("{}.mp4", status.recording_id));
        let bytes = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
        status.duration_ms = timeline
            .duration_ms()
            .max(last_video_pts.saturating_add(1000 / u64::from(config.video_fps)));
        status.bytes_written = bytes;
        let clip = RecordingClip {
            recording_id: status.recording_id.clone(),
            entity_id: status.entity_id.clone(),
            relative_path,
            started_at_ms: status.started_at_ms.unwrap_or(0),
            duration_ms: status.duration_ms,
            bytes_written: bytes,
            video_codec: RecordingVideoCodec::H264,
            audio_codec: RecordingAudioCodec::Aac,
        };
        let manifest = RecordingManifest {
            clip,
            dropped_video: timeline.counters.dropped_video,
            audio_gaps: timeline.counters.audio_gaps,
            silence_samples: timeline.counters.silence_samples,
            timestamp_regressions: timeline.counters.timestamp_regressions,
        };
        if catalog.publish(&partial, &directory, &manifest).is_err() {
            failure = Some(RecordingReasonCode::StorageUnavailable);
        }
    }
    if let Some(reason) = failure {
        status.state = RecordingSessionState::Failed;
        status.reason_code = Some(reason);
        status.duration_ms = timeline
            .duration_ms()
            .max(last_video_pts.saturating_add(1000 / u64::from(config.video_fps)));
        status.bytes_written = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    } else {
        status.state = RecordingSessionState::Completed;
    }
    if let Ok(mut shared) = status_ref.lock() {
        *shared = status.clone();
    }
    let _ = done.send(status);
}

fn available_bytes(path: &PathBuf) -> u64 {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let Ok(path) = CString::new(path.as_os_str().as_bytes()) else {
            return 0;
        };
        let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } == 0 {
            let stats = unsafe { stats.assume_init() };
            return stats.f_bavail.saturating_mul(stats.f_frsize);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::BoundedInputs;
    use crate::frame_timeline::VideoFrame;
    use robo_rover_lib::VideoFrameMetadata;

    #[test]
    fn video_queue_drops_oldest_and_keeps_newest() {
        let queue = BoundedInputs::new(1);
        let frame = |id| VideoFrame {
            metadata: VideoFrameMetadata {
                frame_id: id,
                capture_timestamp_ms: id,
                width: 1,
                height: 1,
            },
            payload: vec![0xff, 0xd8, id as u8, 0xff, 0xd9],
        };
        assert!(queue.video(frame(1)));
        assert!(!queue.video(frame(2)));
        match queue.pop(std::time::Duration::ZERO).unwrap() {
            super::Input::Video(frame) => assert_eq!(frame.metadata.frame_id, 2),
            _ => panic!(),
        }
    }
}
