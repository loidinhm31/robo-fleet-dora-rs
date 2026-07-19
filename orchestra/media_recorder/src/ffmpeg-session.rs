use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const STDERR_CAP: usize = 64 * 1024;
const FFPROBE_TIMEOUT: Duration = Duration::from_secs(15);
// A session can retain up to the recorder's default eight pre-video audio
// frames. Keep that initial burst schedulable so the video pump can provide
// FFmpeg's first probe frame instead of letting the audio pump block the sole
// session worker. The queues remain bounded and are private to each child.
const PIPE_QUEUE_CAPACITY: usize = 8;
const PIPE_SEND_TIMEOUT: Duration = Duration::from_millis(50);
// A coalesced tail retains elapsed video time without retaining every JPEG.
// Cap its logical work too: a persistently stalled encoder must fail within
// finalization limits instead of turning an explicit stop into an unbounded
// catch-up encode.
const MAX_COALESCED_VIDEO_COPIES: u32 = 512;

#[derive(Debug, Clone)]
pub struct FfmpegSpec {
    pub executable: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub sample_rate: u32,
    pub channels: u16,
    pub output: PathBuf,
}

impl FfmpegSpec {
    pub fn args(&self, video_fd: i32, audio_fd: i32) -> Vec<String> {
        vec![
            "-hide_banner".into(),
            "-nostdin".into(),
            "-loglevel".into(),
            "warning".into(),
            // Both pipe formats are fixed. Avoid FFmpeg's default multi-second
            // stream-analysis window, which can fill bounded live pipes before
            // either dedicated writer is consumed.
            "-analyzeduration".into(),
            "0".into(),
            "-probesize".into(),
            "32".into(),
            "-thread_queue_size".into(),
            "512".into(),
            "-f".into(),
            "mjpeg".into(),
            "-framerate".into(),
            self.fps.to_string(),
            "-i".into(),
            format!("pipe:{video_fd}"),
            "-thread_queue_size".into(),
            "512".into(),
            "-f".into(),
            "s16le".into(),
            "-ar".into(),
            self.sample_rate.to_string(),
            "-ac".into(),
            self.channels.to_string(),
            "-i".into(),
            format!("pipe:{audio_fd}"),
            "-map".into(),
            "0:v:0".into(),
            "-map".into(),
            "1:a:0".into(),
            "-c:v".into(),
            "libx264".into(),
            "-preset".into(),
            "veryfast".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-c:a".into(),
            "aac".into(),
            "-ac".into(),
            "1".into(),
            "-movflags".into(),
            "+faststart".into(),
            "-f".into(),
            "mp4".into(),
            "-y".into(),
            self.output.to_string_lossy().into_owned(),
        ]
    }
}

pub struct FfmpegSession {
    child: Child,
    video: PipePump,
    audio: PipePump,
    stderr: Arc<Mutex<Vec<u8>>>,
}

impl FfmpegSession {
    pub fn spawn(spec: &FfmpegSpec) -> Result<Self, String> {
        #[cfg(not(unix))]
        {
            let _ = spec;
            return Err("media recorder currently requires Unix anonymous pipes".into());
        }
        #[cfg(unix)]
        {
            use os_pipe::pipe;
            use std::os::fd::{AsRawFd, RawFd};
            use std::os::unix::process::CommandExt;

            let (video_reader, video_writer) = pipe().map_err(|e| format!("video pipe: {e}"))?;
            let (audio_reader, audio_writer) = pipe().map_err(|e| format!("audio pipe: {e}"))?;
            let video_fd: RawFd = video_reader.as_raw_fd();
            let audio_fd: RawFd = audio_reader.as_raw_fd();
            let mut command = Command::new(&spec.executable);
            command
                .args(spec.args(3, 4))
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            unsafe {
                command.pre_exec(move || {
                    if libc::dup2(video_fd, 3) == -1 || libc::dup2(audio_fd, 4) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::fcntl(3, libc::F_SETFD, 0) == -1
                        || libc::fcntl(4, libc::F_SETFD, 0) == -1
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            set_nonblocking(&video_writer).map_err(|e| format!("video pipe: {e}"))?;
            set_nonblocking(&audio_writer).map_err(|e| format!("audio pipe: {e}"))?;
            let mut child = command.spawn().map_err(|e| format!("spawn ffmpeg: {e}"))?;
            drop(video_reader);
            drop(audio_reader);
            let stderr = Arc::new(Mutex::new(Vec::new()));
            if let Some(mut stream) = child.stderr.take() {
                let capture = Arc::clone(&stderr);
                thread::spawn(move || {
                    let mut buffer = [0u8; 4096];
                    loop {
                        match stream.read(&mut buffer) {
                            Ok(0) | Err(_) => break,
                            Ok(size) => {
                                let mut output = capture.lock().expect("stderr lock");
                                if output.len() < STDERR_CAP {
                                    let remaining = STDERR_CAP - output.len();
                                    output.extend_from_slice(&buffer[..size.min(remaining)]);
                                }
                            }
                        }
                    }
                });
            }
            Ok(Self {
                child,
                video: PipePump::spawn(video_writer, "video", true),
                audio: PipePump::spawn(audio_writer, "audio", false),
                stderr,
            })
        }
    }

    pub fn write_video(&mut self, payload: &[u8]) -> Result<(), String> {
        self.write_video_until(payload, None)
    }

    pub fn write_video_until(
        &mut self,
        payload: &[u8],
        stop: Option<&AtomicBool>,
    ) -> Result<(), String> {
        self.video.send(payload.to_vec(), stop)
    }

    pub fn write_audio(&mut self, payload: &[u8]) -> Result<(), String> {
        self.write_audio_until(payload, None)
    }

    pub fn write_audio_until(
        &mut self,
        payload: &[u8],
        stop: Option<&AtomicBool>,
    ) -> Result<(), String> {
        self.audio.send(payload.to_vec(), stop)
    }

    pub fn write_audio_silence(&mut self, bytes: u64) -> Result<(), String> {
        self.write_audio_silence_until(bytes, None)
    }

    pub fn write_audio_silence_until(
        &mut self,
        bytes: u64,
        stop: Option<&AtomicBool>,
    ) -> Result<(), String> {
        let mut remaining = bytes;
        let zeros = [0u8; 8192];
        while remaining > 0 {
            let size = remaining.min(zeros.len() as u64) as usize;
            self.write_audio_until(&zeros[..size], stop)?;
            remaining -= size as u64;
        }
        Ok(())
    }

    pub fn wait(mut self, timeout: Duration) -> Result<ProcessOutcome, String> {
        let deadline = Instant::now() + timeout;
        self.close_input_senders();
        while !self.inputs_finished() {
            if Instant::now() >= deadline {
                self.cancel_inputs();
                let _ = self.child.kill();
                let status = self.child.wait().map_err(|e| format!("reap ffmpeg: {e}"))?;
                let writer_error = self.join_inputs().err();
                return Ok(ProcessOutcome {
                    success: false,
                    stderr: format!(
                        "ffmpeg timeout; {status}; {}",
                        self.stderr_with(writer_error.as_deref())
                    ),
                });
            }
            thread::sleep(Duration::from_millis(20));
        }
        let writer_error = self.join_inputs().err();
        loop {
            match self
                .child
                .try_wait()
                .map_err(|e| format!("wait ffmpeg: {e}"))?
            {
                Some(status) => {
                    return Ok(ProcessOutcome {
                        success: status.success() && writer_error.is_none(),
                        stderr: self.stderr_with(writer_error.as_deref()),
                    })
                }
                None if Instant::now() >= deadline => {
                    self.child.kill().map_err(|e| format!("kill ffmpeg: {e}"))?;
                    let status = self.child.wait().map_err(|e| format!("reap ffmpeg: {e}"))?;
                    return Ok(ProcessOutcome {
                        success: false,
                        stderr: format!(
                            "ffmpeg timeout; {status}; {}",
                            self.stderr_with(writer_error.as_deref())
                        ),
                    });
                }
                None => thread::sleep(Duration::from_millis(20)),
            }
        }
    }

    fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr.lock().expect("stderr lock"))
            .trim()
            .to_string()
    }

    fn stderr_with(&self, writer_error: Option<&str>) -> String {
        let stderr = self.stderr_text();
        match writer_error {
            Some(error) if stderr.is_empty() => error.to_string(),
            Some(error) => format!("{stderr}; {error}"),
            None => stderr,
        }
    }

    fn close_input_senders(&mut self) {
        self.video.close_sender();
        self.audio.close_sender();
    }

    fn inputs_finished(&self) -> bool {
        self.video.is_finished() && self.audio.is_finished()
    }

    fn cancel_inputs(&self) {
        self.video.cancel();
        self.audio.cancel();
    }

    fn join_inputs(&mut self) -> Result<(), String> {
        self.video.join()?;
        self.audio.join()
    }
}

struct PipePump {
    cancel: Arc<AtomicBool>,
    queue: Arc<PipeQueue>,
    failure: Arc<Mutex<Option<String>>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl PipePump {
    fn spawn(writer: os_pipe::PipeWriter, label: &'static str, coalesce: bool) -> Self {
        let queue = Arc::new(PipeQueue::new(coalesce));
        let cancel = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        let worker_queue = Arc::clone(&queue);
        let worker_cancel = Arc::clone(&cancel);
        let worker_failure = Arc::clone(&failure);
        let worker = thread::spawn(move || {
            pump_pipe(writer, worker_queue, worker_cancel, worker_failure, label)
        });
        Self {
            cancel,
            queue,
            failure,
            worker: Some(worker),
        }
    }

    fn send(&self, payload: Vec<u8>, stop: Option<&AtomicBool>) -> Result<(), String> {
        self.check_failure()?;
        self.queue.enqueue(payload, stop)
    }

    fn close_sender(&mut self) {
        self.queue.close();
    }

    fn is_finished(&self) -> bool {
        self.worker
            .as_ref()
            .is_none_or(thread::JoinHandle::is_finished)
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    fn join(&mut self) -> Result<(), String> {
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| "FFmpeg input writer panicked".to_string())?;
        }
        self.check_failure()
    }

    fn check_failure(&self) -> Result<(), String> {
        self.failure
            .lock()
            .map_err(|_| "FFmpeg input writer failure lock poisoned".to_string())?
            .clone()
            .map_or(Ok(()), Err)
    }
}

struct PipeQueue {
    state: Mutex<PipeQueueState>,
    ready: Condvar,
    coalesce: bool,
}

#[derive(Default)]
struct PipeQueueState {
    entries: VecDeque<PipeEntry>,
    closed: bool,
}

struct PipeEntry {
    payload: Vec<u8>,
    copies: u32,
}

impl PipeQueue {
    fn new(coalesce: bool) -> Self {
        Self {
            state: Mutex::new(PipeQueueState::default()),
            ready: Condvar::new(),
            coalesce,
        }
    }

    fn enqueue(&self, payload: Vec<u8>, stop: Option<&AtomicBool>) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "FFmpeg input queue lock poisoned".to_string())?;
        loop {
            if state.closed {
                return Err("FFmpeg input pipe is closed".into());
            }
            if state.entries.len() < PIPE_QUEUE_CAPACITY {
                state.entries.push_back(PipeEntry { payload, copies: 1 });
                self.ready.notify_one();
                return Ok(());
            }
            if self.coalesce {
                let tail = state
                    .entries
                    .back_mut()
                    .expect("full pipe queue has a tail");
                if tail.copies >= MAX_COALESCED_VIDEO_COPIES {
                    return Err("FFmpeg video handoff backlog limit reached".into());
                }
                tail.payload = payload;
                tail.copies += 1;
                self.ready.notify_one();
                return Ok(());
            }
            let (next, _) = self
                .ready
                .wait_timeout(state, PIPE_SEND_TIMEOUT)
                .map_err(|_| "FFmpeg input queue lock poisoned".to_string())?;
            state = next;
            if stop.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err("FFmpeg input write interrupted by stop".into());
            }
        }
    }

    fn next(&self) -> Option<PipeEntry> {
        let mut state = self.state.lock().ok()?;
        loop {
            if let Some(entry) = state.entries.pop_front() {
                self.ready.notify_all();
                return Some(entry);
            }
            if state.closed {
                return None;
            }
            state = self.ready.wait(state).ok()?;
        }
    }

    fn close(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.closed = true;
            self.ready.notify_all();
        }
    }
}

fn pump_pipe(
    mut writer: os_pipe::PipeWriter,
    queue: Arc<PipeQueue>,
    cancel: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
    label: &str,
) {
    while let Some(entry) = queue.next() {
        for _ in 0..entry.copies {
            if cancel.load(Ordering::Acquire) {
                return;
            }
            if let Err(error) = write_pipe(&mut writer, &entry.payload, Some(&cancel), label) {
                if let Ok(mut recorded) = failure.lock() {
                    *recorded = Some(error);
                }
                return;
            }
        }
    }
}

#[cfg(unix)]
fn set_nonblocking(writer: &os_pipe::PipeWriter) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;

    let fd = writer.as_raw_fd();
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags == -1 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn write_pipe(
    writer: &mut os_pipe::PipeWriter,
    payload: &[u8],
    stop: Option<&AtomicBool>,
    label: &str,
) -> Result<(), String> {
    let mut offset = 0usize;
    while offset < payload.len() {
        match writer.write(&payload[offset..]) {
            Ok(0) => return Err(format!("{label} pipe closed")),
            Ok(written) => offset += written,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if stop.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
                    return Err(format!("{label} pipe write interrupted by stop"));
                }
                #[cfg(unix)]
                {
                    use std::os::fd::AsRawFd;
                    let mut poll_fd = libc::pollfd {
                        fd: writer.as_raw_fd(),
                        events: libc::POLLOUT,
                        revents: 0,
                    };
                    let result = unsafe { libc::poll(&mut poll_fd, 1, 50) };
                    if result < 0 {
                        let error = std::io::Error::last_os_error();
                        if error.kind() == std::io::ErrorKind::Interrupted {
                            continue;
                        }
                        return Err(format!("write {label} pipe: {error}"));
                    }
                }
                #[cfg(not(unix))]
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(format!("write {label} pipe: {error}")),
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ProcessOutcome {
    pub success: bool,
    pub stderr: String,
}

pub fn ffprobe_json(executable: &Path, input: &Path) -> Result<serde_json::Value, String> {
    let mut child = Command::new(executable);
    let mut child = child
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=format_name,duration,size",
            "-show_entries",
            "stream=codec_type,codec_name,channels",
            "-of",
            "json",
        ])
        .arg(input)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("run ffprobe: {e}"))?;
    let deadline = Instant::now() + FFPROBE_TIMEOUT;
    loop {
        match child.try_wait().map_err(|e| format!("wait ffprobe: {e}"))? {
            Some(_) => break,
            None if Instant::now() >= deadline => {
                child.kill().map_err(|e| format!("kill ffprobe: {e}"))?;
                let _ = child.wait();
                return Err("ffprobe timeout".into());
            }
            None => thread::sleep(Duration::from_millis(20)),
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("collect ffprobe: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|e| format!("invalid ffprobe JSON: {e}"))
}

pub fn validate_mp4(executable: &Path, input: &Path) -> Result<(), String> {
    let value = ffprobe_json(executable, input)?;
    let format = value
        .pointer("/format/format_name")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if !format.split(',').any(|name| name == "mov" || name == "mp4") {
        return Err("final media is not an MP4 container".into());
    }
    let duration = value
        .pointer("/format/duration")
        .and_then(|value| {
            value
                .as_str()
                .and_then(|value| value.parse::<f64>().ok())
                .or_else(|| value.as_f64())
        })
        .ok_or("ffprobe returned no duration")?;
    if !duration.is_finite() || duration <= 0.0 {
        return Err("final media has invalid duration".into());
    }
    let streams = value
        .pointer("/streams")
        .and_then(|v| v.as_array())
        .ok_or("ffprobe returned no streams")?;
    let video = streams
        .iter()
        .filter(|s| s["codec_type"] == "video" && s["codec_name"] == "h264")
        .count();
    let audio = streams
        .iter()
        .filter(|s| {
            s["codec_type"] == "audio"
                && s["codec_name"] == "aac"
                && s["channels"].as_u64() == Some(1)
        })
        .count();
    if video != 1 || audio != 1 {
        return Err(format!(
            "expected one H264 and one AAC stream, got video={video} audio={audio}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::FfmpegSpec;
    use std::path::PathBuf;

    #[test]
    fn builder_uses_fixed_flags_and_fd_inputs() {
        let args = FfmpegSpec {
            executable: PathBuf::from("ffmpeg"),
            width: 640,
            height: 480,
            fps: 30,
            sample_rate: 16_000,
            channels: 1,
            output: PathBuf::from("/tmp/out.mp4"),
        }
        .args(3, 4);
        assert!(args.windows(2).any(|pair| pair == ["-i", "pipe:3"]));
        assert!(args.windows(2).any(|pair| pair == ["-i", "pipe:4"]));
        assert!(args.contains(&"-nostdin".into()));
        assert!(!args.iter().any(|arg| arg.contains("sh -c")));
    }
}
