use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const STDERR_CAP: usize = 64 * 1024;
const FFPROBE_TIMEOUT: Duration = Duration::from_secs(15);

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
    video: Option<os_pipe::PipeWriter>,
    audio: Option<os_pipe::PipeWriter>,
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
                video: Some(video_writer),
                audio: Some(audio_writer),
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
        let writer = self
            .video
            .as_mut()
            .ok_or_else(|| "video pipe is closed".to_string())?;
        write_pipe(writer, payload, stop, "video")
    }

    pub fn write_audio(&mut self, payload: &[u8]) -> Result<(), String> {
        self.write_audio_until(payload, None)
    }

    pub fn write_audio_until(
        &mut self,
        payload: &[u8],
        stop: Option<&AtomicBool>,
    ) -> Result<(), String> {
        let writer = self
            .audio
            .as_mut()
            .ok_or_else(|| "audio pipe is closed".to_string())?;
        write_pipe(writer, payload, stop, "audio")
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

    pub fn close_inputs(&mut self) {
        self.video.take();
        self.audio.take();
    }

    pub fn wait(mut self, timeout: Duration) -> Result<ProcessOutcome, String> {
        self.close_inputs();
        let deadline = Instant::now() + timeout;
        loop {
            match self
                .child
                .try_wait()
                .map_err(|e| format!("wait ffmpeg: {e}"))?
            {
                Some(status) => {
                    return Ok(ProcessOutcome {
                        success: status.success(),
                        stderr: self.stderr_text(),
                    })
                }
                None if Instant::now() >= deadline => {
                    self.child.kill().map_err(|e| format!("kill ffmpeg: {e}"))?;
                    let status = self.child.wait().map_err(|e| format!("reap ffmpeg: {e}"))?;
                    return Ok(ProcessOutcome {
                        success: false,
                        stderr: format!("ffmpeg timeout; {status}; {}", self.stderr_text()),
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

#[cfg(not(unix))]
fn set_nonblocking(_writer: &os_pipe::PipeWriter) -> std::io::Result<()> {
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
