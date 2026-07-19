use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub recording_root: PathBuf,
    pub ffmpeg_path: PathBuf,
    pub ffprobe_path: PathBuf,
    pub max_concurrent: usize,
    pub max_duration_ms: u64,
    pub max_output_bytes: u64,
    pub startup_timeout_ms: u64,
    pub finalization_timeout_ms: u64,
    pub min_free_bytes: u64,
    pub queue_capacity: usize,
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
    pub video_fps: u32,
}

impl RecorderConfig {
    pub fn from_env() -> Result<Self, String> {
        let root = PathBuf::from(required("RECORDING_ROOT")?);
        let root = root
            .canonicalize()
            .map_err(|e| format!("RECORDING_ROOT is not canonicalizable: {e}"))?;
        let container_mode = env::var("RECORDING_CONTAINER_MODE").as_deref() == Ok("true");
        if !allowed_root(&root, container_mode) {
            return Err("RECORDING_ROOT is outside the configured recording root policy".into());
        }
        if !root.is_dir() {
            return Err("RECORDING_ROOT must be a directory".into());
        }
        let ffmpeg_path = executable(env::var("FFMPEG_PATH").unwrap_or_else(|_| "ffmpeg".into()))?;
        let ffprobe_path =
            executable(env::var("FFPROBE_PATH").unwrap_or_else(|_| "ffprobe".into()))?;
        validate_tools(&ffmpeg_path, &ffprobe_path)?;
        let timestamp_enabled = env::var("RECORDING_TIMESTAMP_ENABLED")
            .map(|value| value != "false")
            .unwrap_or(true);
        let timestamp_font = env::var("RECORDING_TIMESTAMP_FONT").unwrap_or_else(|_| "Sans".into());
        if timestamp_enabled {
            if timestamp_font.contains(std::path::MAIN_SEPARATOR)
                && !Path::new(&timestamp_font).is_file()
            {
                return Err("recording timestamp font is unavailable".into());
            }
            if !has_tool_entry(
                &run_tool(&ffmpeg_path, &["-hide_banner", "-filters"])?,
                "drawtext",
            ) {
                return Err("FFmpeg drawtext filter is unavailable".into());
            }
        }
        validate_root_access(&root)?;
        Ok(Self {
            recording_root: root,
            ffmpeg_path,
            ffprobe_path,
            max_concurrent: number("RECORDING_MAX_CONCURRENT", 64, 1, 1024)? as usize,
            max_duration_ms: number("RECORDING_MAX_DURATION_SECONDS", 3600, 1, 86_400)? * 1000,
            max_output_bytes: number(
                "RECORDING_MAX_OUTPUT_BYTES",
                4 * 1024 * 1024 * 1024,
                1_048_576,
                u64::MAX,
            )?,
            startup_timeout_ms: number("RECORDING_STARTUP_TIMEOUT_SECONDS", 10, 1, 300)? * 1000,
            finalization_timeout_ms: number("RECORDING_FINALIZATION_TIMEOUT_SECONDS", 15, 1, 300)?
                * 1000,
            min_free_bytes: number("RECORDING_MIN_FREE_BYTES", 1_073_741_824, 0, u64::MAX)?,
            queue_capacity: number("RECORDING_QUEUE_CAPACITY", 8, 1, 256)? as usize,
            audio_sample_rate: number("RECORDING_AUDIO_SAMPLE_RATE", 16_000, 8_000, 192_000)?
                as u32,
            audio_channels: number("RECORDING_AUDIO_CHANNELS", 1, 1, 8)? as u16,
            video_fps: number("RECORDING_VIDEO_FPS", 30, 1, 120)? as u32,
        })
    }
}

fn allowed_root(root: &Path, container_mode: bool) -> bool {
    (container_mode && root == Path::new("/recordings"))
        || (!container_mode && root != Path::new("/home") && root.starts_with("/home"))
}

fn validate_tools(ffmpeg: &Path, ffprobe: &Path) -> Result<(), String> {
    let encoders = run_tool(ffmpeg, &["-hide_banner", "-encoders"])?;
    if !has_tool_entry(&encoders, "libx264") {
        return Err("FFmpeg libx264 encoder is unavailable".into());
    }
    if !has_tool_entry(&encoders, "aac") {
        return Err("FFmpeg AAC encoder is unavailable".into());
    }
    let muxers = run_tool(ffmpeg, &["-hide_banner", "-muxers"])?;
    if !has_tool_entry(&muxers, "mp4") {
        return Err("FFmpeg MP4 muxer is unavailable".into());
    }
    run_tool(ffprobe, &["-version"])?;
    Ok(())
}

fn run_tool(executable: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new(executable)
        .args(args)
        .output()
        .map_err(|error| format!("run {}: {error}", executable.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} failed with status {}",
            executable.display(),
            output.status
        ));
    }
    Ok(format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn has_tool_entry(output: &str, entry: &str) -> bool {
    output
        .lines()
        .any(|line| line.split_whitespace().any(|item| item == entry))
}

fn validate_root_access(root: &Path) -> Result<(), String> {
    let probe = root.join(format!(".media-recorder-readiness-{}", std::process::id()));
    let renamed = probe.with_extension("renamed");
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
            .map_err(|error| format!("recording root create failed: {error}"))?;
        file.write_all(b"ready")
            .map_err(|error| format!("recording root write failed: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("recording root fsync failed: {error}"))?;
        drop(file);
        fs::rename(&probe, &renamed)
            .map_err(|error| format!("recording root rename failed: {error}"))?;
        let contents =
            fs::read(&renamed).map_err(|error| format!("recording root read failed: {error}"))?;
        (contents == b"ready")
            .then_some(())
            .ok_or_else(|| "recording root read verification failed".into())
    })();
    let _ = fs::remove_file(&probe);
    let _ = fs::remove_file(&renamed);
    result
}

fn required(name: &str) -> Result<String, String> {
    env::var(name).map_err(|_| format!("{name} is required"))
}

fn number(name: &str, default: u64, min: u64, max: u64) -> Result<u64, String> {
    let raw = env::var(name).unwrap_or_default();
    if raw.is_empty() {
        return Ok(default);
    }
    let value: u64 = raw.parse().map_err(|_| format!("{name} is not a number"))?;
    if value < min || value > max {
        return Err(format!("{name} is outside its allowed range"));
    }
    Ok(value)
}

fn executable(value: String) -> Result<PathBuf, String> {
    let candidate = PathBuf::from(&value);
    if candidate.components().count() > 1 {
        return is_executable(&candidate)
            .then_some(candidate)
            .ok_or_else(|| format!("executable does not exist or is not executable: {value}"));
    }
    env::var_os("PATH")
        .into_iter()
        .flat_map(|path| env::split_paths(&path).collect::<Vec<_>>())
        .map(|dir| dir.join(&value))
        .find(|path| is_executable(path))
        .ok_or_else(|| format!("executable not found in PATH: {value}"))
}

fn is_executable(path: &PathBuf) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        return path
            .metadata()
            .map(|meta| meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false);
    }
    #[cfg(not(unix))]
    true
}

#[cfg(test)]
mod tests {
    use super::{allowed_root, executable};
    use std::path::Path;

    #[test]
    fn executable_rejects_missing_explicit_path() {
        assert!(executable("/definitely/missing/ffmpeg".into()).is_err());
    }

    #[test]
    fn recording_root_allows_only_native_home_or_fixed_container_path() {
        assert!(allowed_root(Path::new("/recordings"), true));
        assert!(allowed_root(Path::new("/home/operator/recordings"), false));
        assert!(!allowed_root(Path::new("/recordings"), false));
        assert!(!allowed_root(Path::new("/home"), false));
        assert!(!allowed_root(Path::new("/tmp/recordings"), true));
        assert!(!allowed_root(Path::new("/recordings/nested"), true));
    }
}
