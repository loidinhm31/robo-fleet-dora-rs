use std::env;
use std::path::PathBuf;

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
        if root == PathBuf::from("/home") || !root.starts_with("/home") {
            return Err("RECORDING_ROOT must be a dedicated existing directory below /home".into());
        }
        if !root.is_dir() {
            return Err("RECORDING_ROOT must be a directory".into());
        }
        let ffmpeg_path = executable(env::var("FFMPEG_PATH").unwrap_or_else(|_| "ffmpeg".into()))?;
        let ffprobe_path =
            executable(env::var("FFPROBE_PATH").unwrap_or_else(|_| "ffprobe".into()))?;
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
    use super::executable;

    #[test]
    fn executable_rejects_missing_explicit_path() {
        assert!(executable("/definitely/missing/ffmpeg".into()).is_err());
    }
}
