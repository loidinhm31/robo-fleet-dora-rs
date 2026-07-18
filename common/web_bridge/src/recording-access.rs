use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use rand::RngCore;
use robo_rover_lib::{RecordingClip, RecordingPlaybackTicketResult, RECORDING_PROTOCOL_VERSION};
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TICKET_TTL: Duration = Duration::from_secs(90);
const MAX_TICKETS: usize = 2_048;

#[derive(Clone)]
pub struct RecordingAccess {
    root: Option<PathBuf>,
    tickets: Arc<Mutex<HashMap<String, TicketEntry>>>,
}

#[derive(Clone, Debug)]
struct TicketEntry {
    relative_path: String,
    manifest_path: String,
    length: u64,
    identity: FileIdentity,
    manifest_identity: FileIdentity,
    expires_at: Instant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileIdentity {
    pub length: u64,
    pub modified_ns: u128,
    pub file_id: u128,
}

pub struct AuthorizedFile {
    pub file: tokio::fs::File,
    pub length: u64,
}

impl RecordingAccess {
    pub fn from_env() -> Self {
        let container_mode = std::env::var("RECORDING_CONTAINER_MODE").as_deref() == Ok("true");
        let root = std::env::var("RECORDING_ROOT")
            .ok()
            .and_then(|value| safe_root(&value, container_mode));
        if root.is_none() {
            tracing::warn!("recording playback disabled: RECORDING_ROOT is missing or unsafe");
        }
        Self {
            root,
            tickets: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn issue(
        &self,
        request_id: &str,
        clip: &RecordingClip,
    ) -> Result<RecordingPlaybackTicketResult, String> {
        let root = self.root.as_ref().ok_or("recording playback unavailable")?;
        let _ = safe_final_file(root, &clip.relative_path)?;
        let file = secure_open_file(root, &clip.relative_path)?;
        let metadata = file.metadata().map_err(|_| "recording file unavailable")?;
        if !metadata.is_file() {
            return Err("recording file is not regular".into());
        }
        let identity = file_identity(&metadata);
        if identity.length != clip.bytes_written {
            return Err("recording clip changed after catalog lookup".into());
        }
        let manifest = clip
            .relative_path
            .strip_suffix(".mp4")
            .map(|path| format!("{path}.manifest.json"))
            .ok_or("recording path is not an MP4")?;
        let _ = safe_recording_file(root, &manifest)?;
        let manifest_file = secure_open_file(root, &manifest)?;
        let manifest_metadata = manifest_file
            .metadata()
            .map_err(|_| "recording file unavailable")?;
        if !manifest_metadata.is_file() {
            return Err("recording manifest is not regular".into());
        }
        let manifest_identity = file_identity(&manifest_metadata);
        let mut token_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut token_bytes);
        let token = URL_SAFE_NO_PAD.encode(token_bytes);
        let url = format!("/recordings/playback/{token}");
        let expires_at = Instant::now() + TICKET_TTL;
        let expires_at_ms = now_ms().saturating_add(TICKET_TTL.as_millis() as u64);
        let entry = TicketEntry {
            relative_path: clip.relative_path.clone(),
            manifest_path: manifest,
            length: identity.length,
            identity,
            manifest_identity,
            expires_at,
        };
        let mut tickets = self
            .tickets
            .lock()
            .map_err(|_| "ticket store unavailable")?;
        tickets.retain(|_, entry| entry.expires_at > Instant::now());
        if tickets.len() >= MAX_TICKETS {
            return Err("playback ticket capacity reached".into());
        }
        tickets.insert(token.clone(), entry);
        tracing::info!(
            action = "playback_ticket_issue",
            request_id,
            clip_id = %clip.recording_id,
            outcome = "accepted",
            "issued playback ticket"
        );
        Ok(RecordingPlaybackTicketResult {
            protocol_version: RECORDING_PROTOCOL_VERSION,
            request_id: request_id.into(),
            recording_id: clip.recording_id.clone(),
            ticket: token,
            url,
            expires_at_ms,
        })
    }

    pub fn authorize(&self, token: &str) -> Result<AuthorizedFile, String> {
        let root = self.root.as_ref().ok_or("recording playback unavailable")?;
        let entry = {
            let mut tickets = self
                .tickets
                .lock()
                .map_err(|_| "ticket store unavailable")?;
            tickets.retain(|_, entry| entry.expires_at > Instant::now());
            tickets
                .get(token)
                .cloned()
                .ok_or("unknown or expired ticket")?
        };
        let _ = safe_final_file(root, &entry.relative_path)?;
        let file = secure_open_file(root, &entry.relative_path)?;
        let metadata = file.metadata().map_err(|_| "recording file unavailable")?;
        let _ = safe_recording_file(root, &entry.manifest_path)?;
        let manifest_file = secure_open_file(root, &entry.manifest_path)?;
        let manifest_metadata = manifest_file
            .metadata()
            .map_err(|_| "recording file unavailable")?;
        if !metadata.is_file() || !manifest_metadata.is_file() {
            return Err("recording file is not regular".into());
        }
        let manifest_identity = file_identity(&manifest_metadata);
        if file_identity(&metadata) != entry.identity
            || metadata.len() != entry.length
            || manifest_identity != entry.manifest_identity
        {
            return Err("recording file changed or is unavailable".into());
        }
        Ok(AuthorizedFile {
            file: tokio::fs::File::from_std(file),
            length: entry.length,
        })
    }
}

fn safe_root(value: &str, container_mode: bool) -> Option<PathBuf> {
    let root = PathBuf::from(value).canonicalize().ok()?;
    (allowed_root(&root, container_mode) && root.is_dir())
    .then_some(root)
}

fn allowed_root(root: &Path, container_mode: bool) -> bool {
    (container_mode && root == Path::new("/recordings"))
        || (!container_mode && root != Path::new("/home") && root.starts_with("/home"))
}

fn safe_final_file(root: &Path, relative: &str) -> Result<PathBuf, String> {
    if !relative.ends_with(".mp4") {
        return Err("invalid recording path".into());
    }
    safe_recording_file(root, relative)
}

fn safe_recording_file(root: &Path, relative: &str) -> Result<PathBuf, String> {
    if relative.is_empty()
        || relative.contains('\\')
        || relative.contains('\0')
        || relative.starts_with('/')
        || relative.split('/').any(|part| {
            part.is_empty()
                || part == "."
                || part == ".."
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        })
    {
        return Err("invalid recording path".into());
    }
    let mut current = root.to_path_buf();
    for component in relative.split('/') {
        current.push(component);
        let metadata = fs::symlink_metadata(&current).map_err(|_| "recording file unavailable")?;
        if metadata.file_type().is_symlink() {
            return Err("recording symlink is not allowed".into());
        }
    }
    let canonical = current
        .canonicalize()
        .map_err(|_| "recording file unavailable")?;
    canonical
        .starts_with(root)
        .then_some(canonical)
        .ok_or_else(|| "recording path escapes root".into())
}

#[cfg(unix)]
fn secure_open_file(root: &Path, relative: &str) -> Result<fs::File, String> {
    use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

    let root_file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(root)
        .map_err(|_| "recording root unavailable")?;
    let mut directory = unsafe { OwnedFd::from_raw_fd(root_file.into_raw_fd()) };
    let mut components = relative.split('/').peekable();
    while let Some(component) = components.next() {
        let name = CString::new(component).map_err(|_| "invalid recording path")?;
        let is_final = components.peek().is_none();
        let flags = if is_final {
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW
        } else {
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW
        };
        let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err("recording file unavailable".into());
        }
        if is_final {
            return Ok(unsafe { fs::File::from_raw_fd(fd) });
        }
        directory = unsafe { OwnedFd::from_raw_fd(fd) };
    }
    Err("invalid recording path".into())
}

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

#[cfg(not(unix))]
fn secure_open_file(root: &Path, relative: &str) -> Result<fs::File, String> {
    let path = safe_recording_file(root, relative)?;
    fs::OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|_| "recording file unavailable".into())
}

fn file_identity(metadata: &fs::Metadata) -> FileIdentity {
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    FileIdentity {
        length: metadata.len(),
        modified_ns,
        file_id: file_id(metadata),
    }
}

#[cfg(unix)]
fn file_id(metadata: &fs::Metadata) -> u128 {
    use std::os::unix::fs::MetadataExt;
    (u128::from(metadata.dev()) << 64) | u128::from(metadata.ino())
}

#[cfg(not(unix))]
fn file_id(_metadata: &fs::Metadata) -> u128 {
    0
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn safe_final_file_rejects_traversal_and_symlink() {
        let root = tempdir().unwrap();
        fs::write(root.path().join("clip.mp4"), b"clip").unwrap();
        assert!(safe_final_file(root.path(), "../clip.mp4").is_err());
        assert!(safe_final_file(root.path(), "/clip.mp4").is_err());
        assert!(safe_final_file(root.path(), "clip\\mp4").is_err());
    }

    #[test]
    fn recording_root_policy_keeps_container_path_container_only() {
        assert!(allowed_root(Path::new("/recordings"), true));
        assert!(!allowed_root(Path::new("/recordings"), false));
        assert!(allowed_root(Path::new("/home/operator/recordings"), false));
    }
}
