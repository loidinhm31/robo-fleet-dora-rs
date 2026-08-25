use crate::ffmpeg_session::validate_mp4;
use crate::path_resolver::PathResolver;
use robo_rover_lib::{RecordingAudioCodec, RecordingClip, RecordingVideoCodec};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAX_MANIFEST_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingManifest {
    pub clip: RecordingClip,
    pub dropped_video: u64,
    pub audio_gaps: u64,
    pub silence_samples: u64,
    pub timestamp_regressions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogIssue {
    pub path: PathBuf,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ClipCatalog {
    resolver: PathResolver,
    ffprobe: PathBuf,
}

impl ClipCatalog {
    pub fn new(resolver: PathResolver, ffprobe: PathBuf) -> Self {
        Self { resolver, ffprobe }
    }

    pub fn scan(&self) -> (Vec<RecordingClip>, Vec<CatalogIssue>) {
        let mut clips = Vec::new();
        let mut issues = Vec::new();
        self.recover_deletions(self.resolver.root(), &mut issues);
        self.scan_partials(&mut issues);
        self.scan_dir(self.resolver.root(), &mut clips, &mut issues);
        clips.sort_by(|a, b| b.started_at_ms.cmp(&a.started_at_ms));
        (clips, issues)
    }

    pub fn list(
        &self,
        entity_id: Option<&str>,
        directory: Option<&str>,
    ) -> Result<Vec<RecordingClip>, String> {
        if let Some(directory) = directory {
            self.resolver.existing_directory(directory)?;
        }
        Ok(self
            .scan()
            .0
            .into_iter()
            .filter(|clip| {
                entity_id.is_none_or(|entity| entity == clip.entity_id)
                    && directory
                        .is_none_or(|dir| clip.relative_path.starts_with(&format!("{dir}/")))
            })
            .collect())
    }

    pub fn lookup(&self, recording_id: &str) -> Result<RecordingClip, String> {
        robo_rover_lib::validate_uuid("recording_id", recording_id)?;
        self.scan()
            .0
            .into_iter()
            .find(|clip| clip.recording_id == recording_id)
            .ok_or_else(|| "recording clip not found".into())
    }

    /// Permanently removes a finalized clip and its manifest as a pair.
    ///
    /// Paths are derived from the validated manifest only; callers never provide
    /// filesystem paths. On Unix, the containing directory is opened component
    /// by component without following symlinks, and mutation uses that pinned
    /// directory descriptor.
    pub fn delete(&self, recording_id: &str) -> Result<(), String> {
        robo_rover_lib::validate_uuid("recording_id", recording_id)?;
        let manifest_path = self.find_manifest_path(recording_id)?;
        let clip = self.read_manifest(&manifest_path)?;
        if clip.recording_id != recording_id {
            return Err("recording identity mismatch".into());
        }
        let media_path = manifest_path.with_file_name(format!("{recording_id}.mp4"));
        self.revalidate_delete_file(&manifest_path, false)?;
        self.revalidate_delete_file(&media_path, true)?;

        let suffix = uuid::Uuid::new_v4();
        let hidden_manifest =
            manifest_path.with_file_name(format!(".{recording_id}.{suffix}.manifest.delete"));
        let hidden_media =
            media_path.with_file_name(format!(".{recording_id}.{suffix}.mp4.delete"));
        #[cfg(unix)]
        {
            secure_delete_pair(
                self.resolver.root(),
                &manifest_path,
                &media_path,
                &hidden_manifest,
                &hidden_media,
            )
        }
        #[cfg(not(unix))]
        {
            fs::rename(&manifest_path, &hidden_manifest)
                .map_err(|e| format!("hide recording manifest: {e}"))?;
            if let Err(error) = fs::rename(&media_path, &hidden_media) {
                let _ = fs::rename(&hidden_manifest, &manifest_path);
                return Err(format!("hide recording media: {error}"));
            }
            sync_dir(manifest_path.parent().unwrap_or(self.resolver.root()))?;
            let manifest_result = fs::remove_file(&hidden_manifest);
            let media_result = fs::remove_file(&hidden_media);
            sync_dir(manifest_path.parent().unwrap_or(self.resolver.root()))?;
            delete_results(manifest_result, media_result)
        }
    }

    fn recover_deletions(&self, directory: &Path, issues: &mut Vec<CatalogIssue>) {
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) => {
                issues.push(CatalogIssue {
                    path: directory.into(),
                    detail: format!("scan deletion recovery: {error}"),
                });
                return;
            }
        };
        let mut tombstones: HashMap<String, Vec<PathBuf>> = HashMap::new();
        let mut child_directories = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                if path != self.resolver.partial() {
                    child_directories.push(path);
                }
                continue;
            }
            if let Some(recording_id) = delete_tombstone_recording_id(&path) {
                tombstones.entry(recording_id).or_default().push(path);
            }
        }
        for child in child_directories {
            self.recover_deletions(&child, issues);
        }
        for (recording_id, paths) in tombstones {
            #[cfg(unix)]
            if let Err(error) = secure_recover_delete_transaction(
                self.resolver.root(),
                directory,
                &recording_id,
                &paths,
            ) {
                issues.push(CatalogIssue {
                    path: directory.into(),
                    detail: error,
                });
            }
            #[cfg(not(unix))]
            {
                let mut targets: HashSet<PathBuf> = paths.into_iter().collect();
                targets.insert(directory.join(format!("{recording_id}.mp4")));
                targets.insert(directory.join(format!("{recording_id}.manifest.json")));
                let mut failed = false;
                for path in targets {
                    match fs::symlink_metadata(&path) {
                        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
                            failed = true;
                            issues.push(CatalogIssue {
                                path,
                                detail: "delete recovery target is a directory".into(),
                            });
                        }
                        Ok(_) => {
                            if let Err(error) = fs::remove_file(&path) {
                                failed = true;
                                issues.push(CatalogIssue {
                                    path,
                                    detail: format!("complete interrupted deletion: {error}"),
                                });
                            }
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => {
                            failed = true;
                            issues.push(CatalogIssue {
                                path,
                                detail: format!("inspect interrupted deletion: {error}"),
                            });
                        }
                    }
                }
                if !failed {
                    if let Err(error) = sync_dir(directory) {
                        issues.push(CatalogIssue {
                            path: directory.into(),
                            detail: error,
                        });
                    }
                }
            }
        }
    }

    fn find_manifest_path(&self, recording_id: &str) -> Result<PathBuf, String> {
        let mut manifests = Vec::new();
        self.find_manifest_paths(self.resolver.root(), recording_id, &mut manifests);
        match manifests.len() {
            0 => {
                let partial = self
                    .resolver
                    .partial()
                    .join(format!("{recording_id}.mp4.partial"));
                if fs::symlink_metadata(partial).is_ok() {
                    Err("recording is still partial".into())
                } else {
                    Err("recording clip not found".into())
                }
            }
            1 => Ok(manifests.remove(0)),
            _ => Err("recording identity is ambiguous".into()),
        }
    }

    fn find_manifest_paths(&self, directory: &Path, recording_id: &str, paths: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(directory) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.file_type().is_symlink() || path == self.resolver.partial() {
                continue;
            }
            if metadata.is_dir() {
                self.find_manifest_paths(&path, recording_id, paths);
            } else if path.file_name().is_some_and(|name| {
                name.to_string_lossy() == format!("{recording_id}.manifest.json")
            }) {
                paths.push(path);
            }
        }
    }

    fn revalidate_delete_file(&self, path: &Path, media: bool) -> Result<(), String> {
        self.resolver.contained(path)?;
        let metadata =
            fs::symlink_metadata(path).map_err(|_| "recording pair is incomplete".to_string())?;
        if !metadata.file_type().is_file() {
            return Err(if media {
                "recording MP4 is not a regular file"
            } else {
                "recording manifest is not a regular file"
            }
            .into());
        }
        if media && path.extension().is_none_or(|extension| extension != "mp4") {
            return Err("recording path is not an MP4".into());
        }
        Ok(())
    }

    pub fn publish(
        &self,
        partial: &Path,
        directory: &Path,
        manifest: &RecordingManifest,
    ) -> Result<(), String> {
        self.resolver.contained(partial)?;
        self.resolver.contained(directory)?;
        validate_mp4(&self.ffprobe, partial)?;
        let final_path = directory.join(format!("{}.mp4", manifest.clip.recording_id));
        if final_path.exists() {
            return Err("recording output collision".into());
        }
        let expected_relative = directory
            .strip_prefix(self.resolver.root())
            .map(|path| {
                path.join(format!("{}.mp4", manifest.clip.recording_id))
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/")
            })
            .map_err(|_| "output directory is outside recording root".to_string())?;
        if manifest.clip.relative_path != expected_relative {
            return Err("manifest path does not match output directory".into());
        }
        let bytes = serde_json::to_vec(manifest).map_err(|e| format!("serialize manifest: {e}"))?;
        if bytes.len() > MAX_MANIFEST_BYTES as usize {
            return Err("manifest exceeds size limit".into());
        }
        File::open(partial)
            .map_err(|e| format!("open partial: {e}"))?
            .sync_all()
            .map_err(|e| format!("sync partial: {e}"))?;
        fs::rename(partial, &final_path).map_err(|e| format!("publish MP4: {e}"))?;
        self.resolver.contained(&final_path)?;
        let manifest_path = directory.join(format!("{}.manifest.json", manifest.clip.recording_id));
        let temp_path = directory.join(format!(
            "{}.manifest.json.tmp-{}",
            manifest.clip.recording_id,
            uuid::Uuid::new_v4()
        ));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
                .map_err(|e| format!("create manifest temp: {e}"))?;
            file.write_all(&bytes)
                .map_err(|e| format!("write manifest: {e}"))?;
            file.sync_all().map_err(|e| format!("sync manifest: {e}"))?;
            fs::hard_link(&temp_path, &manifest_path)
                .map_err(|e| format!("publish manifest without overwrite: {e}"))?;
            let _ = fs::remove_file(&temp_path);
            Ok(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_file(&temp_path);
            let _ = fs::remove_file(&final_path);
            return Err(error);
        }
        sync_dir(directory)?;
        Ok(())
    }

    fn scan_dir(
        &self,
        directory: &Path,
        clips: &mut Vec<RecordingClip>,
        issues: &mut Vec<CatalogIssue>,
    ) {
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) => {
                issues.push(CatalogIssue {
                    path: directory.into(),
                    detail: error.to_string(),
                });
                return;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let metadata = match fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) => {
                    issues.push(CatalogIssue {
                        path,
                        detail: error.to_string(),
                    });
                    continue;
                }
            };
            if path == self.resolver.partial()
                || path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("."))
            {
                continue;
            }
            if metadata.file_type().is_symlink() {
                issues.push(CatalogIssue {
                    path,
                    detail: "symlink entry ignored".into(),
                });
                continue;
            }
            if metadata.is_dir() {
                self.scan_dir(&path, clips, issues);
                continue;
            }
            if !path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with(".manifest.json"))
            {
                continue;
            }
            match self.read_manifest(&path) {
                Ok(clip) => clips.push(clip),
                Err(detail) => issues.push(CatalogIssue { path, detail }),
            }
        }
    }

    fn scan_partials(&self, issues: &mut Vec<CatalogIssue>) {
        let entries = match fs::read_dir(self.resolver.partial()) {
            Ok(entries) => entries,
            Err(error) => {
                issues.push(CatalogIssue {
                    path: self.resolver.partial().into(),
                    detail: error.to_string(),
                });
                return;
            }
        };
        for entry in entries.flatten() {
            if entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "partial")
            {
                issues.push(CatalogIssue {
                    path: entry.path(),
                    detail: "stale partial media is not publishable".into(),
                });
            }
        }
    }

    fn read_manifest(&self, path: &Path) -> Result<RecordingClip, String> {
        self.resolver.contained(path)?;
        let metadata = fs::symlink_metadata(path).map_err(|e| format!("manifest metadata: {e}"))?;
        if !metadata.file_type().is_file() {
            return Err("manifest is not a regular file".into());
        }
        if metadata.len() > MAX_MANIFEST_BYTES {
            return Err("manifest exceeds size limit".into());
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        File::open(path)
            .map_err(|e| format!("open manifest: {e}"))?
            .read_to_end(&mut bytes)
            .map_err(|e| format!("read manifest: {e}"))?;
        let manifest: RecordingManifest =
            serde_json::from_slice(&bytes).map_err(|e| format!("invalid manifest: {e}"))?;
        manifest.clip.validate()?;
        if manifest.clip.video_codec != RecordingVideoCodec::H264
            || manifest.clip.audio_codec != RecordingAudioCodec::Aac
        {
            return Err("manifest contains unsupported codecs".into());
        }
        let media_path = path.with_file_name(format!("{}.mp4", manifest.clip.recording_id));
        self.resolver.contained(&media_path)?;
        if !media_path.is_file() {
            return Err("manifest has no finalized MP4".into());
        }
        let expected_relative = self.resolver.relative(&media_path)?;
        if manifest.clip.relative_path != expected_relative {
            return Err("manifest path does not match its containing directory".into());
        }
        validate_mp4(&self.ffprobe, &media_path)?;
        Ok(manifest.clip)
    }
}

fn sync_dir(path: &Path) -> Result<(), String> {
    File::open(path)
        .map_err(|e| format!("open output directory: {e}"))?
        .sync_all()
        .map_err(|e| format!("sync output directory: {e}"))
}

fn delete_tombstone_recording_id(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let parts: Vec<_> = name.split('.').collect();
    if parts.len() != 5
        || !parts[0].is_empty()
        || parts[4] != "delete"
        || !matches!(parts[3], "manifest" | "mp4")
    {
        return None;
    }
    robo_rover_lib::validate_uuid("recording_id", parts[1]).ok()?;
    robo_rover_lib::validate_uuid("delete transaction", parts[2]).ok()?;
    Some(parts[1].to_owned())
}

fn delete_results(manifest: std::io::Result<()>, media: std::io::Result<()>) -> Result<(), String> {
    match (manifest, media) {
        (Ok(()), Ok(())) => Ok(()),
        (manifest, media) => Err(format!(
            "delete recording pair failed: manifest={:?}, media={:?}",
            manifest.err().map(|e| e.kind()),
            media.err().map(|e| e.kind())
        )),
    }
}

#[cfg(unix)]
fn secure_delete_pair(
    root: &Path,
    manifest_path: &Path,
    media_path: &Path,
    hidden_manifest: &Path,
    hidden_media: &Path,
) -> Result<(), String> {
    use std::os::fd::AsRawFd;

    let parent = manifest_path
        .parent()
        .ok_or("recording manifest has no parent")?;
    if media_path.parent() != Some(parent)
        || hidden_manifest.parent() != Some(parent)
        || hidden_media.parent() != Some(parent)
    {
        return Err("recording delete pair does not share a directory".into());
    }
    let directory = open_directory_beneath_root(root, parent)?;
    let manifest_name = c_file_name(manifest_path)?;
    let media_name = c_file_name(media_path)?;
    let hidden_manifest_name = c_file_name(hidden_manifest)?;
    let hidden_media_name = c_file_name(hidden_media)?;
    regular_file_at(directory.as_raw_fd(), &manifest_name, "manifest")?;
    regular_file_at(directory.as_raw_fd(), &media_name, "media")?;
    if unsafe {
        libc::renameat(
            directory.as_raw_fd(),
            manifest_name.as_ptr(),
            directory.as_raw_fd(),
            hidden_manifest_name.as_ptr(),
        )
    } < 0
    {
        return Err(format!(
            "hide recording manifest: {}",
            std::io::Error::last_os_error()
        ));
    }
    if unsafe {
        libc::renameat(
            directory.as_raw_fd(),
            media_name.as_ptr(),
            directory.as_raw_fd(),
            hidden_media_name.as_ptr(),
        )
    } < 0
    {
        let error = std::io::Error::last_os_error();
        unsafe {
            libc::renameat(
                directory.as_raw_fd(),
                hidden_manifest_name.as_ptr(),
                directory.as_raw_fd(),
                manifest_name.as_ptr(),
            )
        };
        return Err(format!("hide recording media: {error}"));
    }
    directory
        .sync_all()
        .map_err(|error| format!("sync hidden recording pair: {error}"))?;
    let manifest_result = unlink_at(directory.as_raw_fd(), &hidden_manifest_name);
    let media_result = unlink_at(directory.as_raw_fd(), &hidden_media_name);
    directory
        .sync_all()
        .map_err(|error| format!("sync deleted recording pair: {error}"))?;
    delete_results(manifest_result, media_result)
}

#[cfg(unix)]
fn secure_recover_delete_transaction(
    root: &Path,
    directory_path: &Path,
    recording_id: &str,
    tombstones: &[PathBuf],
) -> Result<(), String> {
    use std::ffi::CString;
    use std::os::fd::AsRawFd;

    robo_rover_lib::validate_uuid("recording_id", recording_id)?;
    let directory = open_directory_beneath_root(root, directory_path)?;
    let mut names = HashSet::from([
        format!("{recording_id}.mp4"),
        format!("{recording_id}.manifest.json"),
    ]);
    for tombstone in tombstones {
        if tombstone.parent() != Some(directory_path)
            || delete_tombstone_recording_id(tombstone).as_deref() != Some(recording_id)
        {
            return Err("invalid delete recovery tombstone".into());
        }
        names.insert(
            tombstone
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("invalid delete recovery file name")?
                .to_owned(),
        );
    }
    for name in names {
        let name = CString::new(name).map_err(|_| "delete recovery file name contains NUL")?;
        match unlink_at(directory.as_raw_fd(), &name) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("complete interrupted deletion: {error}")),
        }
    }
    directory
        .sync_all()
        .map_err(|error| format!("sync recovered deletion: {error}"))
}

#[cfg(unix)]
fn open_directory_beneath_root(root: &Path, directory_path: &Path) -> Result<File, String> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    let relative = directory_path
        .strip_prefix(root)
        .map_err(|_| "recording delete directory escapes root")?;
    let root_name = CString::new(root.as_os_str().as_bytes())
        .map_err(|_| "recording root contains a NUL byte")?;
    let root_fd = unsafe {
        libc::open(
            root_name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if root_fd < 0 {
        return Err(format!(
            "open recording root securely: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut directory = unsafe { File::from_raw_fd(root_fd) };
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            return Err("invalid recording delete directory component".into());
        };
        let name = CString::new(component.as_bytes())
            .map_err(|_| "recording directory contains a NUL byte")?;
        let child_fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if child_fd < 0 {
            return Err(format!(
                "open recording delete directory securely: {}",
                std::io::Error::last_os_error()
            ));
        }
        directory = unsafe { File::from_raw_fd(child_fd) };
    }
    Ok(directory)
}

#[cfg(unix)]
fn c_file_name(path: &Path) -> Result<std::ffi::CString, String> {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(
        path.file_name()
            .ok_or("recording path has no file name")?
            .as_bytes(),
    )
    .map_err(|_| "recording file name contains a NUL byte".into())
}

#[cfg(unix)]
fn regular_file_at(
    directory_fd: std::os::fd::RawFd,
    name: &std::ffi::CString,
    label: &str,
) -> Result<(), String> {
    let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
    if unsafe {
        libc::fstatat(
            directory_fd,
            name.as_ptr(),
            metadata.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } < 0
    {
        return Err(format!(
            "inspect recording {label}: {}",
            std::io::Error::last_os_error()
        ));
    }
    let metadata = unsafe { metadata.assume_init() };
    if metadata.st_mode & libc::S_IFMT != libc::S_IFREG {
        return Err(format!("recording {label} is not a regular file"));
    }
    Ok(())
}

#[cfg(unix)]
fn unlink_at(directory_fd: std::os::fd::RawFd, name: &std::ffi::CString) -> std::io::Result<()> {
    if unsafe { libc::unlinkat(directory_fd, name.as_ptr(), 0) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn startup_scan_does_not_expose_partial_or_invalid_manifest() {
        let root = tempdir().unwrap();
        let resolver = PathResolver::new(root.path()).unwrap();
        let partial = resolver.partial().join("stale.mp4.partial");
        File::create(partial).unwrap();
        let dir = resolver.directory("rover-a").unwrap();
        fs::write(dir.join("bad.manifest.json"), b"{}").unwrap();
        let catalog = ClipCatalog::new(resolver, PathBuf::from("ffprobe"));
        let (clips, issues) = catalog.scan();
        assert!(clips.is_empty());
        assert_eq!(issues.len(), 2);
    }

    #[test]
    fn startup_scan_completes_an_interrupted_delete_transaction() {
        let root = tempdir().unwrap();
        let resolver = PathResolver::new(root.path()).unwrap();
        let directory = resolver.directory("rover-a").unwrap();
        let recording_id = uuid::Uuid::new_v4().to_string();
        let transaction_id = uuid::Uuid::new_v4();
        let media = directory.join(format!("{recording_id}.mp4"));
        let manifest = directory.join(format!("{recording_id}.manifest.json"));
        let tombstone = directory.join(format!(".{recording_id}.{transaction_id}.manifest.delete"));
        fs::write(&media, b"orphaned media").unwrap();
        fs::write(&manifest, b"orphaned manifest").unwrap();
        fs::write(&tombstone, b"hidden manifest").unwrap();

        let catalog = ClipCatalog::new(resolver, PathBuf::from("ffprobe"));
        let (clips, issues) = catalog.scan();

        assert!(clips.is_empty());
        assert!(issues.is_empty(), "unexpected recovery issues: {issues:?}");
        assert!(!media.exists());
        assert!(!manifest.exists());
        assert!(!tombstone.exists());
    }

    #[cfg(unix)]
    #[test]
    fn delete_recovery_rejects_a_symlinked_parent_directory() {
        let root = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let resolver = PathResolver::new(root.path()).unwrap();
        let recording_id = uuid::Uuid::new_v4().to_string();
        let transaction_id = uuid::Uuid::new_v4();
        let outside_media = outside.path().join(format!("{recording_id}.mp4"));
        let outside_tombstone = outside
            .path()
            .join(format!(".{recording_id}.{transaction_id}.manifest.delete"));
        fs::write(&outside_media, b"must remain").unwrap();
        fs::write(&outside_tombstone, b"must remain").unwrap();
        let linked_directory = resolver.root().join("escape");
        std::os::unix::fs::symlink(outside.path(), &linked_directory).unwrap();
        let discovered_tombstone =
            linked_directory.join(outside_tombstone.file_name().expect("tombstone file name"));

        assert!(secure_recover_delete_transaction(
            resolver.root(),
            &linked_directory,
            &recording_id,
            &[discovered_tombstone],
        )
        .is_err());
        assert!(outside_media.exists());
        assert!(outside_tombstone.exists());
    }
}
