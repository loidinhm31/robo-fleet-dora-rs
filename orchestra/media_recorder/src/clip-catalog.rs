use crate::ffmpeg_session::validate_mp4;
use crate::path_resolver::PathResolver;
use robo_rover_lib::{RecordingAudioCodec, RecordingClip, RecordingVideoCodec};
use serde::{Deserialize, Serialize};
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
        let metadata = fs::metadata(path).map_err(|e| format!("manifest metadata: {e}"))?;
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
}
