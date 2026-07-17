use robo_rover_lib::validate_relative_directory;
use std::fs::{self, Metadata};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct PathResolver {
    root: PathBuf,
    partial: PathBuf,
}

impl PathResolver {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|e| format!("root: {e}"))?;
        if !root.is_dir() {
            return Err("recording root is not a directory".into());
        }
        let partial = root.join(".partial");
        ensure_directory(&partial)?;
        restrict(&partial)?;
        Ok(Self { root, partial })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn partial(&self) -> &Path {
        &self.partial
    }

    pub fn directory(&self, relative: &str) -> Result<PathBuf, String> {
        validate_relative_directory(relative)?;
        #[cfg(unix)]
        secure_directory(&self.root, relative)?;
        let mut candidate = self.root.clone();
        for component in relative.split('/') {
            candidate.push(component);
            #[cfg(not(unix))]
            {
                ensure_directory(&candidate)?;
                self.contained(&candidate)?;
                restrict(&candidate)?;
            }
        }
        let canonical = candidate
            .canonicalize()
            .map_err(|e| format!("recording directory: {e}"))?;
        self.contained(&canonical)?;
        Ok(canonical)
    }

    pub fn existing_directory(&self, relative: &str) -> Result<PathBuf, String> {
        robo_rover_lib::validate_relative_directory(relative)?;
        let candidate = self
            .root
            .join(relative)
            .canonicalize()
            .map_err(|e| format!("recording directory lookup: {e}"))?;
        self.contained(&candidate)?;
        candidate
            .is_dir()
            .then_some(candidate)
            .ok_or_else(|| "recording path is not a directory".into())
    }

    pub fn contained(&self, path: &Path) -> Result<(), String> {
        let canonical = path
            .canonicalize()
            .map_err(|e| format!("path containment: {e}"))?;
        canonical
            .starts_with(&self.root)
            .then_some(())
            .ok_or_else(|| format!("path escapes recording root: {}", canonical.display()))
    }

    pub fn relative(&self, path: &Path) -> Result<String, String> {
        self.contained(path)?;
        let relative = path
            .canonicalize()
            .map_err(|e| format!("relative path: {e}"))?
            .strip_prefix(&self.root)
            .map_err(|_| "path is outside recording root".to_string())?
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        if relative.is_empty() {
            return Err("root is not a recording file".into());
        }
        Ok(relative)
    }
}

fn ensure_directory(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => validate_directory_metadata(path, metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(path).map_err(|e| format!("create {}: {e}", path.display()))?;
            validate_directory_metadata(
                path,
                fs::symlink_metadata(path)
                    .map_err(|e| format!("recheck {}: {e}", path.display()))?,
            )
        }
        Err(error) => Err(format!("inspect {}: {error}", path.display())),
    }
}

#[cfg(unix)]
fn secure_directory(root: &Path, relative: &str) -> Result<(), String> {
    use std::ffi::CString;
    use std::fs::File;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    let root_name = CString::new(root.as_os_str().as_bytes())
        .map_err(|_| "recording root contains a NUL byte".to_string())?;
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
    let mut current = unsafe { File::from_raw_fd(root_fd) };
    for component in relative.split('/') {
        let name = CString::new(component.as_bytes())
            .map_err(|_| "recording directory contains a NUL byte".to_string())?;
        let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW;
        let mut child_fd = unsafe { libc::openat(current.as_raw_fd(), name.as_ptr(), flags) };
        if child_fd < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::NotFound {
                return Err(format!("open recording component securely: {error}"));
            }
            if unsafe { libc::mkdirat(current.as_raw_fd(), name.as_ptr(), 0o700) } < 0 {
                let mkdir_error = std::io::Error::last_os_error();
                if mkdir_error.kind() != std::io::ErrorKind::AlreadyExists {
                    return Err(format!("create recording component: {mkdir_error}"));
                }
            }
            child_fd = unsafe { libc::openat(current.as_raw_fd(), name.as_ptr(), flags) };
        }
        if child_fd < 0 {
            return Err(format!(
                "reopen recording component securely: {}",
                std::io::Error::last_os_error()
            ));
        }
        if unsafe { libc::fchmod(child_fd, 0o700) } < 0 {
            let error = std::io::Error::last_os_error();
            unsafe { libc::close(child_fd) };
            return Err(format!("restrict recording component: {error}"));
        }
        current = unsafe { File::from_raw_fd(child_fd) };
    }
    Ok(())
}

fn validate_directory_metadata(path: &Path, metadata: Metadata) -> Result<(), String> {
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "symlink component is not allowed: {}",
            path.display()
        ));
    }
    metadata
        .is_dir()
        .then_some(())
        .ok_or_else(|| format!("recording component is not a directory: {}", path.display()))
}

fn restrict(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|e| format!("restrict {}: {e}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::PathResolver;
    use tempfile::tempdir;

    #[test]
    fn rejects_absolute_parent_and_dot_paths() {
        let root = tempdir().unwrap();
        let resolver = PathResolver::new(root.path()).unwrap();
        for value in ["/tmp", "../escape", "a/../b", "a//b", "a/./b", ""] {
            assert!(resolver.directory(value).is_err(), "accepted {value}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        let root = tempdir().unwrap();
        let outside = tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("escape")).unwrap();
        let resolver = PathResolver::new(root.path()).unwrap();
        assert!(resolver.directory("escape").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_nested_creation_through_symlink() {
        let root = tempdir().unwrap();
        let outside = tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("escape")).unwrap();
        let resolver = PathResolver::new(root.path()).unwrap();
        assert!(resolver.directory("escape/nested").is_err());
        assert!(!outside.path().join("nested").exists());
    }
}
