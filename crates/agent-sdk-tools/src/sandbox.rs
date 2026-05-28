//! Path sandbox for file-system tools.
//!
//! A [`SandboxRoot`] either pins a canonicalized root directory (the default —
//! all path inputs to `read`/`write`/`edit`/`grep`/`glob`/`ls`/`bash.cwd` are
//! validated to stay inside the root) or is `Unrestricted` (`.no_sandbox()`).
//!
//! Sandbox validation:
//! 1. Relative inputs are joined to the root.
//! 2. The longest existing prefix of the result is canonicalized; the
//!    canonical path must `starts_with` the canonicalized root (catches
//!    symlink escapes).
//! 3. The remaining (non-existent) tail must not contain `..` segments
//!    (logical escape rejection for paths a caller is about to create).

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

/// Failures from path sandbox validation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SandboxError {
    /// Resolved path escaped the sandbox root.
    #[error("path '{}' is outside the sandbox root", path.display())]
    OutsideRoot {
        /// The path that was rejected.
        path: PathBuf,
    },
    /// I/O error while canonicalizing the root or a path.
    #[error("sandbox io error on '{}': {source}", path.display())]
    Io {
        /// Path being processed when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Path-sandbox root.
///
/// `Clone` is cheap (internally `Arc`-shared). Construct via
/// [`SandboxRoot::cwd`], [`SandboxRoot::at`], or [`SandboxRoot::unrestricted`].
#[derive(Debug, Clone)]
pub struct SandboxRoot {
    inner: Arc<Inner>,
}

#[derive(Debug)]
enum Inner {
    Rooted(PathBuf),
    Unrestricted,
}

impl SandboxRoot {
    /// Root the sandbox at the current working directory (canonicalized).
    pub fn cwd() -> Result<Self, SandboxError> {
        let cwd = std::env::current_dir().map_err(|source| SandboxError::Io {
            path: PathBuf::from("."),
            source,
        })?;
        Self::at(cwd)
    }

    /// Root the sandbox at `path` (canonicalized; must exist).
    pub fn at(path: impl AsRef<Path>) -> Result<Self, SandboxError> {
        let path = path.as_ref();
        let canonical = std::fs::canonicalize(path).map_err(|source| SandboxError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            inner: Arc::new(Inner::Rooted(canonical)),
        })
    }

    /// Opt out of sandboxing — power-user escape hatch.
    pub fn unrestricted() -> Self {
        Self {
            inner: Arc::new(Inner::Unrestricted),
        }
    }

    /// `true` when sandboxing is active.
    pub fn is_enforced(&self) -> bool {
        matches!(&*self.inner, Inner::Rooted(_))
    }

    /// The canonical root path when enforced; `None` for unrestricted.
    pub fn root_path(&self) -> Option<&Path> {
        match &*self.inner {
            Inner::Rooted(p) => Some(p.as_path()),
            Inner::Unrestricted => None,
        }
    }

    /// Resolve `input` to an absolute path, validating it stays inside the
    /// root when enforced. Works for paths that do not yet exist (writes):
    /// canonicalizes the longest existing ancestor and rejects `..` in the
    /// tail.
    pub fn resolve(&self, input: impl AsRef<Path>) -> Result<PathBuf, SandboxError> {
        let input = input.as_ref();
        match &*self.inner {
            Inner::Unrestricted => {
                let joined = if input.is_absolute() {
                    input.to_path_buf()
                } else {
                    std::env::current_dir()
                        .map_err(|source| SandboxError::Io {
                            path: input.to_path_buf(),
                            source,
                        })?
                        .join(input)
                };
                Ok(joined)
            }
            Inner::Rooted(root) => {
                let joined = if input.is_absolute() {
                    input.to_path_buf()
                } else {
                    root.join(input)
                };

                let (existing, tail) = split_longest_existing(&joined);
                let canonical_existing =
                    std::fs::canonicalize(&existing).map_err(|source| SandboxError::Io {
                        path: existing.clone(),
                        source,
                    })?;

                if !canonical_existing.starts_with(root) {
                    return Err(SandboxError::OutsideRoot {
                        path: joined.clone(),
                    });
                }

                // Reject `..` in the non-existent tail (logical escape).
                let mut depth: i32 = 0;
                for comp in tail.components() {
                    match comp {
                        Component::ParentDir => {
                            depth -= 1;
                            if depth < 0 {
                                return Err(SandboxError::OutsideRoot {
                                    path: joined.clone(),
                                });
                            }
                        }
                        Component::Normal(_) => depth += 1,
                        Component::CurDir => {}
                        Component::RootDir | Component::Prefix(_) => {
                            return Err(SandboxError::OutsideRoot {
                                path: joined.clone(),
                            });
                        }
                    }
                }

                if tail.as_os_str().is_empty() {
                    Ok(canonical_existing)
                } else {
                    Ok(canonical_existing.join(&tail))
                }
            }
        }
    }
}

/// Walk up `path` until an existing ancestor is found. Returns
/// `(existing_prefix, remaining_tail)`. The split is purely by
/// `Path::exists`; symlinks are followed (canonicalize handles escape).
fn split_longest_existing(path: &Path) -> (PathBuf, PathBuf) {
    let mut existing = path.to_path_buf();
    let mut tail = PathBuf::new();
    while !existing.exists() {
        let Some(file_name) = existing.file_name().map(|s| s.to_owned()) else {
            break;
        };
        let prefix = PathBuf::from(&file_name);
        tail = if tail.as_os_str().is_empty() {
            prefix
        } else {
            prefix.join(&tail)
        };
        if !existing.pop() {
            break;
        }
    }
    (existing, tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_parent_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let sb = SandboxRoot::at(tmp.path()).unwrap();
        assert!(sb.resolve("../escape.txt").is_err());
    }

    #[test]
    fn accepts_inside_path() {
        let tmp = tempfile::tempdir().unwrap();
        let sb = SandboxRoot::at(tmp.path()).unwrap();
        let resolved = sb.resolve("nested/new.txt").unwrap();
        assert!(resolved.starts_with(std::fs::canonicalize(tmp.path()).unwrap()));
    }

    #[test]
    fn unrestricted_passes_anything() {
        let sb = SandboxRoot::unrestricted();
        assert!(sb.resolve("/tmp/anywhere").is_ok());
    }
}
