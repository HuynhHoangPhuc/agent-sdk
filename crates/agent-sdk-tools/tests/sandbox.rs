//! Sandbox enforcement: parent escape, symlink escape, opt-out.

use std::path::PathBuf;

use agent_sdk_tools::SandboxRoot;

#[test]
fn rejects_dotdot_escape() {
    let tmp = tempfile::tempdir().unwrap();
    let sb = SandboxRoot::at(tmp.path()).unwrap();
    let err = sb.resolve("../outside.txt").unwrap_err();
    assert!(err.to_string().contains("outside"), "err: {err}");
}

#[test]
fn rejects_absolute_outside_root() {
    let tmp = tempfile::tempdir().unwrap();
    let sb = SandboxRoot::at(tmp.path()).unwrap();
    let elsewhere = std::env::temp_dir().join("agent-sdk-tools-out-of-sandbox.txt");
    let err = sb.resolve(&elsewhere).unwrap_err();
    assert!(err.to_string().contains("outside"), "err: {err}");
}

#[test]
fn accepts_relative_inside_root() {
    let tmp = tempfile::tempdir().unwrap();
    let sb = SandboxRoot::at(tmp.path()).unwrap();
    let resolved = sb.resolve("nested/new.txt").unwrap();
    let root_canon = std::fs::canonicalize(tmp.path()).unwrap();
    assert!(resolved.starts_with(&root_canon));
}

#[cfg(unix)]
#[test]
fn rejects_symlink_escape() {
    use std::os::unix::fs::symlink;

    let outside = tempfile::tempdir().unwrap();
    let inside = tempfile::tempdir().unwrap();
    let link = inside.path().join("escape");
    symlink(outside.path(), &link).unwrap();

    let sb = SandboxRoot::at(inside.path()).unwrap();
    // The symlink target lives outside the root, so canonicalize must reject it.
    let err = sb.resolve("escape/file.txt").unwrap_err();
    assert!(err.to_string().contains("outside"), "err: {err}");
}

#[test]
fn unrestricted_resolves_anything() {
    let sb = SandboxRoot::unrestricted();
    let p: PathBuf = std::env::temp_dir().join("unrestricted-anything.txt");
    let resolved = sb.resolve(&p).unwrap();
    assert!(resolved.is_absolute());
}
