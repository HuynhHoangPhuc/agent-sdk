//! Round-trip tests for `read`, `write`, `edit`.

use agent_sdk_core::Tool;
use agent_sdk_tools::{EditTool, ReadTool, SandboxRoot, WriteTool};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn sandbox() -> (tempfile::TempDir, SandboxRoot) {
    let tmp = tempfile::tempdir().unwrap();
    let sb = SandboxRoot::at(tmp.path()).unwrap();
    (tmp, sb)
}

#[tokio::test]
async fn write_then_read_roundtrip() {
    let (_tmp, sb) = sandbox();
    let writer = WriteTool::new(sb.clone());
    let reader = ReadTool::new(sb);

    writer
        .execute(
            json!({"path": "hello.txt", "content": "hi\nworld\n"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    let result = reader
        .execute(json!({"path": "hello.txt"}), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(result.content, json!("hi\nworld\n"));
    assert!(!result.is_error);
}

#[tokio::test]
async fn write_is_atomic_no_partial_tmp_left() {
    let (tmp, sb) = sandbox();
    let writer = WriteTool::new(sb);
    writer
        .execute(
            json!({"path": "out.txt", "content": "payload"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    // Only `out.txt` should remain (no `.out.txt.*.tmp`).
    let entries: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(entries, vec!["out.txt".to_string()]);
}

#[tokio::test]
async fn edit_errors_on_missing_match() {
    let (_tmp, sb) = sandbox();
    WriteTool::new(sb.clone())
        .execute(
            json!({"path": "f.txt", "content": "alpha beta"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let err = EditTool::new(sb)
        .execute(
            json!({"path": "f.txt", "old_string": "gamma", "new_string": "delta"}),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not found"), "err: {err}");
}

#[tokio::test]
async fn edit_errors_on_non_unique_match() {
    let (_tmp, sb) = sandbox();
    WriteTool::new(sb.clone())
        .execute(
            json!({"path": "f.txt", "content": "x x"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let err = EditTool::new(sb)
        .execute(
            json!({"path": "f.txt", "old_string": "x", "new_string": "y"}),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("matched"), "err: {err}");
}

#[tokio::test]
async fn edit_replace_all_overrides_uniqueness() {
    let (_tmp, sb) = sandbox();
    WriteTool::new(sb.clone())
        .execute(
            json!({"path": "f.txt", "content": "x x"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    EditTool::new(sb.clone())
        .execute(
            json!({"path": "f.txt", "old_string": "x", "new_string": "y", "replace_all": true}),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let r = ReadTool::new(sb)
        .execute(json!({"path": "f.txt"}), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r.content, json!("y y"));
}

#[tokio::test]
async fn write_outside_sandbox_is_rejected() {
    let (_tmp, sb) = sandbox();
    let err = WriteTool::new(sb)
        .execute(
            json!({"path": "../escape.txt", "content": ""}),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("outside"), "err: {err}");
}
