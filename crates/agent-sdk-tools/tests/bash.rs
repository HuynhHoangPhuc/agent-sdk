//! `bash` tool: success/failure, timeout, cancellation reaps the child.

use std::time::Duration;

use agent_sdk_core::Tool;
use agent_sdk_tools::{BashTool, SandboxRoot};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn tool() -> BashTool {
    let tmp = tempfile::tempdir().unwrap();
    let sb = SandboxRoot::at(tmp.path()).unwrap();
    // tmpdir intentionally dropped at end of test; bash cwd is canonicalized at spawn time.
    std::mem::forget(tmp);
    BashTool::new(sb)
}

#[tokio::test]
async fn echo_succeeds() {
    let result = tool()
        .execute(
            json!({"command": "echo hello"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(!result.is_error);
    assert!(
        result.content["stdout"].as_str().unwrap().contains("hello"),
        "unexpected: {:?}",
        result.content
    );
    assert_eq!(result.content["exit_code"], json!(0));
}

#[tokio::test]
async fn nonzero_exit_marks_is_error() {
    let result = tool()
        .execute(
            json!({"command": "exit 3"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(result.is_error);
    assert_eq!(result.content["exit_code"], json!(3));
}

#[tokio::test]
async fn timeout_kills_long_running_command() {
    let err = tool()
        .execute(
            json!({"command": "sleep 5", "timeout_secs": 1}),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("timeout"), "err: {err}");
}

#[tokio::test]
async fn cancellation_aborts_and_reaps_child() {
    let cancel = CancellationToken::new();
    let bash = tool();
    let cancel_clone = cancel.clone();
    let handle = tokio::spawn(async move {
        bash.execute(
            json!({"command": "sleep 30"}),
            cancel_clone,
        )
        .await
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    cancel.cancel();
    let result = tokio::time::timeout(Duration::from_secs(3), handle)
        .await
        .expect("must finish promptly after cancel")
        .expect("join")
        .expect_err("must be cancelled");
    assert!(result.to_string().to_lowercase().contains("cancel"));
}
