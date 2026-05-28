//! `ExternalCommandHook` integration tests.
//!
//! These tests spawn `bash`/`sh` and so are gated to unix targets. Each test
//! writes its script body inline via `bash -c` to avoid touching the
//! filesystem (no temp files, no cleanup) so the tests stay deterministic.

#![cfg(unix)]

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_sdk_core::{
    Agent, AgentError, ExternalCommandHook, FnTool, Hook, HookCtx, HookEvent, HookOutcome,
    ToolResult,
};
use async_trait::async_trait;
use common::{multi_tool_turn, text_turn, FixtureModel};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn bash(script: &str) -> ExternalCommandHook {
    ExternalCommandHook::new("bash").args(["-c", script])
}

#[tokio::test]
async fn external_hook_deny_blocks_tool_call() {
    // Switch on the `event` discriminator: only deny at PreToolUse, pass
    // through for every other event so the loop reaches it.
    let script = r#"
        input=$(cat)
        if echo "$input" | grep -q '"event":"pre_tool_use"'; then
            printf '{"outcome":"deny","reason":"shell says no"}'
        fi
    "#;
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![("u1", "echo", json!({"text": "ping"}))]),
        text_turn("after"),
    ]);
    let echo = FnTool::new(
        "echo",
        "echoes",
        json!({"type": "object"}),
        |_a, _c| async move { Ok(ToolResult::text("ok")) },
    );
    let agent = Agent::builder()
        .model(model)
        .tool(echo)
        .hook(bash(script))
        .build()
        .expect("build");
    let out = agent.run("go").await.expect("run");
    assert_eq!(out.text, "after");
}

#[tokio::test]
async fn external_hook_timeout_returns_halt() {
    // Script sleeps longer than the timeout; the loop must halt promptly.
    let script = "cat > /dev/null; sleep 5";
    let hook = bash(script).timeout(Duration::from_millis(150));

    let model = FixtureModel::new(vec![text_turn("never")]);
    let agent = Agent::builder()
        .model(model)
        .hook(hook)
        .build()
        .expect("build");

    let started = std::time::Instant::now();
    let err = agent.run("hi").await.expect_err("hook timeout -> err");
    let elapsed = started.elapsed();
    assert!(
        matches!(err, AgentError::HookHalt(ref m) if m.contains("timed out")),
        "got {err:?}"
    );
    // Confirm we did NOT wait for the full sleep.
    assert!(
        elapsed < Duration::from_secs(2),
        "hook timeout should be near-immediate, took {elapsed:?}"
    );
}

#[tokio::test]
async fn external_hook_continue_passes_through() {
    let script = r#"
        cat > /dev/null
        printf '{"outcome":"continue"}'
    "#;
    let model = FixtureModel::new(vec![text_turn("ok")]);
    let agent = Agent::builder()
        .model(model)
        .hook(bash(script))
        .build()
        .expect("build");
    let out = agent.run("hi").await.expect("run");
    assert_eq!(out.text, "ok");
}

/// Hook that flips a flag on PostToolUse — proves the loop reached that point.
struct PostToolFlag(Arc<AtomicBool>);

#[async_trait]
impl Hook for PostToolFlag {
    async fn on_event(&self, ctx: HookCtx) -> HookOutcome {
        if matches!(ctx.event, HookEvent::PostToolUse { .. }) {
            self.0.store(true, Ordering::SeqCst);
        }
        HookOutcome::Continue
    }
}

#[tokio::test]
async fn external_hook_cancel_kills_subprocess() {
    // Script would hang forever, but the shared cancel token should kill it.
    let script = "cat > /dev/null; sleep 30";
    let cancel = CancellationToken::new();
    let hook = bash(script)
        .timeout(Duration::from_secs(60))
        .cancel_token(cancel.clone());

    let post_flag = Arc::new(AtomicBool::new(false));
    let model = FixtureModel::new(vec![text_turn("ok")]);
    let agent = Agent::builder()
        .model(model)
        .hook(hook)
        .hook(PostToolFlag(post_flag.clone()))
        .build()
        .expect("build");

    let run = tokio::spawn(async move { agent.run("hi").await });

    // Give the hook a moment to spawn its subprocess, then cancel.
    tokio::time::sleep(Duration::from_millis(100)).await;
    cancel.cancel();

    let result = tokio::time::timeout(Duration::from_secs(3), run)
        .await
        .expect("agent must finish quickly after cancel")
        .expect("join");
    // The hook returned Halt("cancelled"), surfaced as HookHalt.
    assert!(matches!(result, Err(AgentError::HookHalt(_))), "got {result:?}");
    // PostToolFlag never fired — proves the loop short-circuited at PreModel.
    assert!(!post_flag.load(Ordering::SeqCst));
}
