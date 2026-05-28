//! Hook system integration tests.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_sdk_core::{
    Agent, AgentError, FnTool, Hook, HookCtx, HookEvent, HookOutcome, ToolResult,
};
use async_trait::async_trait;
use common::{multi_tool_turn, text_turn, FixtureModel};
use serde_json::json;

/// Hook that records which event types it saw and how many times.
struct RecorderHook {
    pre_model: AtomicUsize,
    post_model: AtomicUsize,
    pre_tool: AtomicUsize,
    post_tool: AtomicUsize,
    user_prompt: AtomicUsize,
    on_finish: AtomicUsize,
    on_error: AtomicUsize,
}

impl RecorderHook {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            pre_model: 0.into(),
            post_model: 0.into(),
            pre_tool: 0.into(),
            post_tool: 0.into(),
            user_prompt: 0.into(),
            on_finish: 0.into(),
            on_error: 0.into(),
        })
    }
}

#[async_trait]
impl Hook for RecorderHook {
    async fn on_event(&self, ctx: HookCtx) -> HookOutcome {
        match ctx.event {
            HookEvent::UserPromptSubmit { .. } => {
                self.user_prompt.fetch_add(1, Ordering::SeqCst);
            }
            HookEvent::PreModel { .. } => {
                self.pre_model.fetch_add(1, Ordering::SeqCst);
            }
            HookEvent::PostModel { .. } => {
                self.post_model.fetch_add(1, Ordering::SeqCst);
            }
            HookEvent::PreToolUse { .. } => {
                self.pre_tool.fetch_add(1, Ordering::SeqCst);
            }
            HookEvent::PostToolUse { .. } => {
                self.post_tool.fetch_add(1, Ordering::SeqCst);
            }
            HookEvent::OnFinish { .. } => {
                self.on_finish.fetch_add(1, Ordering::SeqCst);
            }
            HookEvent::OnError { .. } => {
                self.on_error.fetch_add(1, Ordering::SeqCst);
            }
            _ => {}
        }
        HookOutcome::Continue
    }
}

#[tokio::test]
async fn hooks_fire_for_every_lifecycle_event() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![("u1", "echo", json!({"text": "ping"}))]),
        text_turn("done"),
    ]);
    let echo = FnTool::new(
        "echo",
        "echoes",
        json!({"type": "object"}),
        |_a, _c| async move { Ok(ToolResult::text("ok")) },
    );
    let rec = RecorderHook::new();
    let agent = Agent::builder()
        .model(model)
        .tool(echo)
        .hook_arc(rec.clone())
        .build()
        .expect("build");

    let out = agent.run("hi").await.expect("run");
    assert_eq!(out.text, "done");

    assert_eq!(rec.user_prompt.load(Ordering::SeqCst), 1, "user_prompt");
    assert_eq!(rec.pre_model.load(Ordering::SeqCst), 2, "pre_model");
    assert_eq!(rec.post_model.load(Ordering::SeqCst), 2, "post_model");
    assert_eq!(rec.pre_tool.load(Ordering::SeqCst), 1, "pre_tool");
    assert_eq!(rec.post_tool.load(Ordering::SeqCst), 1, "post_tool");
    assert_eq!(rec.on_finish.load(Ordering::SeqCst), 1, "on_finish");
    assert_eq!(rec.on_error.load(Ordering::SeqCst), 0, "on_error");
}

/// Hook that denies any tool call.
struct DenyToolHook;

#[async_trait]
impl Hook for DenyToolHook {
    async fn on_event(&self, ctx: HookCtx) -> HookOutcome {
        match ctx.event {
            HookEvent::PreToolUse { .. } => HookOutcome::Deny("hook says no".into()),
            _ => HookOutcome::Continue,
        }
    }
}

#[tokio::test]
async fn pre_tool_use_deny_synthesizes_error_and_loop_continues() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![("u1", "echo", json!({"text": "ping"}))]),
        text_turn("after-deny"),
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
        .hook(DenyToolHook)
        .build()
        .expect("build");
    let out = agent.run("go").await.expect("run");
    assert_eq!(out.text, "after-deny");
}

/// Hook that rewrites the system prompt at PreModel.
struct RewriteSystemHook;

#[async_trait]
impl Hook for RewriteSystemHook {
    async fn on_event(&self, ctx: HookCtx) -> HookOutcome {
        match ctx.event {
            HookEvent::PreModel { mut request } => {
                request.system = Some("rewritten".into());
                HookOutcome::ModifyRequest(Box::new(request))
            }
            _ => HookOutcome::Continue,
        }
    }
}

/// Captures the system prompt of the most recent stream() call.
struct CaptureSystemModel {
    last_system: Arc<std::sync::Mutex<Option<String>>>,
}

#[async_trait]
impl agent_sdk_language_model::LanguageModel for CaptureSystemModel {
    fn provider_id(&self) -> &str {
        "capture"
    }
    fn model_id(&self) -> &str {
        "capture-1"
    }
    async fn stream(
        &self,
        req: agent_sdk_language_model::LanguageModelRequest,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> Result<
        agent_sdk_language_model::BoxedEventStream,
        agent_sdk_language_model::ModelError,
    > {
        *self.last_system.lock().unwrap() = req.system.clone();
        let s = futures_util::stream::iter(vec![
            Ok(agent_sdk_language_model::LanguageModelEvent::TextDelta {
                delta: "hi".into(),
            }),
            Ok(agent_sdk_language_model::LanguageModelEvent::Finish {
                reason: agent_sdk_language_model::FinishReason::Stop,
                usage: agent_sdk_language_model::Usage::default(),
            }),
        ]);
        Ok(Box::pin(s))
    }
}

#[tokio::test]
async fn pre_model_modify_request_alters_provider_input() {
    let captured = Arc::new(std::sync::Mutex::new(None));
    let model = CaptureSystemModel {
        last_system: captured.clone(),
    };
    let agent = Agent::builder()
        .model(model)
        .system("original")
        .hook(RewriteSystemHook)
        .build()
        .expect("build");
    let _ = agent.run("hello").await.expect("run");
    assert_eq!(
        captured.lock().unwrap().clone(),
        Some("rewritten".to_string()),
        "PreModel hook should rewrite system"
    );
}

/// Hook that halts at PreModel.
struct HaltHook;

#[async_trait]
impl Hook for HaltHook {
    async fn on_event(&self, ctx: HookCtx) -> HookOutcome {
        match ctx.event {
            HookEvent::PreModel { .. } => HookOutcome::Halt("halt-now".into()),
            _ => HookOutcome::Continue,
        }
    }
}

#[tokio::test]
async fn hook_halt_at_pre_model_ends_loop() {
    let model = FixtureModel::new(vec![text_turn("never")]);
    let agent = Agent::builder()
        .model(model)
        .hook(HaltHook)
        .build()
        .expect("build");
    let err = agent.run("hi").await.expect_err("halt -> error");
    assert!(matches!(err, AgentError::HookHalt(ref m) if m == "halt-now"), "got {err:?}");
}
