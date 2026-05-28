//! Permission policy integration tests.

mod common;

use std::sync::Arc;

use agent_sdk_core::{
    channel_callback, Agent, AllowList, Decision, DenyList, FnTool, ToolResult,
};
use common::{multi_tool_turn, text_turn, FixtureModel};
use serde_json::json;

fn echo_tool() -> FnTool {
    FnTool::new(
        "echo",
        "echoes",
        json!({"type": "object"}),
        |args, _cancel| async move {
            let t = args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok(ToolResult::text(t))
        },
    )
}

#[tokio::test]
async fn deny_list_blocks_call_and_continues() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![("u1", "echo", json!({"text": "ping"}))]),
        text_turn("after-deny"),
    ]);
    let agent = Agent::builder()
        .model(model)
        .tool(echo_tool())
        .permission(DenyList::new(["echo"]))
        .build()
        .expect("build");

    let out = agent.run("go").await.expect("run");
    assert_eq!(out.text, "after-deny");
    // Loop did not error — the tool result was a synthetic error sent to the model.
    assert_eq!(out.turns, 2);
}

#[tokio::test]
async fn allow_list_routes_unknown_to_ask_user_and_resolves_allow() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![("u1", "echo", json!({"text": "ping"}))]),
        text_turn("done"),
    ]);

    let (cb, mut rx) = channel_callback();

    // Spawn a harness task that approves every prompt.
    tokio::spawn(async move {
        while let Some(prompt) = rx.recv().await {
            let _ = prompt.reply.send(Decision::Allow);
        }
    });

    let agent = Agent::builder()
        .model(model)
        .tool(echo_tool())
        .permission(AllowList::new(Vec::<String>::new()))
        .permission_ask(cb)
        .build()
        .expect("build");

    let out = agent.run("go").await.expect("run");
    assert_eq!(out.text, "done");
    assert_eq!(out.turns, 2);
}

#[tokio::test]
async fn ask_user_without_callback_degrades_to_deny() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![("u1", "echo", json!({"text": "ping"}))]),
        text_turn("after"),
    ]);

    let agent = Agent::builder()
        .model(model)
        .tool(echo_tool())
        .permission(AllowList::new(Vec::<String>::new()))
        .build()
        .expect("build");

    let out = agent.run("go").await.expect("run");
    // Loop continued — the deny was reported as a tool error to the model.
    assert_eq!(out.text, "after");
    assert_eq!(out.turns, 2);
}

#[tokio::test]
async fn ask_user_explicit_deny_blocks_call_but_loop_continues() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![("u1", "echo", json!({"text": "ping"}))]),
        text_turn("ok"),
    ]);

    let (cb, mut rx) = channel_callback();
    tokio::spawn(async move {
        if let Some(p) = rx.recv().await {
            let _ = p.reply.send(Decision::Deny("user said no".into()));
        }
    });

    let agent = Agent::builder()
        .model(model)
        .tool(echo_tool())
        .permission(AllowList::new(Vec::<String>::new()))
        .permission_ask(Arc::clone(&cb))
        .build()
        .expect("build");

    let out = agent.run("go").await.expect("run");
    assert_eq!(out.text, "ok");
}
