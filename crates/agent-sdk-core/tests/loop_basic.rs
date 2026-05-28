//! Basic loop behaviours: text-only run, single-tool round-trip, max_turns.

mod common;

use agent_sdk_core::{Agent, AgentError, FnTool, ToolResult};
use common::{text_turn, tool_turn, FixtureModel, ScriptedTurn};
use serde_json::json;

#[tokio::test]
async fn run_returns_aggregated_text() {
    let model = FixtureModel::new(vec![text_turn("hello world")]);
    let agent = Agent::builder().model(model).build().expect("build");
    let out = agent.run("hi").await.expect("run");
    assert_eq!(out.text, "hello world");
    assert_eq!(out.turns, 1);
}

#[tokio::test]
async fn tool_round_trip_then_text() {
    let model = FixtureModel::new(vec![
        tool_turn("toolu_1", "echo", json!({"text": "ping"})),
        text_turn("done: ping"),
    ]);
    let echo = FnTool::new(
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
    );
    let agent = Agent::builder()
        .model(model)
        .tool(echo)
        .build()
        .expect("build");
    let out = agent.run("please").await.expect("run");
    assert_eq!(out.text, "done: ping");
    assert_eq!(out.turns, 2);
}

#[tokio::test]
async fn max_turns_returns_error() {
    // Three tool turns scripted, but max_turns=2 -> error on third.
    let model = FixtureModel::new(vec![
        tool_turn("a", "spin", json!({})),
        tool_turn("b", "spin", json!({})),
        tool_turn("c", "spin", json!({})),
    ]);
    let spin = FnTool::new(
        "spin",
        "no-op",
        json!({"type": "object"}),
        |_args, _cancel| async move { Ok(ToolResult::text("ok")) },
    );
    let agent = Agent::builder()
        .model(model)
        .tool(spin)
        .max_turns(2)
        .build()
        .expect("build");
    let err = agent.run("go").await.expect_err("should error");
    assert!(matches!(err, AgentError::MaxTurns(2)), "got {err:?}");
}

#[tokio::test]
async fn invalid_tool_arguments_is_an_error() {
    use agent_sdk_language_model::{FinishReason, LanguageModelEvent, Usage};
    // Provider streams tool args that don't form valid JSON; loop must surface
    // AgentError::InvalidToolArguments rather than silently sending `{}`.
    let model = FixtureModel::new(vec![ScriptedTurn::instant(vec![
        LanguageModelEvent::ToolCallStart {
            id: "id1".into(),
            name: "echo".into(),
        },
        LanguageModelEvent::ToolCallDelta {
            id: "id1".into(),
            arguments_delta: "{not".into(),
        },
        LanguageModelEvent::ToolCallEnd {
            id: "id1".into(),
            arguments: None,
        },
        LanguageModelEvent::Finish {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
        },
    ])]);
    let echo = FnTool::new(
        "echo",
        "echoes",
        json!({"type": "object"}),
        |_args, _cancel| async move { Ok(ToolResult::text("x")) },
    );
    let agent = Agent::builder()
        .model(model)
        .tool(echo)
        .build()
        .expect("build");
    let out = agent.run("hi").await;
    match out {
        Err(AgentError::InvalidToolArguments { name, .. }) => assert_eq!(name, "echo"),
        other => panic!("expected InvalidToolArguments, got {other:?}"),
    }
}

#[tokio::test]
async fn missing_model_is_a_config_error() {
    let err = Agent::builder().build().expect_err("no model");
    assert!(matches!(err, AgentError::InvalidConfig(_)));
}
