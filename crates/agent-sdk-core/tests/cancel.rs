//! Cancellation: mid-stream, mid-tool, and pre-flight.

mod common;

use std::time::Duration;

use agent_sdk_core::{Agent, AgentError, FnTool, ToolResult};
use agent_sdk_language_model::{FinishReason, LanguageModelEvent, Usage};
use common::{multi_tool_turn, text_turn, FixtureModel, ScriptedTurn};
use serde_json::json;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn cancel_mid_stream() {
    // Stream emits text deltas with a 50ms gap so we can cancel in the middle.
    let model = FixtureModel::new(vec![ScriptedTurn {
        events: vec![
            LanguageModelEvent::TextDelta {
                delta: "hello ".into(),
            },
            LanguageModelEvent::TextDelta {
                delta: "world ".into(),
            },
            LanguageModelEvent::TextDelta {
                delta: "again".into(),
            },
            LanguageModelEvent::Finish {
                reason: FinishReason::Stop,
                usage: Usage::default(),
            },
        ],
        delay_between_events: Duration::from_millis(50),
    }]);
    let agent = Agent::builder().model(model).build().expect("build");
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(60)).await;
        cancel_clone.cancel();
    });

    let out = agent.run_with_cancel("go", cancel).await;
    match out {
        Err(AgentError::Cancelled) => {}
        // The error may also surface as Other("agent run cancelled") via the
        // terminal error event path; both prove the loop honoured cancel.
        Err(AgentError::Other(msg)) if msg.contains("cancel") => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_mid_tool() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![("a", "slow", json!({"ms": 1000}))]),
        text_turn("never reached"),
    ]);
    let slow = FnTool::new(
        "slow",
        "long sleep that respects cancel",
        json!({"type": "object"}),
        |_args, cancel| async move {
            tokio::select! {
                _ = cancel.cancelled() => Err(AgentError::Cancelled),
                _ = tokio::time::sleep(Duration::from_secs(5)) => Ok(ToolResult::text("done")),
            }
        },
    );
    let agent = Agent::builder()
        .model(model)
        .tool(slow)
        .build()
        .expect("build");

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        cancel_clone.cancel();
    });

    let start = std::time::Instant::now();
    let out = agent.run_with_cancel("go", cancel).await;
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_millis(800), "cancel was slow: {elapsed:?}");
    match out {
        Err(AgentError::Cancelled) => {}
        Err(AgentError::Other(msg)) if msg.contains("cancel") => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[tokio::test]
async fn precancelled_token_terminates_immediately() {
    let model = FixtureModel::new(vec![text_turn("never seen")]);
    let agent = Agent::builder().model(model).build().expect("build");
    let cancel = CancellationToken::new();
    cancel.cancel();
    let out = agent.run_with_cancel("go", cancel).await;
    match out {
        Err(AgentError::Cancelled) => {}
        Err(AgentError::Other(msg)) if msg.contains("cancel") => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
}
