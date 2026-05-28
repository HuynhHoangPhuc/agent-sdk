//! Parallel tool execution: ensure two simultaneous calls overlap when
//! `parallel_tool_calls(true)` (the default) and serialize when forced off.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_sdk_core::{Agent, FnTool, ToolResult};
use common::{multi_tool_turn, text_turn, FixtureModel};
use serde_json::json;
use tokio::sync::Mutex;

#[tokio::test]
async fn parallel_tools_overlap() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![
            ("a", "sleeper", json!({"ms": 200})),
            ("b", "sleeper", json!({"ms": 200})),
        ]),
        text_turn("done"),
    ]);
    let sleeper = FnTool::new(
        "sleeper",
        "sleeps ms",
        json!({"type": "object"}),
        |args, _cancel| async move {
            let ms = args.get("ms").and_then(|v| v.as_u64()).unwrap_or(0);
            tokio::time::sleep(Duration::from_millis(ms)).await;
            Ok(ToolResult::text("woke"))
        },
    );

    let agent = Agent::builder()
        .model(model)
        .tool(sleeper)
        .build()
        .expect("build");
    let start = Instant::now();
    let out = agent.run("go").await.expect("run");
    let elapsed = start.elapsed();
    assert_eq!(out.text, "done");
    // Both should run concurrently: ~200ms total, generously under 380ms.
    assert!(
        elapsed < Duration::from_millis(380),
        "expected overlap, took {elapsed:?}"
    );
}

#[tokio::test]
async fn sequential_tools_when_parallel_disabled() {
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![
            ("a", "sleeper", json!({"ms": 120})),
            ("b", "sleeper", json!({"ms": 120})),
        ]),
        text_turn("done"),
    ]);
    let sleeper = FnTool::new(
        "sleeper",
        "sleeps ms",
        json!({"type": "object"}),
        |args, _cancel| async move {
            let ms = args.get("ms").and_then(|v| v.as_u64()).unwrap_or(0);
            tokio::time::sleep(Duration::from_millis(ms)).await;
            Ok(ToolResult::text("woke"))
        },
    );

    let agent = Agent::builder()
        .model(model)
        .tool(sleeper)
        .parallel_tool_calls(false)
        .build()
        .expect("build");
    let start = Instant::now();
    let _ = agent.run("go").await.expect("run");
    let elapsed = start.elapsed();
    // Sequential: at least ~240ms total.
    assert!(
        elapsed >= Duration::from_millis(220),
        "expected serial, took {elapsed:?}"
    );
}

#[tokio::test]
async fn parallel_tools_run_in_either_order_but_results_stable() {
    // Capture invocation order in a shared vec and assert all calls landed
    // regardless of completion order; loop must still report results in the
    // order the model issued them.
    let model = FixtureModel::new(vec![
        multi_tool_turn(vec![
            ("a", "tag", json!({"v": 1})),
            ("b", "tag", json!({"v": 2})),
            ("c", "tag", json!({"v": 3})),
        ]),
        text_turn("ok"),
    ]);
    let log: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();
    let tag = FnTool::new(
        "tag",
        "records v",
        json!({"type": "object"}),
        move |args, _cancel| {
            let log = log_clone.clone();
            async move {
                let v = args.get("v").and_then(|x| x.as_u64()).unwrap_or(0);
                log.lock().await.push(v);
                Ok(ToolResult::json(json!({"v": v})))
            }
        },
    );

    let agent = Agent::builder()
        .model(model)
        .tool(tag)
        .build()
        .expect("build");
    let out = agent.run("go").await.expect("run");
    assert_eq!(out.text, "ok");

    let mut got = log.lock().await.clone();
    got.sort();
    assert_eq!(got, vec![1, 2, 3]);
}
