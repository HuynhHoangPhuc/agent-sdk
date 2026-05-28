//! Expansion + behaviour tests for `#[tool]`.
//!
//! Each test instantiates the generated unit struct, queries its `Tool` impl
//! through the dyn dispatch the agent loop uses, and checks the JSON Schema
//! shape that `schemars` derives from the fn signature.

use agent_sdk_core::{tool, AgentError, Tool, ToolResult};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Echo back the text the model passes in.
#[tool]
async fn echo(
    /// Text to echo verbatim.
    text: String,
) -> Result<ToolResult, AgentError> {
    Ok(ToolResult::text(text))
}

/// Read up to `limit` lines of a file.
#[tool]
async fn read_file(
    /// Absolute path of the file.
    path: String,
    /// Maximum number of lines to read.
    limit: Option<usize>,
) -> Result<ToolResult, AgentError> {
    let limit = limit.unwrap_or(0);
    Ok(ToolResult::text(format!("read {path} limit={limit}")))
}

/// Sum a list of integers.
#[tool]
async fn sum_list(values: Vec<i64>) -> Result<ToolResult, AgentError> {
    let total: i64 = values.iter().sum();
    Ok(ToolResult::json(json!({ "sum": total })))
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct Pair {
    a: i64,
    b: i64,
}

/// Add two integers carried in a struct argument.
#[tool]
async fn add_pair(pair: Pair) -> Result<ToolResult, AgentError> {
    Ok(ToolResult::json(json!({ "result": pair.a + pair.b })))
}

fn schema_of(t: &dyn Tool) -> serde_json::Value {
    t.input_schema()
}

#[tokio::test]
async fn scalar_param_executes_and_describes() {
    let t: Arc<dyn Tool> = Arc::new(echo);
    assert_eq!(t.name(), "echo");
    assert_eq!(t.description(), "Echo back the text the model passes in.");
    let out = t
        .execute(json!({ "text": "hi" }), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(out.content, json!("hi"));
    assert!(!out.is_error);
}

#[tokio::test]
async fn optional_param_is_optional_in_schema() {
    let t: Arc<dyn Tool> = Arc::new(read_file);
    let schema = schema_of(&*t);
    let required = schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect::<Vec<_>>())
        .unwrap_or_default();
    assert!(required.contains(&"path".to_string()));
    assert!(!required.contains(&"limit".to_string()));

    // Omitting the optional field deserializes successfully.
    let out = t
        .execute(json!({ "path": "/tmp/x" }), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(out.content, json!("read /tmp/x limit=0"));
}

#[tokio::test]
async fn doc_comments_appear_in_schema() {
    let schema = schema_of(&read_file);
    let props = schema.get("properties").expect("object schema");
    let path_desc = props
        .get("path")
        .and_then(|v| v.get("description"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(path_desc.contains("Absolute path"), "got: {path_desc:?}");
}

#[tokio::test]
async fn vec_param_round_trips() {
    let t: Arc<dyn Tool> = Arc::new(sum_list);
    let out = t
        .execute(json!({ "values": [1, 2, 3, 4] }), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(out.content, json!({ "sum": 10 }));
}

#[tokio::test]
async fn struct_param_uses_nested_schema() {
    let t: Arc<dyn Tool> = Arc::new(add_pair);
    let out = t
        .execute(
            json!({ "pair": { "a": 2, "b": 3 } }),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(out.content, json!({ "result": 5 }));
}

#[tokio::test]
async fn bad_arguments_surface_as_tool_error() {
    let t: Arc<dyn Tool> = Arc::new(echo);
    // `text` is required and must be a string; passing an int violates both.
    let err = t
        .execute(json!({ "text": 42 }), CancellationToken::new())
        .await
        .unwrap_err();
    match err {
        AgentError::InvalidToolArguments { name, .. } => assert_eq!(name, "echo"),
        other => panic!("expected AgentError::InvalidToolArguments, got {other:?}"),
    }
}
