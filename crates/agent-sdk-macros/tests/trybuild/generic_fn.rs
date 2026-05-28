#![allow(unused_imports, dead_code)]
use agent_sdk_core::{tool, AgentError, ToolResult};

#[tool]
async fn generic<T: ToString>(value: T) -> Result<ToolResult, AgentError> {
    Ok(ToolResult::text(value.to_string()))
}

fn main() {}
