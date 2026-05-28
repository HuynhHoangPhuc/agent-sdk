#![allow(unused_imports, dead_code)]
use agent_sdk_core::{tool, AgentError, ToolResult};

#[tool]
async fn borrows(text: &str) -> Result<ToolResult, AgentError> {
    Ok(ToolResult::text(text.to_string()))
}

fn main() {}
