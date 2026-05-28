#![allow(unused_imports, dead_code)]
use agent_sdk_core::{tool, AgentError, ToolResult};

#[tool]
fn not_async(text: String) -> Result<ToolResult, AgentError> {
    Ok(ToolResult::text(text))
}

fn main() {}
