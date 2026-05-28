//! `read` tool — read a file with optional line-range slicing and byte cap.

use async_trait::async_trait;
use agent_sdk_core::{AgentError, Tool, ToolResult};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::sandbox::SandboxRoot;

/// Read a file. Optional 1-based line `offset` + `limit`; `max_bytes` caps the
/// total bytes returned (default 256 KiB).
#[derive(Debug, Clone)]
pub struct ReadTool {
    sandbox: SandboxRoot,
}

impl ReadTool {
    /// Construct with the given sandbox.
    pub fn new(sandbox: SandboxRoot) -> Self {
        Self { sandbox }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ReadArgs {
    /// Absolute or sandbox-relative path to the file.
    path: String,
    /// 1-based starting line (inclusive). Omit to read from the start.
    #[serde(default)]
    offset: Option<u64>,
    /// Maximum number of lines to return.
    #[serde(default)]
    limit: Option<u64>,
    /// Hard cap on returned bytes (default 256 KiB).
    #[serde(default)]
    max_bytes: Option<u64>,
}

const DEFAULT_MAX_BYTES: u64 = 256 * 1024;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }
    fn description(&self) -> &str {
        "Read a file from disk. Supports optional 1-based line offset/limit and a byte cap."
    }
    fn input_schema(&self) -> serde_json::Value {
        let s = schemars::schema_for!(ReadArgs);
        serde_json::to_value(&s).expect("read schema")
    }
    async fn execute(
        &self,
        arguments: serde_json::Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let args: ReadArgs = serde_json::from_value(arguments).map_err(|e| {
            AgentError::InvalidToolArguments {
                name: "read".into(),
                message: e.to_string(),
            }
        })?;
        let resolved =
            self.sandbox
                .resolve(&args.path)
                .map_err(|e| AgentError::Tool {
                    name: "read".into(),
                    message: e.to_string(),
                })?;

        let bytes = tokio::fs::read(&resolved)
            .await
            .map_err(|e| AgentError::Tool {
                name: "read".into(),
                message: format!("read {}: {}", resolved.display(), e),
            })?;
        let text = String::from_utf8_lossy(&bytes).into_owned();

        let cap = args.max_bytes.unwrap_or(DEFAULT_MAX_BYTES) as usize;
        let (sliced, truncated_lines) = slice_lines(&text, args.offset, args.limit);
        let (capped, truncated_bytes) = cap_bytes(sliced, cap);

        if truncated_lines || truncated_bytes {
            Ok(ToolResult::json(serde_json::json!({
                "content": capped,
                "truncated": true,
            })))
        } else {
            Ok(ToolResult::text(capped))
        }
    }
}

fn slice_lines(text: &str, offset: Option<u64>, limit: Option<u64>) -> (String, bool) {
    if offset.is_none() && limit.is_none() {
        return (text.to_string(), false);
    }
    let start = offset.unwrap_or(1).saturating_sub(1) as usize;
    let max = limit.map(|l| l as usize).unwrap_or(usize::MAX);
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let end = start.saturating_add(max).min(total);
    let truncated = end < total || start > 0;
    if start >= total {
        return (String::new(), truncated);
    }
    (lines[start..end].join("\n"), truncated)
}

fn cap_bytes(text: String, cap: usize) -> (String, bool) {
    if text.len() <= cap {
        return (text, false);
    }
    // Cut on a char boundary at or before `cap`.
    let mut end = cap;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}
