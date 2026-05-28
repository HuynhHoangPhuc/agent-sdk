//! `write` tool — atomic file write (temp + rename).

use async_trait::async_trait;
use agent_sdk_core::{AgentError, Tool, ToolResult};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::sandbox::SandboxRoot;

/// Atomically write `content` to `path`. Parent directories are created as
/// needed; the file is materialized via temp-file + rename so a crash mid-write
/// never leaves a partial file.
#[derive(Debug, Clone)]
pub struct WriteTool {
    sandbox: SandboxRoot,
}

impl WriteTool {
    /// Construct with the given sandbox.
    pub fn new(sandbox: SandboxRoot) -> Self {
        Self { sandbox }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct WriteArgs {
    /// Destination path (absolute or sandbox-relative).
    path: String,
    /// Contents to write.
    content: String,
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }
    fn description(&self) -> &str {
        "Atomically write text to a file. Creates parent directories. Overwrites any existing file."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(WriteArgs)).expect("write schema")
    }
    async fn execute(
        &self,
        arguments: serde_json::Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let args: WriteArgs =
            serde_json::from_value(arguments).map_err(|e| AgentError::InvalidToolArguments {
                name: "write".into(),
                message: e.to_string(),
            })?;
        let resolved = self
            .sandbox
            .resolve(&args.path)
            .map_err(|e| tool_err(e.to_string()))?;

        if let Some(parent) = resolved.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| tool_err(format!("mkdir {}: {}", parent.display(), e)))?;
            }
        }

        let tmp = temp_sibling(&resolved);
        tokio::fs::write(&tmp, args.content.as_bytes())
            .await
            .map_err(|e| tool_err(format!("write {}: {}", tmp.display(), e)))?;
        if let Err(e) = tokio::fs::rename(&tmp, &resolved).await {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(tool_err(format!("rename to {}: {}", resolved.display(), e)));
        }
        Ok(ToolResult::text(format!("wrote {}", resolved.display())))
    }
}

fn temp_sibling(path: &std::path::Path) -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let file_name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".into());
    let tmp_name = format!(".{file_name}.{nanos}.tmp");
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(tmp_name),
        _ => std::path::PathBuf::from(tmp_name),
    }
}

fn tool_err(message: String) -> AgentError {
    AgentError::Tool {
        name: "write".into(),
        message,
    }
}
