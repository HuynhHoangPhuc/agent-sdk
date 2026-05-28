//! `edit` tool — exact-string replacement with unique-match guarantee.

use async_trait::async_trait;
use agent_sdk_core::{AgentError, Tool, ToolResult};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::sandbox::SandboxRoot;

/// Find-and-replace inside a file. Defaults to **exactly one occurrence** of
/// `old_string`; errors if 0 or 2+ matches. Set `replace_all` to replace every
/// occurrence. Writes atomically via temp + rename.
#[derive(Debug, Clone)]
pub struct EditTool {
    sandbox: SandboxRoot,
}

impl EditTool {
    /// Construct with the given sandbox.
    pub fn new(sandbox: SandboxRoot) -> Self {
        Self { sandbox }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct EditArgs {
    /// Path to the file to edit.
    path: String,
    /// Exact substring to replace.
    old_string: String,
    /// Replacement text.
    new_string: String,
    /// Replace every occurrence instead of requiring a unique match.
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }
    fn description(&self) -> &str {
        "Replace a unique substring in a file. Errors if old_string is missing or non-unique; set replace_all to override."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(EditArgs)).expect("edit schema")
    }
    async fn execute(
        &self,
        arguments: serde_json::Value,
        _cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let args: EditArgs =
            serde_json::from_value(arguments).map_err(|e| AgentError::InvalidToolArguments {
                name: "edit".into(),
                message: e.to_string(),
            })?;
        if args.old_string.is_empty() {
            return Err(tool_err("old_string must not be empty".into()));
        }
        let resolved = self
            .sandbox
            .resolve(&args.path)
            .map_err(|e| tool_err(e.to_string()))?;

        let original = tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| tool_err(format!("read {}: {}", resolved.display(), e)))?;

        let count = original.matches(&args.old_string).count();
        if count == 0 {
            return Err(tool_err(format!(
                "old_string not found in {}",
                resolved.display()
            )));
        }
        if count > 1 && !args.replace_all {
            return Err(tool_err(format!(
                "old_string matched {count} times in {}; set replace_all=true or provide a more specific snippet",
                resolved.display()
            )));
        }

        let updated = if args.replace_all {
            original.replace(&args.old_string, &args.new_string)
        } else {
            original.replacen(&args.old_string, &args.new_string, 1)
        };

        // Reuse write's atomic path: write to sibling tmp, rename over.
        let tmp = temp_sibling(&resolved);
        tokio::fs::write(&tmp, updated.as_bytes())
            .await
            .map_err(|e| tool_err(format!("write {}: {}", tmp.display(), e)))?;
        if let Err(e) = tokio::fs::rename(&tmp, &resolved).await {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(tool_err(format!("rename to {}: {}", resolved.display(), e)));
        }

        Ok(ToolResult::text(format!(
            "edited {} ({count} match{})",
            resolved.display(),
            if count == 1 { "" } else { "es" }
        )))
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
    let tmp_name = format!(".{file_name}.{nanos}.edit.tmp");
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(tmp_name),
        _ => std::path::PathBuf::from(tmp_name),
    }
}

fn tool_err(message: String) -> AgentError {
    AgentError::Tool {
        name: "edit".into(),
        message,
    }
}
