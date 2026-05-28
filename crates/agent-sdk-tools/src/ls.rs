//! `ls` tool — directory listing with optional depth + ignore patterns.

use async_trait::async_trait;
use agent_sdk_core::{AgentError, Tool, ToolResult};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::sandbox::SandboxRoot;

/// List entries under a directory. Walks recursively with an optional `depth`
/// cap (1 = direct children only). Honors `.gitignore`.
#[derive(Debug, Clone)]
pub struct LsTool {
    sandbox: SandboxRoot,
}

impl LsTool {
    /// Construct with the given sandbox.
    pub fn new(sandbox: SandboxRoot) -> Self {
        Self { sandbox }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LsArgs {
    /// Directory to list (defaults to sandbox root / cwd).
    #[serde(default)]
    path: Option<String>,
    /// Maximum recursion depth (1 = direct children). Omit for unlimited.
    #[serde(default)]
    depth: Option<usize>,
    /// Maximum entries to return (default 500).
    #[serde(default)]
    max_results: Option<usize>,
}

const DEFAULT_MAX_RESULTS: usize = 500;

#[derive(Debug, serde::Serialize)]
struct LsEntry {
    path: String,
    is_dir: bool,
}

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }
    fn description(&self) -> &str {
        "List directory contents recursively. Optional depth cap. Honors .gitignore."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(LsArgs)).expect("ls schema")
    }
    async fn execute(
        &self,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let args: LsArgs =
            serde_json::from_value(arguments).map_err(|e| AgentError::InvalidToolArguments {
                name: "ls".into(),
                message: e.to_string(),
            })?;

        let start = match args.path {
            Some(p) => self
                .sandbox
                .resolve(&p)
                .map_err(|e| tool_err(e.to_string()))?,
            None => self
                .sandbox
                .root_path()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_default()),
        };

        let max = args.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
        let depth = args.depth;
        let cancel_clone = cancel.clone();

        let (entries, truncated) = tokio::task::spawn_blocking(
            move || -> Result<(Vec<LsEntry>, bool), AgentError> {
                let mut builder = ignore::WalkBuilder::new(&start);
                builder.standard_filters(true).require_git(false).hidden(false);
                if let Some(d) = depth {
                    builder.max_depth(Some(d));
                }
                let mut out = Vec::new();
                let mut truncated = false;
                for dent in builder.build() {
                    if cancel_clone.is_cancelled() {
                        return Err(AgentError::Cancelled);
                    }
                    let dent = match dent {
                        Ok(d) => d,
                        Err(_) => continue,
                    };
                    if dent.depth() == 0 {
                        continue; // skip the root itself
                    }
                    if out.len() >= max {
                        truncated = true;
                        break;
                    }
                    out.push(LsEntry {
                        path: dent.path().display().to_string(),
                        is_dir: dent.file_type().map(|t| t.is_dir()).unwrap_or(false),
                    });
                }
                Ok((out, truncated))
            },
        )
        .await
        .map_err(|e| tool_err(format!("worker join: {e}")))??;

        Ok(ToolResult::json(serde_json::json!({
            "entries": entries,
            "truncated": truncated,
        })))
    }
}

fn tool_err(message: String) -> AgentError {
    AgentError::Tool {
        name: "ls".into(),
        message,
    }
}
