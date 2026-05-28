//! `glob` tool — file paths matching a pattern.

use async_trait::async_trait;
use agent_sdk_core::{AgentError, Tool, ToolResult};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::sandbox::SandboxRoot;

/// Walk the sandboxed tree and emit every path matching `pattern` (globset
/// syntax: `**/*.rs`, `src/*.ts`, ...).
#[derive(Debug, Clone)]
pub struct GlobTool {
    sandbox: SandboxRoot,
}

impl GlobTool {
    /// Construct with the given sandbox.
    pub fn new(sandbox: SandboxRoot) -> Self {
        Self { sandbox }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GlobArgs {
    /// Glob pattern (e.g. `**/*.rs`).
    pattern: String,
    /// Directory to search (defaults to sandbox root / cwd).
    #[serde(default)]
    path: Option<String>,
    /// Maximum paths to return (default 500).
    #[serde(default)]
    max_results: Option<usize>,
}

const DEFAULT_MAX_RESULTS: usize = 500;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }
    fn description(&self) -> &str {
        "List files matching a glob pattern. Honors .gitignore."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(GlobArgs)).expect("glob schema")
    }
    async fn execute(
        &self,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let args: GlobArgs =
            serde_json::from_value(arguments).map_err(|e| AgentError::InvalidToolArguments {
                name: "glob".into(),
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
        let pattern = args.pattern.clone();
        let cancel_clone = cancel.clone();

        let result = tokio::task::spawn_blocking(move || -> Result<(Vec<String>, bool), AgentError> {
            let matcher = globset::GlobBuilder::new(&pattern)
                .literal_separator(true)
                .build()
                .map_err(|e| tool_err(format!("invalid glob: {e}")))?
                .compile_matcher();

            let walker = ignore::WalkBuilder::new(&start)
                .standard_filters(true)
                .require_git(false)
                .hidden(false)
                .build();

            let mut paths = Vec::new();
            let mut truncated = false;
            for dent in walker {
                if cancel_clone.is_cancelled() {
                    return Err(AgentError::Cancelled);
                }
                let dent = match dent {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let path = dent.path();
                let rel = path.strip_prefix(&start).unwrap_or(path);
                if matcher.is_match(rel) || matcher.is_match(path) {
                    if paths.len() >= max {
                        truncated = true;
                        break;
                    }
                    paths.push(path.display().to_string());
                }
            }
            Ok((paths, truncated))
        })
        .await
        .map_err(|e| tool_err(format!("worker join: {e}")))??;

        Ok(ToolResult::json(serde_json::json!({
            "paths": result.0,
            "truncated": result.1,
        })))
    }
}

fn tool_err(message: String) -> AgentError {
    AgentError::Tool {
        name: "glob".into(),
        message,
    }
}
