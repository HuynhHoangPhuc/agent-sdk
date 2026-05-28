//! `grep` tool — regex over a sandboxed directory tree (honors `.gitignore`).

use async_trait::async_trait;
use agent_sdk_core::{AgentError, Tool, ToolResult};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::sandbox::SandboxRoot;

/// Search files under a directory for a regex pattern. Uses the `ignore`
/// crate's parallel-friendly walker so `.gitignore`, `.ignore`, and global
/// excludes are respected automatically.
#[derive(Debug, Clone)]
pub struct GrepTool {
    sandbox: SandboxRoot,
}

impl GrepTool {
    /// Construct with the given sandbox.
    pub fn new(sandbox: SandboxRoot) -> Self {
        Self { sandbox }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GrepArgs {
    /// Regex pattern (Rust `regex` syntax).
    pattern: String,
    /// Directory to search (defaults to the sandbox root / cwd).
    #[serde(default)]
    path: Option<String>,
    /// Match path globs (`*.rs`, `src/**/*.ts`, ...).
    #[serde(default)]
    glob: Option<String>,
    /// Case-insensitive match.
    #[serde(default)]
    case_insensitive: bool,
    /// Maximum matches to return (default 200).
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
struct GrepHit {
    path: String,
    line: u64,
    text: String,
}

const DEFAULT_MAX_RESULTS: usize = 200;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }
    fn description(&self) -> &str {
        "Search files under a directory for a regex pattern. Honors .gitignore."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(GrepArgs)).expect("grep schema")
    }
    async fn execute(
        &self,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let args: GrepArgs =
            serde_json::from_value(arguments).map_err(|e| AgentError::InvalidToolArguments {
                name: "grep".into(),
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
        let case_insensitive = args.case_insensitive;
        let glob = args.glob.clone();
        let sandbox = self.sandbox.clone();
        let cancel_clone = cancel.clone();

        // Walk on a blocking thread; regex/walk are CPU-bound.
        let result = tokio::task::spawn_blocking(move || {
            run_search(&start, &pattern, glob.as_deref(), case_insensitive, max, &sandbox, &cancel_clone)
        })
        .await
        .map_err(|e| tool_err(format!("worker join: {e}")))??;

        Ok(ToolResult::json(serde_json::json!({
            "hits": result.hits,
            "truncated": result.truncated,
        })))
    }
}

struct SearchResult {
    hits: Vec<GrepHit>,
    truncated: bool,
}

fn run_search(
    start: &std::path::Path,
    pattern: &str,
    glob: Option<&str>,
    case_insensitive: bool,
    max: usize,
    sandbox: &SandboxRoot,
    cancel: &CancellationToken,
) -> Result<SearchResult, AgentError> {
    let re = regex::RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .map_err(|e| tool_err(format!("invalid regex: {e}")))?;

    let glob_matcher = match glob {
        Some(g) => Some(
            globset::GlobBuilder::new(g)
                .literal_separator(true)
                .build()
                .map_err(|e| tool_err(format!("invalid glob: {e}")))?
                .compile_matcher(),
        ),
        None => None,
    };

    let walker = ignore::WalkBuilder::new(start)
        .standard_filters(true)
        .require_git(false)
        .hidden(false)
        .build();

    let mut hits = Vec::new();
    let mut truncated = false;
    let root = sandbox.root_path();

    for dent in walker {
        if cancel.is_cancelled() {
            return Err(AgentError::Cancelled);
        }
        let dent = match dent {
            Ok(d) => d,
            Err(_) => continue,
        };
        let path = dent.path();
        if !dent.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        if let Some(root) = root {
            if !path.starts_with(root) {
                continue;
            }
        }
        if let Some(m) = glob_matcher.as_ref() {
            if !m.is_match(path) {
                continue;
            }
        }
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue, // skip binaries / unreadables
        };
        for (idx, line) in contents.lines().enumerate() {
            if re.is_match(line) {
                if hits.len() >= max {
                    truncated = true;
                    return Ok(SearchResult { hits, truncated });
                }
                hits.push(GrepHit {
                    path: path.display().to_string(),
                    line: (idx + 1) as u64,
                    text: line.to_string(),
                });
            }
        }
    }
    Ok(SearchResult { hits, truncated })
}

fn tool_err(message: String) -> AgentError {
    AgentError::Tool {
        name: "grep".into(),
        message,
    }
}
