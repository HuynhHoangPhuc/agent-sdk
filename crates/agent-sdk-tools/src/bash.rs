//! `bash` tool — run a shell command with timeout + cancel + output capture.
//!
//! - cwd defaults to the sandbox root (or the resolved `cwd` argument).
//! - The child is `kill_on_drop`, so dropping the future on cancellation also
//!   reaps the process (no zombie / orphan).
//! - stdout + stderr are merged into a single capped buffer; output beyond
//!   `max_bytes` is truncated and a notice appended.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use agent_sdk_core::{AgentError, Tool, ToolResult};
use serde::Deserialize;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::sandbox::SandboxRoot;

/// Execute a shell command (`sh -c <command>`).
#[derive(Debug, Clone)]
pub struct BashTool {
    sandbox: SandboxRoot,
}

impl BashTool {
    /// Construct with the given sandbox.
    pub fn new(sandbox: SandboxRoot) -> Self {
        Self { sandbox }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct BashArgs {
    /// Shell command line, executed as `sh -c <command>`.
    command: String,
    /// Working directory (defaults to sandbox root / cwd).
    #[serde(default)]
    cwd: Option<String>,
    /// Hard timeout in seconds (default 120, max 600).
    #[serde(default)]
    timeout_secs: Option<u64>,
    /// Combined stdout+stderr byte cap (default 64 KiB).
    #[serde(default)]
    max_bytes: Option<u64>,
}

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_TIMEOUT_SECS: u64 = 600;
const DEFAULT_MAX_BYTES: usize = 64 * 1024;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> &str {
        "Run a shell command (sh -c). Supports cwd, timeout, output cap. Honors the sandbox root."
    }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(BashArgs)).expect("bash schema")
    }
    async fn execute(
        &self,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        let args: BashArgs =
            serde_json::from_value(arguments).map_err(|e| AgentError::InvalidToolArguments {
                name: "bash".into(),
                message: e.to_string(),
            })?;

        let cwd = match args.cwd {
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

        let timeout = Duration::from_secs(
            args.timeout_secs
                .unwrap_or(DEFAULT_TIMEOUT_SECS)
                .min(MAX_TIMEOUT_SECS),
        );
        let cap = args.max_bytes.map(|b| b as usize).unwrap_or(DEFAULT_MAX_BYTES);

        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg(&args.command)
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().map_err(|e| tool_err(format!("spawn sh: {e}")))?;
        let mut stdout = child.stdout.take().expect("piped stdout");
        let mut stderr = child.stderr.take().expect("piped stderr");

        let mut out_buf = Vec::with_capacity(cap.min(8 * 1024));
        let mut err_buf = Vec::with_capacity(cap.min(8 * 1024));

        let collect = async {
            let drain_out = read_capped(&mut stdout, &mut out_buf, cap);
            let drain_err = read_capped(&mut stderr, &mut err_buf, cap);
            tokio::join!(drain_out, drain_err);
            child.wait().await
        };

        let status = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                // child + pipe readers dropped → kill_on_drop reaps
                return Err(AgentError::Cancelled);
            }
            res = tokio::time::timeout(timeout, collect) => {
                match res {
                    Ok(Ok(status)) => status,
                    Ok(Err(e)) => return Err(tool_err(format!("wait: {e}"))),
                    Err(_) => {
                        // dropped here → kill_on_drop
                        return Err(tool_err(format!(
                            "command exceeded timeout of {}s",
                            timeout.as_secs()
                        )));
                    }
                }
            }
        };

        let stdout_str = truncate_lossy(&out_buf, cap);
        let stderr_str = truncate_lossy(&err_buf, cap);
        let exit_code = status.code();
        let payload = serde_json::json!({
            "stdout": stdout_str,
            "stderr": stderr_str,
            "exit_code": exit_code,
        });
        let mut result = ToolResult::json(payload);
        result.is_error = !status.success();
        Ok(result)
    }
}

async fn read_capped<R>(reader: &mut R, sink: &mut Vec<u8>, cap: usize)
where
    R: AsyncReadExt + Unpin,
{
    let mut chunk = [0u8; 4096];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) => return,
            Ok(n) => {
                let remaining = cap.saturating_sub(sink.len());
                if remaining == 0 {
                    // drain to EOF without storing more
                    continue;
                }
                let take = n.min(remaining);
                sink.extend_from_slice(&chunk[..take]);
            }
            Err(_) => return,
        }
    }
}

fn truncate_lossy(buf: &[u8], cap: usize) -> String {
    let mut s = String::from_utf8_lossy(buf).into_owned();
    if buf.len() >= cap {
        s.push_str("\n…[output truncated]");
    }
    s
}

fn tool_err(message: String) -> AgentError {
    AgentError::Tool {
        name: "bash".into(),
        message,
    }
}
