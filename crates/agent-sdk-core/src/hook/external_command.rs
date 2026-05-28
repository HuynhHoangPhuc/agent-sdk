//! [`ExternalCommandHook`] — invoke an external subprocess as a hook.
//!
//! The hook spawns a configured command, writes a JSON-serialized
//! [`HookCtx`](super::HookCtx) to stdin, and reads a JSON-shaped
//! [`HookOutcome`](super::HookOutcome) from stdout. Both timeout and the
//! shared [`CancellationToken`] kill the child via `kill_on_drop(true)`.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::ctx::{HookCtx, HookOutcome};
use super::trait_::Hook;

/// Default timeout if the caller does not override it.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Subprocess-backed [`Hook`].
///
/// `program` + `args` form the command to spawn. The command is shared across
/// every event the hook is registered for; it is the script's responsibility
/// to dispatch on the `event` discriminator in the JSON payload.
pub struct ExternalCommandHook {
    program: String,
    args: Vec<String>,
    timeout: Duration,
    cancel: CancellationToken,
}

impl ExternalCommandHook {
    /// Build a hook that runs `program` with no arguments and the default 5s
    /// timeout.
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            cancel: CancellationToken::new(),
        }
    }

    /// Add CLI args to the spawned command.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Override the per-call timeout. Hitting it surfaces as
    /// [`HookOutcome::Halt`].
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Link an external cancellation token. When fired, every in-flight
    /// invocation kills its subprocess and returns [`HookOutcome::Halt`].
    pub fn cancel_token(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }
}

/// Wire-form of [`HookOutcome`] read from the subprocess.
///
/// We accept a small, explicit JSON shape rather than reflecting `HookOutcome`
/// directly — `HookOutcome` is not `Deserialize` (it carries a
/// `Box<LanguageModelRequest>` that scripts cannot construct sanely).
#[derive(Debug, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum WireOutcome {
    Continue,
    Deny { reason: String },
    Halt { reason: String },
}

#[async_trait]
impl Hook for ExternalCommandHook {
    async fn on_event(&self, ctx: HookCtx) -> HookOutcome {
        let payload = match serde_json::to_vec(&ctx) {
            Ok(b) => b,
            Err(e) => return HookOutcome::Halt(format!("hook ctx serialize failed: {e}")),
        };

        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return HookOutcome::Halt(format!("hook spawn failed: {e}")),
        };

        if let Some(mut stdin) = child.stdin.take() {
            if let Err(e) = stdin.write_all(&payload).await {
                let _ = child.kill().await;
                return HookOutcome::Halt(format!("hook stdin write failed: {e}"));
            }
            // Drop stdin to signal EOF to the child.
            drop(stdin);
        }

        let wait = child.wait_with_output();
        let outcome = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => {
                return HookOutcome::Halt("hook cancelled".into());
            }
            _ = tokio::time::sleep(self.timeout) => {
                return HookOutcome::Halt(format!(
                    "hook timed out after {}ms",
                    self.timeout.as_millis()
                ));
            }
            res = wait => res,
        };

        let output = match outcome {
            Ok(o) => o,
            Err(e) => return HookOutcome::Halt(format!("hook wait failed: {e}")),
        };

        if output.stdout.trim_ascii().is_empty() {
            // No stdout body == pass-through.
            return HookOutcome::Continue;
        }

        match serde_json::from_slice::<WireOutcome>(&output.stdout) {
            Ok(WireOutcome::Continue) => HookOutcome::Continue,
            Ok(WireOutcome::Deny { reason }) => HookOutcome::Deny(reason),
            Ok(WireOutcome::Halt { reason }) => HookOutcome::Halt(reason),
            Err(e) => HookOutcome::Halt(format!("hook output not JSON: {e}")),
        }
    }
}
