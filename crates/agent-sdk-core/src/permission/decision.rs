//! [`Decision`] — outcome of a [`PermissionPolicy`](super::PermissionPolicy) check.

use serde::{Deserialize, Serialize};

/// Permission verdict for a single tool call.
///
/// `#[non_exhaustive]` so we can add verdicts (e.g., `AllowOnce`,
/// `AllowWithLimits`) in later phases without a major bump.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Decision {
    /// Run the tool.
    Allow,
    /// Block the tool. The reason is surfaced to the model as a tool-result
    /// error so the loop keeps going.
    Deny(String),
    /// Defer to the harness: ask the user a question, then map the answer
    /// back to `Allow` or `Deny`. Resolution is done by the loop via the
    /// configured [`AskUserCallback`](super::AskUserCallback). When no
    /// callback is registered, the loop treats `AskUser` as `Deny`.
    AskUser {
        /// Prompt the harness should show the user.
        prompt: String,
    },
}
