//! Error type returned by the agent loop.

use agent_sdk_language_model::ModelError;
use thiserror::Error;

/// Errors surfaced by [`Agent::run`](crate::Agent::run) and
/// [`Agent::run_stream`](crate::Agent::run_stream).
///
/// Variants are intentionally flat (no nested error trees) so the v0.2 C-ABI
/// layer can map each variant onto a stable integer code without traversing a
/// type hierarchy. The enum is `#[non_exhaustive]` so new failure modes can be
/// added without a major bump.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AgentError {
    /// The underlying [`LanguageModel`](agent_sdk_language_model::LanguageModel)
    /// returned an error or emitted an [`Error`](agent_sdk_language_model::LanguageModelEvent::Error)
    /// event mid-stream.
    #[error("provider error: {0}")]
    Provider(#[from] ModelError),

    /// A [`Tool`](crate::Tool) execution failed or returned malformed output.
    #[error("tool '{name}' failed: {message}")]
    Tool {
        /// Tool name as advertised on its [`Tool::name`](crate::Tool::name) method.
        name: String,
        /// Human-readable failure detail.
        message: String,
    },

    /// A model emitted a tool-call argument stream whose concatenation did not
    /// parse as JSON.
    #[error("tool '{name}' produced invalid argument JSON: {message}")]
    InvalidToolArguments {
        /// Tool name the assistant tried to invoke.
        name: String,
        /// Parser error detail.
        message: String,
    },

    /// A hook returned `Halt`, stopping the loop intentionally.
    #[error("hook halted the loop: {0}")]
    HookHalt(String),

    /// A [`PermissionPolicy`](crate::PermissionPolicy) denied a tool call.
    #[error("permission denied for tool '{name}': {message}")]
    Permission {
        /// Tool name the assistant tried to invoke.
        name: String,
        /// Reason the policy gave for denial.
        message: String,
    },

    /// Reserved for the MCP client (Phase 10).
    #[error("mcp error: {0}")]
    Mcp(String),

    /// The loop hit its [`max_turns`](crate::AgentBuilder::max_turns) ceiling
    /// before terminating naturally. The [`RunOutput`](crate::RunOutput) on the
    /// caller side still holds whatever transcript was produced.
    #[error("reached max_turns ({0})")]
    MaxTurns(u32),

    /// Caller cancelled the run via the supplied cancellation token.
    #[error("agent run cancelled")]
    Cancelled,

    /// The builder rejected a configuration choice (e.g., no model supplied).
    #[error("invalid agent configuration: {0}")]
    InvalidConfig(String),

    /// Catch-all for anything that does not fit the variants above.
    #[error("{0}")]
    Other(String),
}
