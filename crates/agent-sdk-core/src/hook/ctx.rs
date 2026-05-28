//! Event payload + outcome for the [`Hook`](super::Hook) trait.
//!
//! `HookEvent` carries every interesting transition the loop wants to expose:
//! the user prompt being submitted, the pre/post-model boundary, every tool
//! invocation, terminal errors, and the loop's final summary. `HookOutcome` is
//! the verb the hook returns — pass, swap the request, deny, or halt.

use agent_sdk_language_model::{FinishReason, LanguageModelRequest, Message, Usage};
use serde::Serialize;
use serde_json::Value;

use crate::tool::ToolResult;

/// Lifecycle position at which a hook runs.
///
/// Each variant carries the data the loop has on hand at that point. The enum
/// is `#[non_exhaustive]` so additional events (e.g. compaction, subagent
/// transitions) can ship in later phases without a major bump.
///
/// Only `Serialize` is derived — `HookEvent`/`HookCtx` flow *from* the loop to
/// hooks (notably to the subprocess stdin of `ExternalCommandHook`). The
/// reverse direction is the much smaller [`HookOutcome`], which the external
/// hook handles via its own wire shape (see `external_command.rs`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum HookEvent {
    /// User input was just appended to the session, before any model call.
    /// Fires at most once per run, at the very start, and only when the
    /// caller passed a non-empty input string. Session-seeded runs that pass
    /// an empty `input` (e.g. resuming an existing conversation) skip this
    /// event — there is no user prompt to observe.
    UserPromptSubmit {
        /// Raw user input string passed to `Agent::run` / `Agent::run_stream`.
        input: String,
    },

    /// About to call the language model. The hook may replace the request via
    /// [`HookOutcome::ModifyRequest`].
    PreModel {
        /// Request the loop is about to send. Hooks may modify and return a
        /// new value via [`HookOutcome::ModifyRequest`].
        request: LanguageModelRequest,
    },

    /// Model returned an assistant message; not yet appended to the session.
    PostModel {
        /// Assistant message produced this turn.
        message: Message,
        /// Token usage reported for this turn.
        usage: Usage,
    },

    /// About to execute a tool. Permission policy and any hook denial here can
    /// short-circuit the call.
    PreToolUse {
        /// Provider-assigned tool-call id.
        tool_use_id: String,
        /// Tool name being invoked.
        name: String,
        /// Final parsed arguments.
        arguments: Value,
    },

    /// Tool execution finished (or was denied with a synthesized result).
    PostToolUse {
        /// Provider-assigned tool-call id.
        tool_use_id: String,
        /// Tool name that ran.
        name: String,
        /// Tool result as it will be sent back to the model.
        #[serde(serialize_with = "serialize_tool_result")]
        result: ToolResult,
    },

    /// The loop hit an error and is about to surface it to the caller.
    OnError {
        /// Stringified error message.
        message: String,
    },

    /// The loop finished successfully and is about to emit `Finish`.
    OnFinish {
        /// Total turns executed (1-based count).
        turns: u32,
        /// Reason the loop ended.
        reason: FinishReason,
        /// Aggregate usage across the run.
        usage: Usage,
    },
}

/// Ambient context every hook invocation receives.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct HookCtx {
    /// Turn index at the moment the hook fires (0-based).
    pub turn: u32,
    /// Provider id of the active model, e.g. `"anthropic"`.
    pub provider_id: String,
    /// Model id, e.g. `"claude-opus-4-7"`.
    pub model_id: String,
    /// Optional session id taken from `Session::id`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_id: Option<String>,
    /// Event-specific payload.
    pub event: HookEvent,
}

/// Verb returned by a hook.
///
/// `ModifyRequest` is only honoured for [`HookEvent::PreModel`]; the loop
/// ignores it for every other event.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HookOutcome {
    /// Carry on as normal.
    Continue,
    /// Replace the in-flight `LanguageModelRequest` (only meaningful in
    /// [`HookEvent::PreModel`]).
    ModifyRequest(Box<LanguageModelRequest>),
    /// Block this tool call with a reason. Surfaced to the model as a
    /// tool-result error when used at [`HookEvent::PreToolUse`]; surfaced as
    /// `AgentError::HookHalt` for non-tool events.
    Deny(String),
    /// Stop the entire run with `AgentError::HookHalt`.
    Halt(String),
}

fn serialize_tool_result<S>(
    result: &ToolResult,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(2))?;
    map.serialize_entry("content", &result.content)?;
    map.serialize_entry("is_error", &result.is_error)?;
    map.end()
}
