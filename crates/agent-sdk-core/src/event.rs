//! Public event stream emitted by the agent loop.
//!
//! Events are pushed onto a [`tokio::sync::mpsc`] channel by the loop task and
//! delivered to the caller via [`AgentEventStream`] — a thin wrapper that
//! implements [`futures_core::Stream`] for Rust ergonomics while keeping the
//! underlying type concrete (no `impl Stream`/GAT/HRTB) so the v0.2 C-ABI
//! layer can wrap it without API breakage.

use std::pin::Pin;
use std::task::{Context, Poll};

use agent_sdk_language_model::{FinishReason, Usage};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use crate::AgentError;

/// A single event delivered to the caller while the agent loop is running.
///
/// Variants are `#[non_exhaustive]` so new event kinds (e.g. compaction
/// progress, sub-agent transitions) can be added without a major bump.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentEvent {
    /// A fragment of assistant text.
    TextDelta {
        /// Turn index, zero-based.
        turn: u32,
        /// Text to append to the in-progress assistant message.
        delta: String,
    },

    /// A fragment of reasoning trace (provider extended-thinking).
    ReasoningDelta {
        /// Turn index, zero-based.
        turn: u32,
        /// Reasoning fragment.
        delta: String,
    },

    /// The assistant has begun a tool call. Argument JSON arrives via
    /// [`Self::ToolCallEnd`] once finalized — argument deltas are buffered
    /// internally rather than emitted as events to keep the public stream
    /// focused on high-level transitions.
    ToolCallStart {
        /// Turn index, zero-based.
        turn: u32,
        /// Provider-assigned tool-call id.
        id: String,
        /// Tool name the model is invoking.
        name: String,
    },

    /// Tool-call arguments have been fully accumulated.
    ToolCallEnd {
        /// Turn index, zero-based.
        turn: u32,
        /// Provider-assigned tool-call id.
        id: String,
        /// Tool name the model invoked.
        name: String,
        /// Final parsed arguments.
        arguments: serde_json::Value,
    },

    /// A tool finished executing.
    ToolResult {
        /// Turn index, zero-based.
        turn: u32,
        /// Provider-assigned tool-call id this result corresponds to.
        id: String,
        /// Tool name that produced the result.
        name: String,
        /// Tool output (text, JSON, or structured content).
        content: serde_json::Value,
        /// True if the tool reported a failure.
        is_error: bool,
    },

    /// A single model turn finished.
    TurnEnd {
        /// Turn index, zero-based.
        turn: u32,
        /// Why the provider stopped this turn.
        reason: FinishReason,
        /// Token usage for this turn.
        usage: Usage,
    },

    /// The entire run finished successfully.
    Finish {
        /// Total number of turns executed (1-based count).
        turns: u32,
        /// Reason for terminating the loop (provider stop, no tool calls,
        /// [`stop_when`](crate::AgentBuilder::stop_when), or `max_turns`).
        reason: FinishReason,
        /// Aggregate token usage across all turns.
        usage: Usage,
    },

    /// Terminal error event. No further events will follow.
    Error {
        /// Stringified error message. The same error is also returned from
        /// [`run`](crate::Agent::run) / surfaced after the stream closes.
        message: String,
    },
}

/// FFI-safe stream of [`AgentEvent`]s produced by
/// [`Agent::run_stream`](crate::Agent::run_stream).
///
/// Wraps a [`tokio::sync::mpsc::Receiver`]; the agent loop runs on a spawned
/// task feeding the sender. The wrapper implements [`Stream`] so Rust callers
/// can use `while let Some(event) = stream.next().await { ... }`; the v0.2
/// C-ABI layer pulls via [`Self::recv`] directly. After the stream closes,
/// [`Self::final_result`] yields the typed terminal status (success or the
/// concrete [`AgentError`] variant the loop hit).
#[derive(Debug)]
pub struct AgentEventStream {
    rx: mpsc::Receiver<AgentEvent>,
    finish: Option<oneshot::Receiver<Result<(), AgentError>>>,
}

impl AgentEventStream {
    /// Construct from a receiver. Used internally by the loop task.
    pub(crate) fn new(
        rx: mpsc::Receiver<AgentEvent>,
        finish: oneshot::Receiver<Result<(), AgentError>>,
    ) -> Self {
        Self {
            rx,
            finish: Some(finish),
        }
    }

    /// Block-pull the next event. Returns `None` once the loop has finished
    /// (sender dropped). Exposed for C-ABI wrappers; Rust users prefer
    /// `StreamExt::next`.
    pub async fn recv(&mut self) -> Option<AgentEvent> {
        self.rx.recv().await
    }

    /// Await the typed terminal status of the run. Must be called after the
    /// event stream has been drained (otherwise the loop is still running).
    /// Subsequent calls after the first return [`AgentError::Other`] with a
    /// "result already consumed" message.
    pub async fn final_result(&mut self) -> Result<(), AgentError> {
        match self.finish.take() {
            Some(rx) => match rx.await {
                Ok(res) => res,
                Err(_) => Err(AgentError::Other(
                    "agent loop dropped without final result".into(),
                )),
            },
            None => Err(AgentError::Other(
                "agent loop final result already consumed".into(),
            )),
        }
    }
}

impl Stream for AgentEventStream {
    type Item = AgentEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

/// Outcome of a non-streaming [`Agent::run`](crate::Agent::run) invocation:
/// the collected events plus a final summary.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RunOutput {
    /// Every event that was emitted, in order.
    pub events: Vec<AgentEvent>,
    /// Final aggregated text (concatenation of every [`AgentEvent::TextDelta`]).
    pub text: String,
    /// Total turns executed.
    pub turns: u32,
    /// Why the loop ended.
    pub reason: FinishReason,
    /// Aggregate token usage across the run.
    pub usage: Usage,
}

/// Aggregate two [`Usage`] values.
pub(crate) fn add_usage(a: Usage, b: Usage) -> Usage {
    let mut out = Usage::default();
    out.input_tokens = a.input_tokens.saturating_add(b.input_tokens);
    out.output_tokens = a.output_tokens.saturating_add(b.output_tokens);
    out.cached_input_tokens = a.cached_input_tokens.saturating_add(b.cached_input_tokens);
    out
}

/// Map an [`AgentError`] onto a terminal [`AgentEvent::Error`].
pub(crate) fn error_event(err: &AgentError) -> AgentEvent {
    AgentEvent::Error {
        message: err.to_string(),
    }
}
