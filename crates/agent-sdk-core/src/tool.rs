//! The [`Tool`] trait and convenience adapters.
//!
//! `Tool` is intentionally **object-safe** (no generics, no `Self` in return
//! position) so an agent can hold `Vec<Arc<dyn Tool>>` and the v0.2 C-ABI
//! layer can pass tool handles across the FFI boundary.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::AgentError;

/// Output of a [`Tool::execute`] call.
///
/// `content` is the structured payload sent back to the model as the matching
/// [`ContentBlock::ToolResult`](agent_sdk_language_model::ContentBlock::ToolResult).
/// `is_error` lets a tool signal a failure to the model without raising it as
/// an [`AgentError`] that would tear down the run.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ToolResult {
    /// Structured tool output. Typically a string or JSON object.
    pub content: serde_json::Value,
    /// `true` if the tool wants the model to see this as a failure.
    pub is_error: bool,
}

impl ToolResult {
    /// Success result with a plain-text payload.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: serde_json::Value::String(text.into()),
            is_error: false,
        }
    }

    /// Success result with a structured JSON payload.
    pub fn json(value: serde_json::Value) -> Self {
        Self {
            content: value,
            is_error: false,
        }
    }

    /// Error result with a text payload; sent to the model as `is_error=true`.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: serde_json::Value::String(text.into()),
            is_error: true,
        }
    }
}

/// A tool the agent can invoke on the model's behalf.
///
/// Object-safe: `&self` receivers, no generics, no associated types, no GATs.
/// Implementations must be `Send + Sync` so they can be stored as
/// `Arc<dyn Tool>` and shared across tool-execution tasks.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name, advertised to the model. Must be unique within an [`Agent`](crate::Agent).
    fn name(&self) -> &str;

    /// Human-readable description shown to the model.
    fn description(&self) -> &str;

    /// JSON Schema (draft 2020-12) describing the tool's argument shape.
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool against the given arguments.
    ///
    /// Implementations MUST honour `cancel` promptly. Returning
    /// `Err(AgentError::Tool { .. })` aborts the run; returning `Ok(result)`
    /// with `result.is_error = true` reports the failure to the model and
    /// lets the loop continue.
    async fn execute(
        &self,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError>;
}

/// Closure adapter that turns an `async fn(args) -> Result<ToolResult,_>` into
/// a [`Tool`]. Useful for tests and for callers who do not want to define a
/// dedicated struct per tool. The full proc-macro derivation lands in Phase 4.
pub struct FnTool {
    name: String,
    description: String,
    schema: serde_json::Value,
    #[allow(clippy::type_complexity)]
    func: Arc<
        dyn Fn(
                serde_json::Value,
                CancellationToken,
            ) -> Pin<Box<dyn Future<Output = Result<ToolResult, AgentError>> + Send>>
            + Send
            + Sync,
    >,
}

impl FnTool {
    /// Build a [`FnTool`] from name, description, JSON Schema, and an async
    /// closure. The closure is moved into the tool and called for every
    /// invocation; arguments and the cancel token are passed through verbatim.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
        func: F,
    ) -> Self
    where
        F: Fn(serde_json::Value, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolResult, AgentError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            func: Arc::new(move |args, cancel| Box::pin(func(args, cancel))),
        }
    }
}

impl std::fmt::Debug for FnTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for FnTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<ToolResult, AgentError> {
        (self.func)(arguments, cancel).await
    }
}
