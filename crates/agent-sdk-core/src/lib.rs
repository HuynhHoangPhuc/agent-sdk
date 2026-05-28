//! # `agent-sdk-core`
//!
//! Streaming agent loop with tool calling, cancellation, parallel tool
//! execution, and a pluggable [`Session`] / [`SessionStore`]. Hooks and
//! permission policies (Phase 5), the proc-macro tool DSL (Phase 4), the
//! built-in coding toolkit (Phase 6), the MCP client (Phase 10), and
//! subagents (Phase 11) all build on top of the primitives defined here.
//!
//! ## FFI-safe public surface
//!
//! The public API obeys the FFI discipline (see `docs/ffi-safety-checklist.md`):
//!
//! - No `impl Stream` / GAT / HRTB at the public surface.
//! - [`AgentEventStream`] is a concrete `mpsc::Receiver`-backed wrapper.
//! - Public enums are `#[non_exhaustive]`.
//! - The builder takes owned values; no lifetimes leak into public types.
//! - The [`Tool`] trait is object-safe so agents hold `Arc<dyn Tool>`.
//!
//! ## Minimal example
//!
//! ```ignore
//! use agent_sdk_core::{Agent, FnTool, ToolResult};
//! use serde_json::json;
//!
//! # async fn demo() -> Result<(), agent_sdk_core::AgentError> {
//! let echo = FnTool::new(
//!     "echo",
//!     "echoes its input",
//!     serde_json::json!({"type": "object"}),
//!     |args, _cancel| async move {
//!         let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
//!         Ok(ToolResult::text(text))
//!     },
//! );
//!
//! let agent = Agent::builder()
//!     .model(/* impl LanguageModel */ todo!())
//!     .system("be terse")
//!     .tool(echo)
//!     .max_turns(4)
//!     .build()?;
//!
//! let out = agent.run("say hi").await?;
//! println!("{}", out.text);
//! # Ok(()) }
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod agent;
mod builder;
mod error;
mod event;
mod loop_;
mod session;
pub mod stop;
mod tool;

pub use crate::agent::Agent;
pub use crate::builder::AgentBuilder;
pub use crate::error::AgentError;
pub use crate::event::{AgentEvent, AgentEventStream, RunOutput};
pub use crate::session::{Session, SessionStore};
pub use crate::stop::{
    no_tool_calls, stop_count_is, And, LoopState, Never, NoToolCalls, Or, StopCondition,
    StopConditionExt, StopCountIs,
};
pub use crate::tool::{FnTool, Tool, ToolResult};
