//! OpenAI provider crate for the `agent-sdk` ecosystem.
//!
//! Implements [`LanguageModel`](agent_sdk_language_model::LanguageModel) against
//! two OpenAI APIs:
//!
//! * **Chat Completions** (`/v1/chat/completions`) — via [`chat`].
//! * **Responses API** (`/v1/responses`) — via [`responses`].
//!
//! Both share the same HTTP/SSE plumbing in [`client`](crate::client) +
//! [`sse`](crate::sse); only the request/event mapping differs per API.
//!
//! See `plans/260528-0939-rust-agent-harness-sdk-v0.1/phase-07-openai-provider.md`.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod chat;
mod client;
mod event_map_chat;
mod event_map_responses;
mod models;
mod responses;
mod sse;

pub use crate::client::{OpenAI, OpenAIApi, OpenAIConfig};

/// Default base URL for OpenAI's public API.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Construct an [`OpenAI`] bound to the **Chat Completions** API.
///
/// ```ignore
/// let m = agent_sdk_openai::chat("gpt-4o-mini", env_api_key)?;
/// ```
pub fn chat(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> Result<OpenAI, agent_sdk_language_model::ModelError> {
    OpenAI::chat(model_id, api_key)
}

/// Construct an [`OpenAI`] bound to the **Responses API**.
///
/// ```ignore
/// let m = agent_sdk_openai::responses("gpt-4o", env_api_key)?;
/// ```
pub fn responses(
    model_id: impl Into<String>,
    api_key: impl Into<String>,
) -> Result<OpenAI, agent_sdk_language_model::ModelError> {
    OpenAI::responses(model_id, api_key)
}
