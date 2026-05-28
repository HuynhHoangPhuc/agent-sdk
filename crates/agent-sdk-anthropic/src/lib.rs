//! Anthropic Messages API provider for the `agent-sdk` ecosystem.
//!
//! Implements the [`LanguageModel`](agent_sdk_language_model::LanguageModel)
//! trait against Anthropic's `/v1/messages` endpoint with server-sent-events
//! streaming. This crate is the **reference provider** the spec is validated
//! against — see `plans/260528-0939-rust-agent-harness-sdk-v0.1/phase-02-*.md`.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod client;
mod event_map;
mod models;
mod request_map;
mod sse;

pub use crate::client::{Anthropic, AnthropicConfig};

/// Default base URL for Anthropic's public API.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic API version header value the provider sends with every request.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Construct an [`Anthropic`] model bound to `claude-sonnet-4-6`.
///
/// This is a convenience wrapper around [`Anthropic::new`] for the model the
/// internal harness defaults to. Other model ids can be passed directly:
///
/// ```ignore
/// let m = agent_sdk_anthropic::Anthropic::new("claude-opus-4-7", env_api_key);
/// ```
pub fn claude_sonnet_4_6(api_key: impl Into<String>) -> Anthropic {
    Anthropic::new("claude-sonnet-4-6", api_key)
}
