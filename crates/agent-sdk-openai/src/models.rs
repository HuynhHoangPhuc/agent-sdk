//! Shared wire types and helpers across both OpenAI APIs.

use serde::Deserialize;

/// Per-request usage block returned by both Chat and Responses streams.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub(crate) struct WireUsage {
    /// Chat naming.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Chat naming.
    #[serde(default)]
    pub completion_tokens: u32,
    /// Responses naming.
    #[serde(default)]
    pub input_tokens: u32,
    /// Responses naming.
    #[serde(default)]
    pub output_tokens: u32,
    /// Chat: `prompt_tokens_details.cached_tokens`. Flattened into a peer field
    /// here via a per-API helper since serde flatten over nested types is noisy.
    #[serde(default)]
    pub cached_tokens: u32,
}

impl WireUsage {
    /// Convert to the spec's `Usage`, picking whichever naming the provider used.
    pub(crate) fn into_spec(self) -> agent_sdk_language_model::Usage {
        let input = if self.input_tokens > 0 {
            self.input_tokens
        } else {
            self.prompt_tokens
        };
        let output = if self.output_tokens > 0 {
            self.output_tokens
        } else {
            self.completion_tokens
        };
        let mut u = agent_sdk_language_model::Usage::default();
        u.input_tokens = input;
        u.output_tokens = output;
        u.cached_input_tokens = self.cached_tokens;
        u
    }
}
