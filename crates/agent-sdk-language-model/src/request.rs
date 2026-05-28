//! The single request type passed to [`LanguageModel::stream`](crate::LanguageModel::stream).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Message, ToolChoice, ToolSpec};

/// Input to a language-model invocation.
///
/// Designed as a closed, serde-stable struct so callers and providers can rely
/// on a single shared shape. Provider-specific knobs (Anthropic prompt cache
/// markers, OpenAI logprobs, Gemini safety settings) MUST be passed via
/// [`Self::provider_metadata`] rather than added as first-class fields here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LanguageModelRequest {
    /// Conversation history, oldest first.
    pub messages: Vec<Message>,

    /// Optional system prompt. Some providers fold this into the messages
    /// array; the spec keeps it separate so callers don't have to know which.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub system: Option<String>,

    /// Tools the model is allowed to call this turn.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolSpec>,

    /// How aggressively the model should pick tools. Defaults to [`ToolChoice::Auto`].
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// Sampling temperature in `[0.0, 2.0]`. Provider-clamped if out of range.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub temperature: Option<f32>,

    /// Hard upper bound on output tokens for this turn.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_tokens: Option<u32>,

    /// Stop sequences. Provider will halt generation if any sequence is emitted.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<String>,

    /// Provider-specific escape hatch. Keys are namespaced by provider id
    /// (e.g. `"anthropic.cache_control"`, `"openai.logprobs"`). Unknown keys
    /// MUST be ignored by providers that do not recognise them.
    ///
    /// `BTreeMap` is used (not `HashMap`) so the serialized form is stable
    /// — important for caching keys and snapshot tests.
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub provider_metadata: BTreeMap<String, serde_json::Value>,
}
