//! Chat-message and content-block primitives.
//!
//! These types are deliberately provider-agnostic. Provider-specific fields
//! (e.g. Anthropic cache_control, OpenAI logprobs) must live in
//! [`LanguageModelRequest::provider_metadata`](crate::LanguageModelRequest::provider_metadata)
//! or on the per-event payloads — never on these core structs.

use serde::{Deserialize, Serialize};

/// Author role for a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    /// System-level instructions. Most providers either accept this directly or
    /// fold it into a dedicated `system` field on the request.
    System,
    /// End-user input.
    User,
    /// Model output.
    Assistant,
    /// Result of a tool execution, sent back to the model.
    Tool,
}

/// A single message in a conversation. `content` carries one or more
/// [`ContentBlock`]s — multimodal/tool-use messages can interleave blocks of
/// different kinds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Message {
    /// Who authored this message.
    pub role: Role,
    /// Ordered list of content blocks that make up the message body.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Construct a plain-text user message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Construct a plain-text system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Construct a plain-text assistant message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(text)],
        }
    }
}

/// A single piece of content within a [`Message`].
///
/// `#[serde(tag = "type")]` is used so the wire form is the same shape most
/// providers already use (Anthropic, OpenAI tool-output blocks, Gemini parts).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentBlock {
    /// Plain UTF-8 text.
    Text {
        /// The text payload.
        text: String,
    },

    /// A tool/function the assistant has decided to invoke. The matching result
    /// is sent back as [`ContentBlock::ToolResult`] in a follow-up message.
    ToolUse {
        /// Provider-assigned identifier used to correlate the matching
        /// [`ContentBlock::ToolResult`].
        id: String,
        /// Tool name, matching one of the [`ToolSpec`](crate::ToolSpec)s the
        /// caller supplied in the request.
        name: String,
        /// JSON-encoded arguments for the tool.
        input: serde_json::Value,
    },

    /// The output of a previously requested tool call.
    ToolResult {
        /// The `id` of the originating [`ContentBlock::ToolUse`].
        tool_use_id: String,
        /// Tool output, typically text or JSON. Providers will serialize this
        /// in their native shape.
        content: serde_json::Value,
        /// `Some(true)` if the tool failed; `None` is treated as success.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        is_error: Option<bool>,
    },

    /// Reasoning trace (e.g. Anthropic extended thinking, OpenAI o-series
    /// reasoning summaries). Optional — providers without reasoning emit none.
    Reasoning {
        /// The reasoning text. Encrypted/redacted blocks are passed through
        /// verbatim by providers that need to round-trip them.
        text: String,
    },
}

impl ContentBlock {
    /// Convenience constructor for a text block.
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }
}
