//! Wire types for the Anthropic Messages API.
//!
//! Only the subset the provider needs for v0.1 is modeled. Unknown fields are
//! tolerated via `serde(default)` / `flatten` where reasonable so minor API
//! additions don't break the parser.

use serde::{Deserialize, Serialize};

/// Top-level request body sent to `POST /v1/messages`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct MessagesRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}

/// Anthropic-side chat message (only `user` / `assistant` roles are valid;
/// `system` is hoisted to the top-level `system` field by `request_map`).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireMessage {
    pub role: &'static str,
    pub content: Vec<WireContent>,
}

/// Content block as serialized to the Anthropic wire.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireContent {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Thinking {
        thinking: String,
    },
}

/// Tool definition as Anthropic expects it.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// ---- Streaming event wire types -----------------------------------------

/// Anthropic SSE event payload (`data:` JSON line).
///
/// The `type` discriminant is the SSE `event:` line for cross-referencing
/// but Anthropic always echoes the same string inside the JSON body, which is
/// what we deserialize against here.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum StreamEvent {
    MessageStart {
        message: StreamMessageStart,
    },
    ContentBlockStart {
        index: u32,
        content_block: StreamContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: StreamDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: StreamMessageDelta,
        #[serde(default)]
        usage: Option<StreamUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: StreamError,
    },
    /// Unknown event — Anthropic occasionally adds new event types
    /// (e.g. `tool_use`, server notes); we forward them as opaque.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamMessageStart {
    #[serde(default)]
    pub usage: Option<StreamUsage>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum StreamContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        // Documented for wire fidelity; arguments arrive via `input_json_delta`.
        #[serde(default)]
        #[allow(dead_code)]
        input: serde_json::Value,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum StreamDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamMessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct StreamUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamError {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}
