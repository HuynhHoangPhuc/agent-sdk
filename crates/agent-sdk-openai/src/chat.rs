//! OpenAI **Chat Completions** request mapping and wire types.
//!
//! Endpoint: `POST /v1/chat/completions`. Streaming via `stream: true`.

use agent_sdk_language_model::{
    ContentBlock, LanguageModelRequest, Message, ModelError, Role, ToolChoice,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::models::WireUsage;

/// Provider-metadata key opting *out* of parallel tool calls.
/// Default per `Phase 1 Q7` is `parallel=true` (matches OpenAI default).
pub(crate) const PARALLEL_TOOL_CALLS_KEY: &str = "openai.parallel_tool_calls";

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ChatTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    pub stream: bool,
    pub stream_options: StreamOptions,
}

/// `stream_options.include_usage=true` so the final chunk carries `usage`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatMessage {
    pub role: &'static str,
    pub content: Value,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCallOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatToolCallOut {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ChatFunctionOut,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatFunctionOut {
    pub name: String,
    /// JSON-encoded arguments string (the wire format).
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatTool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ChatToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub(crate) fn build_request(
    model_id: &str,
    req: &LanguageModelRequest,
) -> Result<ChatRequest, ModelError> {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Hoist explicit `system` prompt into a leading system message.
    if let Some(s) = req.system.as_ref() {
        messages.push(ChatMessage {
            role: "system",
            content: Value::String(s.clone()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }
    for m in &req.messages {
        messages.extend(map_message(m)?);
    }

    let tools = req.tools.iter().map(map_tool).collect::<Vec<_>>();
    let tool_choice = map_tool_choice(&req.tool_choice)?;
    let parallel = req
        .provider_metadata
        .get(PARALLEL_TOOL_CALLS_KEY)
        .and_then(|v| v.as_bool());

    Ok(ChatRequest {
        model: model_id.to_string(),
        messages,
        tools,
        tool_choice,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        stop: req.stop.clone(),
        parallel_tool_calls: parallel,
        stream: true,
        stream_options: StreamOptions {
            include_usage: true,
        },
    })
}

fn map_message(msg: &Message) -> Result<Vec<ChatMessage>, ModelError> {
    match msg.role {
        Role::System => {
            let text = collect_text(&msg.content);
            Ok(vec![ChatMessage {
                role: "system",
                content: Value::String(text),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }])
        }
        Role::User => {
            let text = collect_text(&msg.content);
            Ok(vec![ChatMessage {
                role: "user",
                content: Value::String(text),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }])
        }
        Role::Assistant => {
            // Assistant may carry text + tool_use blocks. Tool calls become
            // `tool_calls` array; text concatenated as content (or empty).
            let mut tool_calls = Vec::new();
            let mut text = String::new();
            for b in &msg.content {
                match b {
                    ContentBlock::Text { text: t } => text.push_str(t),
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(ChatToolCallOut {
                            id: id.clone(),
                            kind: "function",
                            function: ChatFunctionOut {
                                name: name.clone(),
                                arguments: serde_json::to_string(input).map_err(|e| {
                                    ModelError::InvalidRequest(format!("encode tool args: {e}"))
                                })?,
                            },
                        });
                    }
                    ContentBlock::Reasoning { .. } => { /* drop — opaque to OpenAI Chat */ }
                    _ => {
                        return Err(ModelError::InvalidRequest(
                            "unsupported ContentBlock for OpenAI Chat assistant message".into(),
                        ))
                    }
                }
            }
            Ok(vec![ChatMessage {
                role: "assistant",
                content: if text.is_empty() {
                    Value::Null
                } else {
                    Value::String(text)
                },
                tool_calls,
                tool_call_id: None,
            }])
        }
        Role::Tool => {
            // Each ToolResult block becomes its own role:tool message.
            let mut out = Vec::new();
            for b in &msg.content {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = b
                {
                    let body = match content {
                        Value::String(s) => s.clone(),
                        v => v.to_string(),
                    };
                    out.push(ChatMessage {
                        role: "tool",
                        content: Value::String(body),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(tool_use_id.clone()),
                    });
                }
            }
            Ok(out)
        }
        _ => Err(ModelError::InvalidRequest(
            "unsupported Role variant for OpenAI Chat provider".into(),
        )),
    }
}

fn collect_text(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        if let ContentBlock::Text { text } = b {
            out.push_str(text);
        }
    }
    out
}

fn map_tool(spec: &agent_sdk_language_model::ToolSpec) -> ChatTool {
    ChatTool {
        kind: "function",
        function: ChatToolFunction {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters: spec.input_schema.clone(),
        },
    }
}

fn map_tool_choice(choice: &ToolChoice) -> Result<Option<Value>, ModelError> {
    Ok(match choice {
        ToolChoice::Auto => None,
        ToolChoice::Required => Some(Value::String("required".into())),
        ToolChoice::Tool { name } => {
            Some(json!({"type":"function","function":{"name": name}}))
        }
        ToolChoice::None => Some(Value::String("none".into())),
        _ => {
            return Err(ModelError::InvalidRequest(
                "unsupported ToolChoice variant for OpenAI Chat provider".into(),
            ))
        }
    })
}

// ---- Streaming wire types ----------------------------------------------

/// One `data:` chunk from the Chat Completions stream.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatStreamChunk {
    #[serde(default)]
    pub choices: Vec<ChatStreamChoice>,
    #[serde(default)]
    pub usage: Option<ChatStreamUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatStreamChoice {
    #[serde(default)]
    pub delta: ChatStreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatStreamDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ChatStreamToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatStreamToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChatStreamFn>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatStreamFn {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatStreamUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

impl ChatStreamUsage {
    pub(crate) fn into_wire(self) -> WireUsage {
        WireUsage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            cached_tokens: self
                .prompt_tokens_details
                .map(|d| d.cached_tokens)
                .unwrap_or(0),
            ..WireUsage::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_language_model::{Message, ToolSpec};

    fn req_with_user(text: &str) -> LanguageModelRequest {
        let mut r = LanguageModelRequest::default();
        r.messages.push(Message::user(text));
        r
    }

    #[test]
    fn hoists_system_prompt_as_leading_message() {
        let mut r = req_with_user("hi");
        r.system = Some("be terse".into());
        let built = build_request("gpt-x", &r).unwrap();
        assert_eq!(built.messages[0].role, "system");
        assert_eq!(built.messages[0].content, Value::String("be terse".into()));
        assert_eq!(built.messages[1].role, "user");
    }

    #[test]
    fn stream_options_include_usage_is_true() {
        let built = build_request("gpt-x", &req_with_user("hi")).unwrap();
        assert!(built.stream);
        assert!(built.stream_options.include_usage);
    }

    #[test]
    fn maps_tool_choice_variants() {
        let mut r = req_with_user("hi");
        r.tool_choice = ToolChoice::Required;
        assert_eq!(
            build_request("m", &r).unwrap().tool_choice.unwrap(),
            Value::String("required".into())
        );
        r.tool_choice = ToolChoice::Tool { name: "bash".into() };
        assert_eq!(
            build_request("m", &r).unwrap().tool_choice.unwrap(),
            json!({"type":"function","function":{"name":"bash"}})
        );
        r.tool_choice = ToolChoice::None;
        assert_eq!(
            build_request("m", &r).unwrap().tool_choice.unwrap(),
            Value::String("none".into())
        );
        r.tool_choice = ToolChoice::Auto;
        assert!(build_request("m", &r).unwrap().tool_choice.is_none());
    }

    #[test]
    fn parallel_tool_calls_opt_out_propagates() {
        let mut r = req_with_user("hi");
        r.provider_metadata
            .insert(PARALLEL_TOOL_CALLS_KEY.into(), Value::Bool(false));
        let built = build_request("m", &r).unwrap();
        assert_eq!(built.parallel_tool_calls, Some(false));
    }

    #[test]
    fn tools_map_to_function_shape() {
        let mut r = req_with_user("hi");
        r.tools
            .push(ToolSpec::new("echo", "d", json!({"type":"object"})));
        let built = build_request("m", &r).unwrap();
        assert_eq!(built.tools.len(), 1);
        assert_eq!(built.tools[0].kind, "function");
        assert_eq!(built.tools[0].function.name, "echo");
    }
}
