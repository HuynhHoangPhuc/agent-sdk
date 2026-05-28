//! OpenAI **Responses API** request mapping and wire types.
//!
//! Endpoint: `POST /v1/responses`. Streaming via `stream: true`. Events arrive
//! as *typed* SSE frames (`event:` line indicates the event name).

use agent_sdk_language_model::{
    ContentBlock, LanguageModelRequest, Message, ModelError, Role, ToolChoice,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// `POST /v1/responses` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ResponsesRequest {
    pub model: String,
    pub input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ResponsesTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    pub stream: bool,
    /// Whether to persist the response server-side (Responses API default is
    /// `true`; we set `false` to match Chat behaviour and avoid surprise storage).
    pub store: bool,
}

/// One item in the `input` array.
///
/// Three variants are emitted by v0.1's request mapper:
/// - `message` (role + content parts)
/// - `function_call` (assistant tool invocation)
/// - `function_call_output` (tool result)
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponsesInputItem {
    Message {
        role: &'static str,
        content: Vec<ResponsesContentPart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

/// Content part within a `message` item.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponsesContentPart {
    /// Used inside `role:user|system` messages.
    InputText { text: String },
    /// Used inside `role:assistant` messages.
    OutputText { text: String },
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ResponsesTool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub(crate) fn build_request(
    model_id: &str,
    req: &LanguageModelRequest,
) -> Result<ResponsesRequest, ModelError> {
    let mut input: Vec<ResponsesInputItem> = Vec::new();
    for m in &req.messages {
        input.extend(map_message(m)?);
    }

    let tools = req.tools.iter().map(map_tool).collect::<Vec<_>>();
    let tool_choice = map_tool_choice(&req.tool_choice)?;

    Ok(ResponsesRequest {
        model: model_id.to_string(),
        input,
        instructions: req.system.clone(),
        tools,
        tool_choice,
        temperature: req.temperature,
        max_output_tokens: req.max_tokens,
        stream: true,
        store: false,
    })
}

fn map_message(msg: &Message) -> Result<Vec<ResponsesInputItem>, ModelError> {
    match msg.role {
        Role::System => Ok(vec![ResponsesInputItem::Message {
            role: "system",
            content: vec![ResponsesContentPart::InputText {
                text: collect_text(&msg.content),
            }],
        }]),
        Role::User => Ok(vec![ResponsesInputItem::Message {
            role: "user",
            content: vec![ResponsesContentPart::InputText {
                text: collect_text(&msg.content),
            }],
        }]),
        Role::Assistant => {
            let mut out: Vec<ResponsesInputItem> = Vec::new();
            let mut parts: Vec<ResponsesContentPart> = Vec::new();
            for b in &msg.content {
                match b {
                    ContentBlock::Text { text } => parts.push(ResponsesContentPart::OutputText {
                        text: text.clone(),
                    }),
                    ContentBlock::ToolUse { id, name, input } => {
                        if !parts.is_empty() {
                            out.push(ResponsesInputItem::Message {
                                role: "assistant",
                                content: std::mem::take(&mut parts),
                            });
                        }
                        out.push(ResponsesInputItem::FunctionCall {
                            call_id: id.clone(),
                            name: name.clone(),
                            arguments: serde_json::to_string(input).map_err(|e| {
                                ModelError::InvalidRequest(format!("encode tool args: {e}"))
                            })?,
                        });
                    }
                    ContentBlock::Reasoning { .. } => { /* drop — encrypted-reasoning round-trip deferred */
                    }
                    _ => {
                        return Err(ModelError::InvalidRequest(
                            "unsupported ContentBlock for OpenAI Responses assistant message"
                                .into(),
                        ))
                    }
                }
            }
            if !parts.is_empty() {
                out.push(ResponsesInputItem::Message {
                    role: "assistant",
                    content: parts,
                });
            }
            Ok(out)
        }
        Role::Tool => {
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
                    out.push(ResponsesInputItem::FunctionCallOutput {
                        call_id: tool_use_id.clone(),
                        output: body,
                    });
                }
            }
            Ok(out)
        }
        _ => Err(ModelError::InvalidRequest(
            "unsupported Role variant for OpenAI Responses provider".into(),
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

fn map_tool(spec: &agent_sdk_language_model::ToolSpec) -> ResponsesTool {
    ResponsesTool {
        kind: "function",
        name: spec.name.clone(),
        description: spec.description.clone(),
        parameters: spec.input_schema.clone(),
    }
}

fn map_tool_choice(choice: &ToolChoice) -> Result<Option<Value>, ModelError> {
    Ok(match choice {
        ToolChoice::Auto => None,
        ToolChoice::Required => Some(Value::String("required".into())),
        ToolChoice::Tool { name } => Some(json!({"type":"function","name": name})),
        ToolChoice::None => Some(Value::String("none".into())),
        _ => {
            return Err(ModelError::InvalidRequest(
                "unsupported ToolChoice variant for OpenAI Responses provider".into(),
            ))
        }
    })
}

// ---- Streaming wire types ---------------------------------------------

/// Typed Responses stream events the mapper recognises.
///
/// The `event:` line on the SSE frame routes the JSON payload into one of
/// these variants via `EVENT_*` discriminants. Unrecognised events are
/// dropped by the mapper.
#[derive(Debug, Clone)]
pub(crate) enum ResponsesEvent {
    OutputItemAdded(OutputItemAdded),
    OutputTextDelta(OutputTextDelta),
    FunctionCallArgsDelta(FunctionCallArgsDelta),
    OutputItemDone(OutputItemDone),
    ReasoningSummaryDelta(ReasoningSummaryDelta),
    Completed(ResponseCompleted),
    Failed(ResponseFailed),
    Ignored,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OutputItemAdded {
    pub item: ResponsesOutputItem,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponsesOutputItem {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    /// `output_item.done` for `function_call` items carries the fully-assembled
    /// JSON-encoded `arguments` string. We round-trip it through `ToolCallEnd`
    /// so callers don't have to re-concatenate the per-fragment deltas.
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OutputTextDelta {
    pub delta: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FunctionCallArgsDelta {
    pub item_id: String,
    pub delta: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OutputItemDone {
    pub item: ResponsesOutputItem,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ReasoningSummaryDelta {
    pub delta: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponseCompleted {
    pub response: CompletedResponse,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CompletedResponse {
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct IncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponseFailed {
    pub response: FailedResponse,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FailedResponse {
    pub error: ResponsesError,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResponsesError {
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub code: Option<String>,
}

/// Parse `(event_name, json_data)` into a [`ResponsesEvent`].
///
/// Unknown event names yield [`ResponsesEvent::Ignored`] rather than erroring,
/// so the API can evolve without breaking the stream.
pub(crate) fn parse_event(event: &str, data: &str) -> Result<ResponsesEvent, serde_json::Error> {
    Ok(match event {
        "response.output_item.added" => {
            ResponsesEvent::OutputItemAdded(serde_json::from_str(data)?)
        }
        "response.output_text.delta" => {
            ResponsesEvent::OutputTextDelta(serde_json::from_str(data)?)
        }
        "response.function_call_arguments.delta" => {
            ResponsesEvent::FunctionCallArgsDelta(serde_json::from_str(data)?)
        }
        "response.output_item.done" => ResponsesEvent::OutputItemDone(serde_json::from_str(data)?),
        "response.reasoning_summary_text.delta" => {
            ResponsesEvent::ReasoningSummaryDelta(serde_json::from_str(data)?)
        }
        "response.completed" => ResponsesEvent::Completed(serde_json::from_str(data)?),
        "response.failed" => ResponsesEvent::Failed(serde_json::from_str(data)?),
        _ => ResponsesEvent::Ignored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_language_model::Message;

    fn req_with_user(text: &str) -> LanguageModelRequest {
        let mut r = LanguageModelRequest::default();
        r.messages.push(Message::user(text));
        r
    }

    #[test]
    fn system_goes_into_instructions_not_input() {
        let mut r = req_with_user("hi");
        r.system = Some("be terse".into());
        let built = build_request("gpt-x", &r).unwrap();
        assert_eq!(built.instructions.as_deref(), Some("be terse"));
        assert_eq!(built.input.len(), 1);
    }

    #[test]
    fn user_msg_maps_to_input_text_part() {
        let built = build_request("m", &req_with_user("hi")).unwrap();
        match &built.input[0] {
            ResponsesInputItem::Message { role, content } => {
                assert_eq!(*role, "user");
                assert!(matches!(content[0], ResponsesContentPart::InputText { ref text } if text == "hi"));
            }
            _ => panic!("expected message item"),
        }
    }

    #[test]
    fn store_defaults_to_false() {
        let built = build_request("m", &req_with_user("hi")).unwrap();
        assert!(!built.store);
        assert!(built.stream);
    }

    #[test]
    fn parses_known_event_types() {
        let ev = parse_event(
            "response.output_text.delta",
            r#"{"delta":"hello","item_id":"i_1","output_index":0,"content_index":0}"#,
        )
        .unwrap();
        match ev {
            ResponsesEvent::OutputTextDelta(d) => assert_eq!(d.delta, "hello"),
            _ => panic!("wrong variant"),
        }
        let ig = parse_event("response.something_new", "{}").unwrap();
        assert!(matches!(ig, ResponsesEvent::Ignored));
    }
}
