//! Map [`LanguageModelRequest`] onto the Anthropic Messages wire request.

use agent_sdk_language_model::{
    ContentBlock, LanguageModelRequest, Message, ModelError, Role, ToolChoice,
};
use serde_json::{json, Value};

use crate::models::{MessagesRequest, WireContent, WireMessage, WireTool};

/// Provider-metadata key carrying a `cache_control` hint, e.g.
/// `{"anthropic.cache_control": {"type": "ephemeral"}}`.
///
/// v0.1 keeps the shape narrow: when present and non-null, the value is
/// attached as `cache_control` on the final text block of the **last
/// non-system message** in the conversation. Richer placement (per-message,
/// per-tool, per-system) is deferred to v0.2.
pub(crate) const CACHE_CONTROL_KEY: &str = "anthropic.cache_control";

/// Sensible default cap when the caller didn't pick one — Anthropic requires
/// `max_tokens` on every request.
const DEFAULT_MAX_TOKENS: u32 = 4096;

pub(crate) fn build_request(
    model_id: &str,
    req: &LanguageModelRequest,
) -> Result<MessagesRequest, ModelError> {
    let system = build_system(req);
    let tools = req.tools.iter().map(map_tool).collect::<Vec<_>>();
    let tool_choice = map_tool_choice(&req.tool_choice)?;
    let messages = map_messages(req)?;

    Ok(MessagesRequest {
        model: model_id.to_string(),
        messages,
        system,
        tools,
        tool_choice,
        temperature: req.temperature,
        max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        stop_sequences: req.stop.clone(),
        stream: true,
    })
}

fn build_system(req: &LanguageModelRequest) -> Option<Value> {
    // Hoist explicit `system` field and any system-role messages into a single
    // text-block array — Anthropic requires system as either a string or an
    // array of text blocks, not an inline message.
    let mut blocks: Vec<Value> = Vec::new();
    if let Some(s) = req.system.as_ref() {
        blocks.push(json!({ "type": "text", "text": s }));
    }
    for m in &req.messages {
        if m.role == Role::System {
            for b in &m.content {
                if let ContentBlock::Text { text } = b {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
            }
        }
    }
    if blocks.is_empty() {
        None
    } else {
        Some(Value::Array(blocks))
    }
}

fn map_tool(spec: &agent_sdk_language_model::ToolSpec) -> WireTool {
    WireTool {
        name: spec.name.clone(),
        description: spec.description.clone(),
        input_schema: spec.input_schema.clone(),
    }
}

fn map_tool_choice(choice: &ToolChoice) -> Result<Option<Value>, ModelError> {
    Ok(match choice {
        ToolChoice::Auto => None, // Anthropic default
        ToolChoice::Required => Some(json!({"type": "any"})),
        ToolChoice::Tool { name } => Some(json!({"type": "tool", "name": name})),
        ToolChoice::None => Some(json!({"type": "none"})),
        // ToolChoice is non_exhaustive — refuse to translate unknown variants
        // rather than silently misrouting via a default.
        _ => {
            return Err(ModelError::InvalidRequest(
                "unsupported ToolChoice variant for Anthropic provider".into(),
            ))
        }
    })
}

fn map_messages(req: &LanguageModelRequest) -> Result<Vec<WireMessage>, ModelError> {
    let cache_marker = req.provider_metadata.get(CACHE_CONTROL_KEY).and_then(|v| {
        if v.is_null() {
            None
        } else {
            Some(v.clone())
        }
    });

    // Locate the index of the last non-system message that contains a text
    // block, so we can attach cache_control to its final text block.
    let last_text_msg_idx = req
        .messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role != Role::System)
        .filter(|(_, m)| {
            m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text { .. }))
        })
        .map(|(i, _)| i)
        .next_back();

    let mut out: Vec<WireMessage> = Vec::with_capacity(req.messages.len());
    for (i, msg) in req.messages.iter().enumerate() {
        if msg.role == Role::System {
            continue;
        }
        let role = wire_role(msg)?;
        let mut content: Vec<WireContent> = Vec::with_capacity(msg.content.len());
        for (j, block) in msg.content.iter().enumerate() {
            let is_last_text_in_target = cache_marker.is_some()
                && Some(i) == last_text_msg_idx
                && matches!(block, ContentBlock::Text { .. })
                && msg
                    .content
                    .iter()
                    .enumerate()
                    .skip(j + 1)
                    .all(|(_, b)| !matches!(b, ContentBlock::Text { .. }));
            let cache_control = if is_last_text_in_target {
                cache_marker.clone()
            } else {
                None
            };
            content.push(map_block(block, cache_control)?);
        }
        out.push(WireMessage { role, content });
    }
    Ok(out)
}

fn wire_role(msg: &Message) -> Result<&'static str, ModelError> {
    match msg.role {
        Role::User => Ok("user"),
        Role::Assistant => Ok("assistant"),
        // Tool results travel inside a user message in Anthropic's protocol.
        Role::Tool => Ok("user"),
        Role::System => Err(ModelError::InvalidRequest(
            "system role messages must be hoisted before reaching map_messages".into(),
        )),
        // Role is non_exhaustive — refuse unknown variants so a future addition
        // can't silently masquerade as `user` on the wire.
        _ => Err(ModelError::InvalidRequest(
            "unsupported Role variant for Anthropic provider".into(),
        )),
    }
}

fn map_block(
    block: &ContentBlock,
    cache_control: Option<Value>,
) -> Result<WireContent, ModelError> {
    Ok(match block {
        ContentBlock::Text { text } => WireContent::Text {
            text: text.clone(),
            cache_control,
        },
        ContentBlock::ToolUse { id, name, input } => WireContent::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => WireContent::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
        },
        ContentBlock::Reasoning { text, signature } => WireContent::Thinking {
            thinking: text.clone(),
            signature: signature.clone(),
        },
        // ContentBlock is non_exhaustive — refuse unknown variants so callers
        // see a clear error rather than a silently-empty wire block.
        _ => {
            return Err(ModelError::InvalidRequest(
                "unsupported ContentBlock variant for Anthropic provider".into(),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_language_model::{Message, ToolSpec};
    use serde_json::json;

    fn req_with_user(text: &str) -> LanguageModelRequest {
        let mut req = LanguageModelRequest::default();
        req.messages.push(Message::user(text));
        req
    }

    #[test]
    fn hoists_system_into_top_level() {
        let mut req = req_with_user("hi");
        req.system = Some("be terse".into());
        let built = build_request("claude-sonnet-4-6", &req).unwrap();
        let sys = built.system.unwrap();
        assert_eq!(sys[0]["type"], "text");
        assert_eq!(sys[0]["text"], "be terse");
    }

    #[test]
    fn merges_system_role_messages_into_system() {
        let mut req = LanguageModelRequest::default();
        req.messages
            .extend([Message::system("extra rules"), Message::user("hi")]);
        req.system = Some("base".into());
        let built = build_request("claude-sonnet-4-6", &req).unwrap();
        let sys = built.system.unwrap();
        assert_eq!(sys.as_array().unwrap().len(), 2);
        assert_eq!(built.messages.len(), 1);
        assert_eq!(built.messages[0].role, "user");
    }

    #[test]
    fn cache_control_attaches_to_last_user_text_block() {
        let mut req = LanguageModelRequest::default();
        req.messages
            .extend([Message::user("first"), Message::user("second")]);
        req.provider_metadata
            .insert(CACHE_CONTROL_KEY.into(), json!({"type": "ephemeral"}));
        let built = build_request("claude-sonnet-4-6", &req).unwrap();
        let last = &built.messages[1].content[0];
        match last {
            WireContent::Text { cache_control, .. } => {
                assert_eq!(
                    cache_control.as_ref().unwrap(),
                    &json!({"type":"ephemeral"})
                );
            }
            _ => panic!("expected text"),
        }
        let first = &built.messages[0].content[0];
        match first {
            WireContent::Text { cache_control, .. } => assert!(cache_control.is_none()),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn maps_tool_choice_variants() {
        let mut req = req_with_user("hi");
        req.tool_choice = ToolChoice::Required;
        let built = build_request("m", &req).unwrap();
        assert_eq!(built.tool_choice.unwrap(), json!({"type":"any"}));

        req.tool_choice = ToolChoice::Tool {
            name: "bash".into(),
        };
        let built = build_request("m", &req).unwrap();
        assert_eq!(
            built.tool_choice.unwrap(),
            json!({"type":"tool","name":"bash"})
        );

        req.tool_choice = ToolChoice::None;
        let built = build_request("m", &req).unwrap();
        assert_eq!(built.tool_choice.unwrap(), json!({"type":"none"}));

        req.tool_choice = ToolChoice::Auto;
        let built = build_request("m", &req).unwrap();
        assert!(built.tool_choice.is_none());
    }

    #[test]
    fn max_tokens_defaults_when_unset() {
        let req = req_with_user("hi");
        let built = build_request("m", &req).unwrap();
        assert_eq!(built.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn tools_passthrough() {
        let mut req = req_with_user("hi");
        req.tools
            .push(ToolSpec::new("echo", "d", json!({"type":"object"})));
        let built = build_request("m", &req).unwrap();
        assert_eq!(built.tools.len(), 1);
        assert_eq!(built.tools[0].name, "echo");
    }
}
