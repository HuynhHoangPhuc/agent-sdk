//! Map Anthropic streaming events into `LanguageModelEvent`s.
//!
//! Anthropic emits text/tool/thinking content via a *block-indexed* protocol:
//! `content_block_start` opens an indexed block, `content_block_delta` carries
//! incremental updates, `content_block_stop` closes it. We translate this to
//! the spec's flat event stream by tracking per-index block metadata.

use std::collections::HashMap;

use agent_sdk_language_model::{FinishReason, LanguageModelEvent, Usage};

use crate::models::{StreamContentBlock, StreamDelta, StreamEvent, StreamUsage};

/// Translator state that accumulates partial usage and tracks active blocks
/// so deltas can be mapped to the right kind (text vs. tool_use vs. thinking).
#[derive(Debug, Default)]
pub(crate) struct EventMapper {
    blocks: HashMap<u32, BlockKind>,
    /// Final accounted usage; updated by message_start + message_delta.
    usage: Usage,
    /// Latest stop reason from message_delta, if any.
    stop_reason: Option<String>,
}

#[derive(Debug, Clone)]
enum BlockKind {
    Text,
    ToolUse { id: String },
    Thinking,
}

impl EventMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map one Anthropic stream event into zero or more spec events.
    pub fn map(&mut self, ev: StreamEvent) -> Vec<LanguageModelEvent> {
        match ev {
            StreamEvent::MessageStart { message } => {
                if let Some(u) = message.usage {
                    self.merge_usage(&u);
                }
                Vec::new()
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                StreamContentBlock::Text { text } => {
                    self.blocks.insert(index, BlockKind::Text);
                    if text.is_empty() {
                        Vec::new()
                    } else {
                        vec![LanguageModelEvent::TextDelta { delta: text }]
                    }
                }
                StreamContentBlock::ToolUse { id, name, .. } => {
                    self.blocks
                        .insert(index, BlockKind::ToolUse { id: id.clone() });
                    vec![LanguageModelEvent::ToolCallStart { id, name }]
                }
                StreamContentBlock::Thinking { thinking } => {
                    self.blocks.insert(index, BlockKind::Thinking);
                    if thinking.is_empty() {
                        Vec::new()
                    } else {
                        vec![LanguageModelEvent::ReasoningDelta { delta: thinking }]
                    }
                }
                StreamContentBlock::Other => Vec::new(),
            },
            StreamEvent::ContentBlockDelta { index, delta } => {
                match (self.blocks.get(&index), delta) {
                    (Some(BlockKind::Text), StreamDelta::TextDelta { text }) => {
                        vec![LanguageModelEvent::TextDelta { delta: text }]
                    }
                    (
                        Some(BlockKind::ToolUse { id }),
                        StreamDelta::InputJsonDelta { partial_json },
                    ) => {
                        vec![LanguageModelEvent::ToolCallDelta {
                            id: id.clone(),
                            arguments_delta: partial_json,
                        }]
                    }
                    (Some(BlockKind::Thinking), StreamDelta::ThinkingDelta { thinking }) => {
                        vec![LanguageModelEvent::ReasoningDelta { delta: thinking }]
                    }
                    (Some(BlockKind::Thinking), StreamDelta::SignatureDelta { signature }) => {
                        vec![LanguageModelEvent::ReasoningSignature { signature }]
                    }
                    _ => Vec::new(),
                }
            }
            StreamEvent::ContentBlockStop { index } => match self.blocks.remove(&index) {
                Some(BlockKind::ToolUse { id }) => vec![LanguageModelEvent::ToolCallEnd {
                    id,
                    arguments: None,
                }],
                _ => Vec::new(),
            },
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = delta.stop_reason {
                    self.stop_reason = Some(reason);
                }
                if let Some(u) = usage {
                    self.merge_usage(&u);
                }
                Vec::new()
            }
            StreamEvent::MessageStop => {
                let reason = map_stop_reason(self.stop_reason.as_deref());
                vec![LanguageModelEvent::Finish {
                    reason,
                    usage: self.usage,
                }]
            }
            StreamEvent::Ping | StreamEvent::Other => Vec::new(),
            StreamEvent::Error { error } => vec![LanguageModelEvent::Error {
                message: format!("{}: {}", error.kind, error.message),
            }],
        }
    }

    fn merge_usage(&mut self, u: &StreamUsage) {
        // Anthropic reports cumulative usage; later events refine the totals
        // (message_delta carries the final output_tokens). Take max so a
        // partial message_start doesn't get overwritten by a zero default.
        self.usage.input_tokens = self.usage.input_tokens.max(u.input_tokens);
        self.usage.output_tokens = self.usage.output_tokens.max(u.output_tokens);
        self.usage.cached_input_tokens = self.usage.cached_input_tokens.max(
            u.cache_read_input_tokens
                .saturating_add(u.cache_creation_input_tokens),
        );
    }
}

fn map_stop_reason(s: Option<&str>) -> FinishReason {
    match s {
        Some("end_turn") | Some("stop_sequence") | None => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolUse,
        Some("refusal") => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{StreamMessageDelta, StreamMessageStart};
    use agent_sdk_language_model::LanguageModelEvent as E;

    #[test]
    fn text_lifecycle_emits_deltas_then_finish() {
        let mut m = EventMapper::new();
        let out = m.map(StreamEvent::MessageStart {
            message: StreamMessageStart {
                usage: Some(StreamUsage {
                    input_tokens: 10,
                    ..Default::default()
                }),
            },
        });
        assert!(out.is_empty());
        let out = m.map(StreamEvent::ContentBlockStart {
            index: 0,
            content_block: StreamContentBlock::Text {
                text: String::new(),
            },
        });
        assert!(out.is_empty());
        let out = m.map(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: StreamDelta::TextDelta {
                text: "hello".into(),
            },
        });
        assert!(matches!(out.first(), Some(E::TextDelta { delta }) if delta == "hello"));
        let out = m.map(StreamEvent::ContentBlockStop { index: 0 });
        assert!(out.is_empty());
        let out = m.map(StreamEvent::MessageDelta {
            delta: StreamMessageDelta {
                stop_reason: Some("end_turn".into()),
            },
            usage: Some(StreamUsage {
                output_tokens: 5,
                ..Default::default()
            }),
        });
        assert!(out.is_empty());
        let out = m.map(StreamEvent::MessageStop);
        match out.first().unwrap() {
            E::Finish { reason, usage } => {
                assert_eq!(*reason, FinishReason::Stop);
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected finish, got {:?}", other),
        }
    }

    #[test]
    fn tool_use_lifecycle_emits_start_delta_end() {
        let mut m = EventMapper::new();
        let out = m.map(StreamEvent::ContentBlockStart {
            index: 1,
            content_block: StreamContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "bash".into(),
                input: serde_json::Value::Null,
            },
        });
        assert!(
            matches!(out.first(), Some(E::ToolCallStart { id, name }) if id == "toolu_1" && name == "bash")
        );

        let out = m.map(StreamEvent::ContentBlockDelta {
            index: 1,
            delta: StreamDelta::InputJsonDelta {
                partial_json: "{\"cmd\":".into(),
            },
        });
        assert!(matches!(out.first(), Some(E::ToolCallDelta { .. })));

        let out = m.map(StreamEvent::ContentBlockStop { index: 1 });
        assert!(matches!(out.first(), Some(E::ToolCallEnd { id, .. }) if id == "toolu_1"));
    }

    #[test]
    fn stop_reason_tool_use_maps_to_tooluse_finish() {
        let mut m = EventMapper::new();
        m.map(StreamEvent::MessageDelta {
            delta: StreamMessageDelta {
                stop_reason: Some("tool_use".into()),
            },
            usage: None,
        });
        let out = m.map(StreamEvent::MessageStop);
        match out.first().unwrap() {
            E::Finish { reason, .. } => assert_eq!(*reason, FinishReason::ToolUse),
            _ => panic!(),
        }
    }

    #[test]
    fn unknown_event_is_dropped() {
        let mut m = EventMapper::new();
        assert!(m.map(StreamEvent::Ping).is_empty());
        assert!(m.map(StreamEvent::Other).is_empty());
    }
}
