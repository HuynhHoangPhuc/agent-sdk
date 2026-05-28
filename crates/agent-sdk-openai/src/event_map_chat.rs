//! Map Chat Completions SSE chunks into [`LanguageModelEvent`]s.
//!
//! Chat's stream model: each chunk carries `choices[0].delta` with optional
//! text or tool_call fragments, plus `finish_reason` on the *last* per-choice
//! chunk. `usage` arrives in a separate chunk with empty `choices` when
//! `stream_options.include_usage=true`.
//!
//! Tool-call deltas are indexed; `id` and `function.name` appear on the first
//! delta for each index, with subsequent deltas streaming `function.arguments`
//! fragments.

use std::collections::HashMap;

use agent_sdk_language_model::{FinishReason, LanguageModelEvent, Usage};

use crate::chat::{ChatStreamChunk, ChatStreamToolCall};

#[derive(Debug, Default)]
pub(crate) struct ChatEventMapper {
    /// Open tool-call ids by chunk index, so we can emit ToolCallEnd at the
    /// end of the stream and ignore deltas for unknown indices.
    open_tools: HashMap<u32, String>,
    finish_reason: Option<FinishReason>,
    usage: Usage,
    /// Suppress duplicate Finish if upstream sends both finish_reason then
    /// a separate usage-only chunk.
    finished: bool,
}

impl ChatEventMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map one SSE chunk into zero or more spec events.
    pub fn map(&mut self, chunk: ChatStreamChunk) -> Vec<LanguageModelEvent> {
        let mut out = Vec::new();
        if let Some(u) = chunk.usage {
            let spec = u.into_wire().into_spec();
            // Merge — usage chunk is authoritative.
            self.usage.input_tokens = spec.input_tokens.max(self.usage.input_tokens);
            self.usage.output_tokens = spec.output_tokens.max(self.usage.output_tokens);
            self.usage.cached_input_tokens =
                spec.cached_input_tokens.max(self.usage.cached_input_tokens);
        }
        for choice in chunk.choices {
            // Text deltas — drop empty content (e.g. role-only first chunk).
            if let Some(c) = choice.delta.content {
                if !c.is_empty() {
                    out.push(LanguageModelEvent::TextDelta { delta: c });
                }
            }
            for tc in choice.delta.tool_calls {
                self.handle_tool_call(tc, &mut out);
            }
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(map_finish_reason(&reason));
            }
        }
        out
    }

    /// Called when the byte stream ends; emits ToolCallEnd for any open
    /// tool calls and the terminal Finish event.
    pub fn flush(&mut self) -> Vec<LanguageModelEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut out = Vec::new();
        // Emit ToolCallEnd in index order so output is deterministic.
        let mut ids: Vec<(u32, String)> = self.open_tools.drain().collect();
        ids.sort_by_key(|(k, _)| *k);
        for (_, id) in ids {
            out.push(LanguageModelEvent::ToolCallEnd { id, arguments: None });
        }
        out.push(LanguageModelEvent::Finish {
            reason: self.finish_reason.unwrap_or(FinishReason::Stop),
            usage: self.usage,
        });
        out
    }

    fn handle_tool_call(&mut self, tc: ChatStreamToolCall, out: &mut Vec<LanguageModelEvent>) {
        let known = self.open_tools.contains_key(&tc.index);
        // First sighting: register id+name. Both must be present per OpenAI spec.
        if !known {
            let id = match tc.id {
                Some(i) => i,
                None => return, // ignore stray fragment
            };
            let name = tc
                .function
                .as_ref()
                .and_then(|f| f.name.clone())
                .unwrap_or_default();
            self.open_tools.insert(tc.index, id.clone());
            out.push(LanguageModelEvent::ToolCallStart { id, name });
        }
        if let Some(f) = tc.function {
            if let Some(args) = f.arguments {
                if !args.is_empty() {
                    let id = self
                        .open_tools
                        .get(&tc.index)
                        .cloned()
                        .unwrap_or_default();
                    out.push(LanguageModelEvent::ToolCallDelta {
                        id,
                        arguments_delta: args,
                    });
                }
            }
        }
    }
}

fn map_finish_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatStreamChoice, ChatStreamDelta, ChatStreamFn, ChatStreamUsage};
    use agent_sdk_language_model::LanguageModelEvent as E;

    fn chunk_with_delta(delta: ChatStreamDelta, finish: Option<&str>) -> ChatStreamChunk {
        ChatStreamChunk {
            choices: vec![ChatStreamChoice {
                delta,
                finish_reason: finish.map(str::to_string),
            }],
            usage: None,
        }
    }

    #[test]
    fn text_then_finish_emits_textdelta_and_finish() {
        let mut m = ChatEventMapper::new();
        let out = m.map(chunk_with_delta(
            ChatStreamDelta {
                content: Some("hi".into()),
                tool_calls: Vec::new(),
            },
            None,
        ));
        assert!(matches!(out.first(), Some(E::TextDelta { delta }) if delta == "hi"));

        let out = m.map(chunk_with_delta(ChatStreamDelta::default(), Some("stop")));
        assert!(out.is_empty());

        // Usage-only chunk (separate)
        let chunk = ChatStreamChunk {
            choices: Vec::new(),
            usage: Some(ChatStreamUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                prompt_tokens_details: None,
            }),
        };
        m.map(chunk);

        let flushed = m.flush();
        match flushed.last().unwrap() {
            E::Finish { reason, usage } => {
                assert_eq!(*reason, FinishReason::Stop);
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 5);
            }
            o => panic!("expected Finish, got {o:?}"),
        }
    }

    #[test]
    fn tool_call_lifecycle_start_delta_end_in_finish() {
        let mut m = ChatEventMapper::new();
        let chunk = chunk_with_delta(
            ChatStreamDelta {
                content: None,
                tool_calls: vec![ChatStreamToolCall {
                    index: 0,
                    id: Some("call_1".into()),
                    function: Some(ChatStreamFn {
                        name: Some("bash".into()),
                        arguments: Some("{\"cmd\":".into()),
                    }),
                }],
            },
            None,
        );
        let out = m.map(chunk);
        assert!(matches!(out[0], E::ToolCallStart { ref id, ref name } if id == "call_1" && name == "bash"));
        assert!(matches!(out[1], E::ToolCallDelta { ref id, .. } if id == "call_1"));

        let chunk = chunk_with_delta(
            ChatStreamDelta {
                content: None,
                tool_calls: vec![ChatStreamToolCall {
                    index: 0,
                    id: None,
                    function: Some(ChatStreamFn {
                        name: None,
                        arguments: Some("\"ls\"}".into()),
                    }),
                }],
            },
            Some("tool_calls"),
        );
        let out = m.map(chunk);
        assert!(matches!(out[0], E::ToolCallDelta { .. }));

        let flushed = m.flush();
        // ToolCallEnd then Finish(ToolUse)
        assert!(matches!(flushed[0], E::ToolCallEnd { ref id, .. } if id == "call_1"));
        assert!(matches!(flushed[1], E::Finish { reason: FinishReason::ToolUse, .. }));
    }
}
