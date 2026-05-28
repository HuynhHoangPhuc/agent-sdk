//! Map Responses API typed events into [`LanguageModelEvent`]s.
//!
//! Responses streams a sequence of typed events identifying output items
//! (message / function_call / reasoning) and their per-part deltas. We track
//! per-item kind so deltas dispatch correctly and so `output_item.done` can
//! emit the right terminator (e.g. `ToolCallEnd` for function calls).

use std::collections::HashMap;

use agent_sdk_language_model::{FinishReason, LanguageModelEvent, Usage};

use crate::responses::{
    FunctionCallArgsDelta, OutputItemAdded, OutputItemDone, OutputTextDelta, ReasoningSummaryDelta,
    ResponseCompleted, ResponseFailed, ResponsesEvent,
};

#[derive(Debug, Clone)]
enum ItemKind {
    Message,
    FunctionCall { call_id: String },
    Reasoning,
    Other,
}

#[derive(Debug, Default)]
pub(crate) struct ResponsesEventMapper {
    items: HashMap<String, ItemKind>,
    /// Set when a function_call item was seen this turn — used so the terminal
    /// `Finish` can classify `ToolUse` even after `item_done` removed it.
    had_function_call: bool,
    finished: bool,
}

impl ResponsesEventMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map one parsed event into zero or more spec events.
    pub fn map(&mut self, ev: ResponsesEvent) -> Vec<LanguageModelEvent> {
        match ev {
            ResponsesEvent::OutputItemAdded(OutputItemAdded { item }) => {
                let id = match item.id.clone() {
                    Some(i) => i,
                    None => return Vec::new(),
                };
                match item.kind.as_str() {
                    "message" => {
                        self.items.insert(id, ItemKind::Message);
                        Vec::new()
                    }
                    "function_call" => {
                        let call_id = item.call_id.clone().unwrap_or_else(|| id.clone());
                        let name = item.name.clone().unwrap_or_default();
                        self.items.insert(
                            id,
                            ItemKind::FunctionCall {
                                call_id: call_id.clone(),
                            },
                        );
                        self.had_function_call = true;
                        vec![LanguageModelEvent::ToolCallStart { id: call_id, name }]
                    }
                    "reasoning" => {
                        self.items.insert(id, ItemKind::Reasoning);
                        Vec::new()
                    }
                    _ => {
                        self.items.insert(id, ItemKind::Other);
                        Vec::new()
                    }
                }
            }
            ResponsesEvent::OutputTextDelta(OutputTextDelta { delta }) => {
                if delta.is_empty() {
                    Vec::new()
                } else {
                    vec![LanguageModelEvent::TextDelta { delta }]
                }
            }
            ResponsesEvent::FunctionCallArgsDelta(FunctionCallArgsDelta { item_id, delta }) => {
                match self.items.get(&item_id) {
                    Some(ItemKind::FunctionCall { call_id }) => {
                        vec![LanguageModelEvent::ToolCallDelta {
                            id: call_id.clone(),
                            arguments_delta: delta,
                        }]
                    }
                    _ => Vec::new(),
                }
            }
            ResponsesEvent::ReasoningSummaryDelta(ReasoningSummaryDelta { delta }) => {
                if delta.is_empty() {
                    Vec::new()
                } else {
                    vec![LanguageModelEvent::ReasoningDelta { delta }]
                }
            }
            ResponsesEvent::OutputItemDone(OutputItemDone { item }) => {
                let id = match item.id.clone() {
                    Some(i) => i,
                    None => return Vec::new(),
                };
                let args = item.arguments.clone();
                match self.items.remove(&id) {
                    Some(ItemKind::FunctionCall { call_id }) => {
                        // Parse the fully-assembled arguments string if present
                        // so consumers can avoid concatenating per-fragment deltas.
                        // Malformed JSON falls through to `None` — the deltas are
                        // still authoritative for any caller that reassembles.
                        let parsed = args
                            .as_deref()
                            .and_then(|s| serde_json::from_str(s).ok());
                        vec![LanguageModelEvent::ToolCallEnd {
                            id: call_id,
                            arguments: parsed,
                        }]
                    }
                    _ => Vec::new(),
                }
            }
            ResponsesEvent::Completed(ResponseCompleted { response }) => {
                self.finished = true;
                let usage = response
                    .usage
                    .map(|u| {
                        let mut spec = Usage::default();
                        spec.input_tokens = u.input_tokens;
                        spec.output_tokens = u.output_tokens;
                        spec.cached_input_tokens =
                            u.input_tokens_details.map(|d| d.cached_tokens).unwrap_or(0);
                        spec
                    })
                    .unwrap_or_default();
                let reason = if let Some(d) = response.incomplete_details {
                    map_incomplete(&d.reason)
                } else if response.status.as_deref() == Some("incomplete") {
                    FinishReason::Length
                } else if self.had_function_call {
                    FinishReason::ToolUse
                } else {
                    FinishReason::Stop
                };
                vec![LanguageModelEvent::Finish { reason, usage }]
            }
            ResponsesEvent::Failed(ResponseFailed { response }) => {
                self.finished = true;
                let msg = if let Some(code) = response.error.code {
                    format!("{}: {}", code, response.error.message)
                } else {
                    response.error.message
                };
                vec![LanguageModelEvent::Error { message: msg }]
            }
            ResponsesEvent::Ignored => Vec::new(),
        }
    }

    /// Called when the byte stream ends; emit a synthetic Finish if upstream
    /// never sent `response.completed` / `response.failed`.
    pub fn flush(&mut self) -> Vec<LanguageModelEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        vec![LanguageModelEvent::Finish {
            reason: FinishReason::Other,
            usage: Usage::default(),
        }]
    }
}

fn map_incomplete(reason: &str) -> FinishReason {
    match reason {
        "max_output_tokens" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::{
        CompletedResponse, ResponsesOutputItem, ResponsesUsage,
    };
    use agent_sdk_language_model::LanguageModelEvent as E;

    fn item(kind: &str, id: &str, call_id: Option<&str>, name: Option<&str>) -> ResponsesOutputItem {
        ResponsesOutputItem {
            kind: kind.into(),
            id: Some(id.into()),
            call_id: call_id.map(str::to_string),
            name: name.map(str::to_string),
            arguments: None,
        }
    }

    #[test]
    fn text_only_lifecycle_emits_textdelta_and_finish_stop() {
        let mut m = ResponsesEventMapper::new();
        m.map(ResponsesEvent::OutputItemAdded(OutputItemAdded {
            item: item("message", "msg_1", None, None),
        }));
        let out = m.map(ResponsesEvent::OutputTextDelta(OutputTextDelta {
            delta: "hi".into(),
        }));
        assert!(matches!(out[0], E::TextDelta { ref delta } if delta == "hi"));
        let out = m.map(ResponsesEvent::Completed(ResponseCompleted {
            response: CompletedResponse {
                usage: Some(ResponsesUsage {
                    input_tokens: 8,
                    output_tokens: 3,
                    input_tokens_details: None,
                }),
                incomplete_details: None,
                status: Some("completed".into()),
            },
        }));
        match &out[0] {
            E::Finish { reason, usage } => {
                assert_eq!(*reason, FinishReason::Stop);
                assert_eq!(usage.input_tokens, 8);
                assert_eq!(usage.output_tokens, 3);
            }
            o => panic!("expected Finish, got {o:?}"),
        }
    }

    #[test]
    fn function_call_lifecycle_emits_start_delta_end_and_tooluse() {
        let mut m = ResponsesEventMapper::new();
        let out = m.map(ResponsesEvent::OutputItemAdded(OutputItemAdded {
            item: item("function_call", "item_1", Some("call_1"), Some("bash")),
        }));
        assert!(matches!(out[0], E::ToolCallStart { ref id, ref name } if id == "call_1" && name == "bash"));

        let out = m.map(ResponsesEvent::FunctionCallArgsDelta(
            FunctionCallArgsDelta {
                item_id: "item_1".into(),
                delta: "{\"cmd\":\"ls\"}".into(),
            },
        ));
        assert!(matches!(out[0], E::ToolCallDelta { ref id, .. } if id == "call_1"));

        let out = m.map(ResponsesEvent::OutputItemDone(OutputItemDone {
            item: item("function_call", "item_1", Some("call_1"), Some("bash")),
        }));
        assert!(matches!(out[0], E::ToolCallEnd { ref id, .. } if id == "call_1"));

        let out = m.map(ResponsesEvent::Completed(ResponseCompleted {
            response: CompletedResponse {
                usage: None,
                incomplete_details: None,
                status: None,
            },
        }));
        assert!(matches!(out[0], E::Finish { reason: FinishReason::ToolUse, .. }));
    }

    #[test]
    fn flush_emits_finish_only_if_not_already_done() {
        let mut m = ResponsesEventMapper::new();
        let f1 = m.flush();
        assert_eq!(f1.len(), 1);
        let f2 = m.flush();
        assert!(f2.is_empty());
    }
}
