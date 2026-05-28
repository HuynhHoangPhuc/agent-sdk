//! Shared fixture provider used by the core integration tests.
//!
//! `FixtureModel` is a scripted [`LanguageModel`] that replays one
//! pre-recorded stream of [`LanguageModelEvent`]s per `stream()` call. The
//! scripts drive the agent loop deterministically without needing an HTTP
//! mock.

#![allow(dead_code)]

use std::pin::Pin;
use std::sync::Mutex;
use std::time::Duration;

use agent_sdk_language_model::{
    BoxedEventStream, FinishReason, LanguageModel, LanguageModelEvent, LanguageModelRequest,
    ModelError, Usage,
};
use async_trait::async_trait;
use futures_util::stream;
use tokio_util::sync::CancellationToken;

/// One scripted turn — the events the fixture should emit, optionally with a
/// per-event delay so tests can interleave cancellation.
pub struct ScriptedTurn {
    pub events: Vec<LanguageModelEvent>,
    pub delay_between_events: Duration,
}

impl ScriptedTurn {
    pub fn instant(events: Vec<LanguageModelEvent>) -> Self {
        Self {
            events,
            delay_between_events: Duration::ZERO,
        }
    }
}

/// Fixture provider — pops one [`ScriptedTurn`] per `stream()` call.
pub struct FixtureModel {
    turns: Mutex<Vec<ScriptedTurn>>,
}

impl FixtureModel {
    pub fn new(turns: Vec<ScriptedTurn>) -> Self {
        // Reverse so we can pop in order.
        let mut t = turns;
        t.reverse();
        Self {
            turns: Mutex::new(t),
        }
    }
}

#[async_trait]
impl LanguageModel for FixtureModel {
    fn provider_id(&self) -> &str {
        "fixture"
    }
    fn model_id(&self) -> &str {
        "fixture-1"
    }

    async fn stream(
        &self,
        _req: LanguageModelRequest,
        cancel: CancellationToken,
    ) -> Result<BoxedEventStream, ModelError> {
        let turn = self.turns.lock().unwrap().pop().unwrap_or_else(|| {
            ScriptedTurn::instant(vec![LanguageModelEvent::Finish {
                reason: FinishReason::Stop,
                usage: Usage::default(),
            }])
        });

        let delay = turn.delay_between_events;
        let events = turn.events;

        let s = stream::unfold(
            (events.into_iter(), delay, cancel),
            |(mut iter, delay, cancel)| async move {
                if cancel.is_cancelled() {
                    return None;
                }
                if !delay.is_zero() {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return None,
                        _ = tokio::time::sleep(delay) => {}
                    }
                }
                iter.next()
                    .map(|ev| (Ok::<_, ModelError>(ev), (iter, delay, cancel)))
            },
        );
        Ok(Pin::from(Box::new(s) as Box<_>))
    }
}

/// Convenience: build a simple "text-then-finish" turn.
pub fn text_turn(text: &str) -> ScriptedTurn {
    ScriptedTurn::instant(vec![
        LanguageModelEvent::TextDelta {
            delta: text.into(),
        },
        LanguageModelEvent::Finish {
            reason: FinishReason::Stop,
            usage: Usage::default(),
        },
    ])
}

/// Convenience: build a "tool-call then finish(tool_use)" turn for a single
/// tool. The tool-call args are emitted in two deltas to exercise the
/// accumulator.
pub fn tool_turn(id: &str, name: &str, args_json: serde_json::Value) -> ScriptedTurn {
    let args_str = args_json.to_string();
    let mid = args_str.len() / 2;
    let (left, right) = args_str.split_at(mid);
    ScriptedTurn::instant(vec![
        LanguageModelEvent::ToolCallStart {
            id: id.into(),
            name: name.into(),
        },
        LanguageModelEvent::ToolCallDelta {
            id: id.into(),
            arguments_delta: left.into(),
        },
        LanguageModelEvent::ToolCallDelta {
            id: id.into(),
            arguments_delta: right.into(),
        },
        LanguageModelEvent::ToolCallEnd {
            id: id.into(),
            arguments: None,
        },
        LanguageModelEvent::Finish {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
        },
    ])
}

/// Multi-tool turn (parallel ids).
pub fn multi_tool_turn(calls: Vec<(&str, &str, serde_json::Value)>) -> ScriptedTurn {
    let mut events = Vec::new();
    for (id, name, args) in &calls {
        events.push(LanguageModelEvent::ToolCallStart {
            id: (*id).into(),
            name: (*name).into(),
        });
        events.push(LanguageModelEvent::ToolCallEnd {
            id: (*id).into(),
            arguments: Some(args.clone()),
        });
    }
    events.push(LanguageModelEvent::Finish {
        reason: FinishReason::ToolUse,
        usage: Usage::default(),
    });
    ScriptedTurn::instant(events)
}
