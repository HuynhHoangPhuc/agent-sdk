//! The [`Agent`] handle and its `run` / `run_stream` entry points.

use std::collections::HashMap;
use std::sync::Arc;

use agent_sdk_language_model::{
    ContentBlock, FinishReason, LanguageModel, Message, ToolChoice, ToolSpec, Usage,
};
use futures_util::StreamExt;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::hook::Hook;
use crate::loop_::{run_loop, LoopConfig};
use crate::permission::{AskUserCallback, PermissionPolicy};
use crate::stop::ArcStopCondition;
use crate::{
    AgentBuilder, AgentError, AgentEvent, AgentEventStream, RunOutput, Session, Tool,
};

/// Runtime handle returned by [`AgentBuilder::build`].
///
/// `Agent` is cheap to clone (it's effectively `Arc`s under the hood through
/// the trait objects). Cloning lets the same configuration be reused for
/// multiple runs.
pub struct Agent {
    pub(crate) model: Arc<dyn LanguageModel>,
    pub(crate) tools: HashMap<String, Arc<dyn Tool>>,
    pub(crate) tool_specs: Vec<ToolSpec>,
    pub(crate) system: Option<String>,
    pub(crate) tool_choice: ToolChoice,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) max_turns: u32,
    pub(crate) parallel_tool_calls: bool,
    pub(crate) stop_when: ArcStopCondition,
    pub(crate) default_session: Option<Session>,
    pub(crate) hooks: Vec<Arc<dyn Hook>>,
    pub(crate) permission: Arc<dyn PermissionPolicy>,
    pub(crate) ask_user: Option<AskUserCallback>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("provider_id", &self.model.provider_id())
            .field("model_id", &self.model.model_id())
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("max_turns", &self.max_turns)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Start a new [`AgentBuilder`].
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Run the agent loop to completion and return aggregated output.
    ///
    /// Convenience over [`Self::run_stream`]: spins the stream, accumulates
    /// every event into [`RunOutput`], and surfaces the first error (if any).
    pub async fn run(&self, input: impl Into<String>) -> Result<RunOutput, AgentError> {
        let cancel = CancellationToken::new();
        self.run_with_cancel(input, cancel).await
    }

    /// Like [`Self::run`] but with an externally-controlled cancel token.
    pub async fn run_with_cancel(
        &self,
        input: impl Into<String>,
        cancel: CancellationToken,
    ) -> Result<RunOutput, AgentError> {
        let mut stream = self.run_stream_with_cancel(input, cancel);
        let mut events = Vec::new();
        let mut text = String::new();
        let mut turns: u32 = 0;
        let mut reason = FinishReason::Other;
        let mut usage = Usage::default();

        while let Some(event) = stream.next().await {
            match &event {
                AgentEvent::TextDelta { delta, .. } => text.push_str(delta),
                AgentEvent::TurnEnd { .. } => {
                    // Usage is aggregated below via the Finish event; per-turn
                    // accumulation is exposed through events only.
                }
                AgentEvent::Finish {
                    turns: t,
                    reason: r,
                    usage: u,
                } => {
                    turns = *t;
                    reason = *r;
                    usage = *u;
                }
                _ => {}
            }
            events.push(event);
        }

        stream.final_result().await?;

        Ok(RunOutput {
            events,
            text,
            turns,
            reason,
            usage,
        })
    }

    /// Start a streaming run. The returned [`AgentEventStream`] yields events
    /// in real time and closes when the loop finishes (or errors).
    pub fn run_stream(&self, input: impl Into<String>) -> AgentEventStream {
        self.run_stream_with_cancel(input, CancellationToken::new())
    }

    /// Like [`Self::run_stream`] with an externally-controlled cancel token.
    pub fn run_stream_with_cancel(
        &self,
        input: impl Into<String>,
        cancel: CancellationToken,
    ) -> AgentEventStream {
        let mut session = self.default_session.clone().unwrap_or_default();
        let input = input.into();
        if !input.is_empty() {
            session.push(Message::new(
                agent_sdk_language_model::Role::User,
                vec![ContentBlock::text(input.clone())],
            ));
        }

        let (tx, rx) = mpsc::channel::<AgentEvent>(64);
        let (finish_tx, finish_rx) = oneshot::channel::<Result<(), AgentError>>();

        let cfg = LoopConfig {
            model: self.model.clone(),
            tools: self.tools.clone(),
            tool_specs: self.tool_specs.clone(),
            system: self.system.clone(),
            tool_choice: self.tool_choice.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
            parallel_tool_calls: self.parallel_tool_calls,
            stop_when: self.stop_when.clone(),
            session,
            cancel,
            hooks: self.hooks.clone(),
            permission: self.permission.clone(),
            ask_user: self.ask_user.clone(),
            initial_input: input,
        };

        tokio::spawn(async move {
            run_loop(cfg, tx, finish_tx).await;
        });

        AgentEventStream::new(rx, finish_rx)
    }
}
