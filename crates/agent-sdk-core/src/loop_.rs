//! The streaming agent loop.
//!
//! `run_loop` is spawned by [`Agent::run_stream`](crate::Agent::run_stream)
//! onto a tokio task. It drives the model, accumulates streaming events into
//! a complete assistant message, executes tools (parallel by default), and
//! pushes [`AgentEvent`]s onto an `mpsc` sender consumed by
//! [`AgentEventStream`](crate::AgentEventStream).
//!
//! Hook and permission call sites are stubbed for Phase 5 — the loop runs
//! every hook point as a no-op pass-through so wiring them later is purely
//! additive.

use std::collections::HashMap;
use std::sync::Arc;

use agent_sdk_language_model::{
    ContentBlock, FinishReason, LanguageModel, LanguageModelEvent, LanguageModelRequest, Message,
    ModelError, Role, ToolChoice, ToolSpec, Usage,
};
use futures_util::StreamExt;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::event::{add_usage, error_event};
use crate::stop::{ArcStopCondition, LoopState, StopCondition};
use crate::{AgentError, AgentEvent, Session, Tool, ToolResult};

/// Configuration for one run of the agent loop, owned by the spawned task.
pub(crate) struct LoopConfig {
    pub model: Arc<dyn LanguageModel>,
    pub tools: HashMap<String, Arc<dyn Tool>>,
    pub tool_specs: Vec<ToolSpec>,
    pub system: Option<String>,
    pub tool_choice: ToolChoice,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_turns: u32,
    pub parallel_tool_calls: bool,
    pub stop_when: ArcStopCondition,
    pub session: Session,
    pub cancel: CancellationToken,
}

/// Per-tool-use accumulator while a streamed turn is in flight.
#[derive(Default)]
struct ToolUseAccum {
    name: String,
    arg_buffer: String,
    final_args: Option<serde_json::Value>,
    /// Order in which the tool call first appeared this turn. Used to drive
    /// stable ordering of `ContentBlock::ToolUse` blocks.
    order: u32,
    /// `true` once the `ToolCallStart` event has been forwarded to the agent
    /// stream — used to suppress accidental duplicate emissions if a provider
    /// sends two starts for the same id.
    announced: bool,
}

/// Entry point used by the spawned task.
pub(crate) async fn run_loop(
    cfg: LoopConfig,
    tx: mpsc::Sender<AgentEvent>,
    finish: oneshot::Sender<Result<(), AgentError>>,
) {
    let final_result = match drive(cfg, &tx).await {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = tx.send(error_event(&err)).await;
            Err(err)
        }
    };
    let _ = finish.send(final_result);
}

async fn drive(mut cfg: LoopConfig, tx: &mpsc::Sender<AgentEvent>) -> Result<(), AgentError> {
    let mut total_usage = Usage::default();
    let mut turn: u32 = 0;

    loop {
        if cfg.cancel.is_cancelled() {
            return Err(AgentError::Cancelled);
        }

        // Hooks: UserPromptSubmit (first turn) / PreModel — wired in Phase 5.

        let mut req = LanguageModelRequest::default();
        req.messages = cfg.session.messages.clone();
        req.system = cfg.system.clone();
        req.tools = cfg.tool_specs.clone();
        req.tool_choice = cfg.tool_choice.clone();
        req.temperature = cfg.temperature;
        req.max_tokens = cfg.max_tokens;

        let turn_outcome = run_turn(&cfg, req, turn, tx).await?;
        // Hooks: PostModel — wired in Phase 5.

        let TurnOutcome {
            assistant_message,
            finish_reason,
            usage,
            tool_uses,
        } = turn_outcome;

        total_usage = add_usage(total_usage, usage);
        cfg.session.push(assistant_message);

        send_event(
            tx,
            AgentEvent::TurnEnd {
                turn,
                reason: finish_reason,
                usage,
            },
        )
        .await?;

        turn = turn.saturating_add(1);

        let tool_call_count = tool_uses.len() as u32;

        // Natural termination: no tool calls => provider considered the turn final.
        if tool_uses.is_empty() {
            send_event(
                tx,
                AgentEvent::Finish {
                    turns: turn,
                    reason: finish_reason,
                    usage: total_usage,
                },
            )
            .await?;
            return Ok(());
        }

        // Run tools and append their results to the session.
        let tool_results = execute_tools(&cfg, &tool_uses, turn - 1, tx).await?;
        cfg.session.push(Message::new(Role::User, tool_results));

        // Stop predicate (after a turn with tool calls).
        let state = LoopState {
            turns: turn,
            last_reason: finish_reason,
            last_tool_call_count: tool_call_count,
        };
        if cfg.stop_when.should_stop(&state) {
            send_event(
                tx,
                AgentEvent::Finish {
                    turns: turn,
                    reason: finish_reason,
                    usage: total_usage,
                },
            )
            .await?;
            return Ok(());
        }

        if turn >= cfg.max_turns {
            return Err(AgentError::MaxTurns(cfg.max_turns));
        }
    }
}

/// What one model turn produced after the provider stream is fully consumed.
struct TurnOutcome {
    assistant_message: Message,
    finish_reason: FinishReason,
    usage: Usage,
    tool_uses: Vec<FinalToolUse>,
}

/// A fully-accumulated tool call from one turn.
struct FinalToolUse {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

async fn run_turn(
    cfg: &LoopConfig,
    req: LanguageModelRequest,
    turn: u32,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<TurnOutcome, AgentError> {
    let mut stream = tokio::select! {
        biased;
        _ = cfg.cancel.cancelled() => return Err(AgentError::Cancelled),
        res = cfg.model.stream(req, cfg.cancel.clone()) => res?,
    };

    let mut text_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut reasoning_signature: Option<String> = None;
    let mut tool_calls: HashMap<String, ToolUseAccum> = HashMap::new();
    let mut next_order: u32 = 0;
    let mut finish_reason: Option<FinishReason> = None;
    let mut usage = Usage::default();

    loop {
        let event = tokio::select! {
            biased;
            _ = cfg.cancel.cancelled() => return Err(AgentError::Cancelled),
            next = stream.next() => match next {
                Some(ev) => ev,
                None => break,
            }
        };

        match event {
            Ok(LanguageModelEvent::TextDelta { delta }) => {
                text_buf.push_str(&delta);
                send_event(tx, AgentEvent::TextDelta { turn, delta }).await?;
            }
            Ok(LanguageModelEvent::ReasoningDelta { delta }) => {
                reasoning_buf.push_str(&delta);
                send_event(tx, AgentEvent::ReasoningDelta { turn, delta }).await?;
            }
            Ok(LanguageModelEvent::ReasoningSignature { signature }) => {
                reasoning_signature = Some(signature);
            }
            Ok(LanguageModelEvent::ToolCallStart { id, name }) => {
                let entry = tool_calls.entry(id.clone()).or_default();
                if !entry.announced {
                    entry.name = name.clone();
                    entry.order = next_order;
                    next_order = next_order.saturating_add(1);
                    entry.announced = true;
                    send_event(tx, AgentEvent::ToolCallStart { turn, id, name }).await?;
                }
            }
            Ok(LanguageModelEvent::ToolCallDelta {
                id,
                arguments_delta,
            }) => {
                let entry = tool_calls.entry(id).or_default();
                entry.arg_buffer.push_str(&arguments_delta);
            }
            Ok(LanguageModelEvent::ToolCallEnd { id, arguments }) => {
                let entry = tool_calls.entry(id.clone()).or_default();
                if let Some(value) = arguments {
                    entry.final_args = Some(value);
                }
            }
            Ok(LanguageModelEvent::Finish { reason, usage: u }) => {
                finish_reason = Some(reason);
                usage = u;
            }
            Ok(LanguageModelEvent::Error { message }) => {
                // Wrap once — Display would otherwise read "provider error: provider error: …".
                return Err(AgentError::Provider(ModelError::Other(message)));
            }
            // LanguageModelEvent is #[non_exhaustive]; ignore unknown variants
            // forward-compatibly.
            Ok(_) => {}
            Err(err) => return Err(AgentError::Provider(err)),
        }
    }

    let finish_reason = finish_reason.unwrap_or(FinishReason::Other);

    // Finalize tool-call args from buffers; emit ToolCallEnd for each.
    let mut tool_uses_ordered: Vec<(String, ToolUseAccum)> = tool_calls.into_iter().collect();
    tool_uses_ordered.sort_by_key(|(_, accum)| accum.order);

    let mut final_tool_uses: Vec<FinalToolUse> = Vec::with_capacity(tool_uses_ordered.len());

    for (id, accum) in tool_uses_ordered {
        let arguments = if let Some(v) = accum.final_args {
            v
        } else if accum.arg_buffer.trim().is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            match serde_json::from_str(&accum.arg_buffer) {
                Ok(v) => v,
                Err(e) => {
                    return Err(AgentError::InvalidToolArguments {
                        name: accum.name,
                        message: e.to_string(),
                    });
                }
            }
        };

        send_event(
            tx,
            AgentEvent::ToolCallEnd {
                turn,
                id: id.clone(),
                name: accum.name.clone(),
                arguments: arguments.clone(),
            },
        )
        .await?;

        final_tool_uses.push(FinalToolUse {
            id,
            name: accum.name,
            arguments,
        });
    }

    let mut content: Vec<ContentBlock> = Vec::new();
    if !reasoning_buf.is_empty() {
        content.push(ContentBlock::Reasoning {
            text: reasoning_buf,
            signature: reasoning_signature,
        });
    }
    if !text_buf.is_empty() {
        content.push(ContentBlock::text(text_buf));
    }
    for tu in &final_tool_uses {
        content.push(ContentBlock::ToolUse {
            id: tu.id.clone(),
            name: tu.name.clone(),
            input: tu.arguments.clone(),
        });
    }

    let assistant_message = Message::new(Role::Assistant, content);

    Ok(TurnOutcome {
        assistant_message,
        finish_reason,
        usage,
        tool_uses: final_tool_uses,
    })
}

async fn execute_tools(
    cfg: &LoopConfig,
    tool_uses: &[FinalToolUse],
    turn: u32,
    tx: &mpsc::Sender<AgentEvent>,
) -> Result<Vec<ContentBlock>, AgentError> {
    // Pre-flight: every requested tool must be registered. Surfacing this
    // before spawning avoids leaving sibling tool tasks running unaware that
    // the loop has already decided to abort.
    for tu in tool_uses {
        if !cfg.tools.contains_key(&tu.name) {
            return Err(AgentError::Tool {
                name: tu.name.clone(),
                message: "tool not registered with agent".into(),
            });
        }
    }

    let mut results: Vec<(usize, ContentBlock)> = Vec::with_capacity(tool_uses.len());

    if cfg.parallel_tool_calls && tool_uses.len() > 1 {
        let mut set: JoinSet<(usize, FinalToolUse, Result<ToolResult, AgentError>)> = JoinSet::new();
        for (idx, tu) in tool_uses.iter().enumerate() {
            // Pre-flight above guarantees presence.
            let tool = cfg.tools.get(&tu.name).expect("validated above").clone();
            let args = tu.arguments.clone();
            let cancel = cfg.cancel.clone();
            let owned = FinalToolUse {
                id: tu.id.clone(),
                name: tu.name.clone(),
                arguments: tu.arguments.clone(),
            };
            set.spawn(async move {
                let res = tool.execute(args, cancel).await;
                (idx, owned, res)
            });
        }

        loop {
            let next = tokio::select! {
                biased;
                _ = cfg.cancel.cancelled() => {
                    set.abort_all();
                    return Err(AgentError::Cancelled);
                }
                next = set.join_next() => next,
            };
            let Some(join_res) = next else { break };
            match join_res {
                Ok((idx, tu, Ok(result))) => {
                    if send_event(
                        tx,
                        AgentEvent::ToolResult {
                            turn,
                            id: tu.id.clone(),
                            name: tu.name.clone(),
                            content: result.content.clone(),
                            is_error: result.is_error,
                        },
                    )
                    .await
                    .is_err()
                    {
                        set.abort_all();
                        return Err(AgentError::Cancelled);
                    }
                    results.push((
                        idx,
                        ContentBlock::ToolResult {
                            tool_use_id: tu.id,
                            content: result.content,
                            is_error: if result.is_error { Some(true) } else { None },
                        },
                    ));
                }
                Ok((_, tu, Err(err))) => {
                    set.abort_all();
                    return Err(map_tool_err(tu.name, err));
                }
                Err(join_err) => {
                    set.abort_all();
                    if join_err.is_cancelled() {
                        return Err(AgentError::Cancelled);
                    }
                    return Err(AgentError::Other(format!("tool task panicked: {join_err}")));
                }
            }
        }

        results.sort_by_key(|(idx, _)| *idx);
    } else {
        for (idx, tu) in tool_uses.iter().enumerate() {
            let tool = cfg.tools.get(&tu.name).expect("validated above").clone();
            let result = tokio::select! {
                biased;
                _ = cfg.cancel.cancelled() => return Err(AgentError::Cancelled),
                res = tool.execute(tu.arguments.clone(), cfg.cancel.clone()) => res,
            };
            let result = match result {
                Ok(r) => r,
                Err(err) => return Err(map_tool_err(tu.name.clone(), err)),
            };
            send_event(
                tx,
                AgentEvent::ToolResult {
                    turn,
                    id: tu.id.clone(),
                    name: tu.name.clone(),
                    content: result.content.clone(),
                    is_error: result.is_error,
                },
            )
            .await?;
            results.push((
                idx,
                ContentBlock::ToolResult {
                    tool_use_id: tu.id.clone(),
                    content: result.content,
                    is_error: if result.is_error { Some(true) } else { None },
                },
            ));
        }
    }

    Ok(results.into_iter().map(|(_, c)| c).collect())
}

/// Send an event to the caller. If the receiver has been dropped (the caller
/// stopped consuming the [`AgentEventStream`]), treat it as an implicit
/// cancellation: the loop has no audience.
async fn send_event(
    tx: &mpsc::Sender<AgentEvent>,
    event: AgentEvent,
) -> Result<(), AgentError> {
    tx.send(event).await.map_err(|_| AgentError::Cancelled)
}

/// Normalize a tool-side error: if it's already a structured agent error,
/// keep it; otherwise wrap with the tool name for traceability.
fn map_tool_err(name: String, err: AgentError) -> AgentError {
    match err {
        AgentError::Tool { .. }
        | AgentError::Cancelled
        | AgentError::Permission { .. }
        | AgentError::HookHalt(_) => err,
        other => AgentError::Tool {
            name,
            message: other.to_string(),
        },
    }
}
