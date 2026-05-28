//! The streaming agent loop.
//!
//! `run_loop` is spawned by [`Agent::run_stream`](crate::Agent::run_stream)
//! onto a tokio task. It drives the model, accumulates streaming events into
//! a complete assistant message, executes tools (parallel by default), and
//! pushes [`AgentEvent`]s onto an `mpsc` sender consumed by
//! [`AgentEventStream`](crate::AgentEventStream).
//!
//! Hooks fire at every lifecycle transition (`PreModel`, `PostModel`,
//! `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `OnError`, `OnFinish`),
//! and the configured [`PermissionPolicy`] is consulted just before each tool
//! executes.

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
use crate::hook::{Hook, HookCtx, HookEvent, HookOutcome};
use crate::permission::{AskUserCallback, Decision, PermissionPolicy};
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
    pub hooks: Vec<Arc<dyn Hook>>,
    pub permission: Arc<dyn PermissionPolicy>,
    pub ask_user: Option<AskUserCallback>,
    /// Original user input — passed verbatim to the `UserPromptSubmit` hook
    /// before the first model turn.
    pub initial_input: String,
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
    // Snapshot what `drive` needs to fire OnError before the cfg moves in.
    let hooks_for_err = cfg.hooks.clone();
    let provider_id = cfg.model.provider_id().to_string();
    let model_id = cfg.model.model_id().to_string();
    let session_id = if cfg.session.id.is_empty() {
        None
    } else {
        Some(cfg.session.id.clone())
    };

    let final_result = match drive(cfg, &tx).await {
        Ok(()) => Ok(()),
        Err(err) => {
            // OnError: notify every hook. Observers only — they cannot
            // suppress the underlying error.
            for hook in &hooks_for_err {
                let _ = hook
                    .on_event(HookCtx {
                        turn: 0,
                        provider_id: provider_id.clone(),
                        model_id: model_id.clone(),
                        session_id: session_id.clone(),
                        event: HookEvent::OnError {
                            message: err.to_string(),
                        },
                    })
                    .await;
            }
            let _ = tx.send(error_event(&err)).await;
            Err(err)
        }
    };
    let _ = finish.send(final_result);
}

/// Result of pushing one [`HookEvent`] through the hook chain.
enum HookFlow {
    /// Pass through. `Some(req)` carries the (possibly modified) request for
    /// `PreModel`; every other event uses `None`.
    Continue(Option<LanguageModelRequest>),
    /// A hook denied — surfaced to the model as a tool-result error at
    /// `PreToolUse`; treated as `HookHalt` elsewhere.
    Deny(String),
    /// A hook explicitly halted the loop.
    Halt(String),
}

/// Dispatch one event through every registered hook in order. Stops at the
/// first hook that returns `Deny`/`Halt`. `ModifyRequest` is honoured for
/// `PreModel` only; for every other event it is treated as `Continue`.
async fn dispatch_hooks(cfg: &LoopConfig, turn: u32, event: HookEvent) -> HookFlow {
    let session_id = if cfg.session.id.is_empty() {
        None
    } else {
        Some(cfg.session.id.clone())
    };
    let is_pre_model = matches!(event, HookEvent::PreModel { .. });
    // Seed the request carrier from the event itself so we always have a
    // valid request to return for PreModel even when no hook modifies it.
    let mut carried_request: Option<LanguageModelRequest> = match &event {
        HookEvent::PreModel { request } => Some(request.clone()),
        _ => None,
    };

    let mut event = event;
    for hook in &cfg.hooks {
        let ctx = HookCtx {
            turn,
            provider_id: cfg.model.provider_id().to_string(),
            model_id: cfg.model.model_id().to_string(),
            session_id: session_id.clone(),
            event: event.clone(),
        };
        match hook.on_event(ctx).await {
            HookOutcome::Continue => {}
            HookOutcome::ModifyRequest(req) => {
                if is_pre_model {
                    carried_request = Some(*req.clone());
                    // Update the event so subsequent hooks see the new request.
                    event = HookEvent::PreModel { request: *req };
                }
            }
            HookOutcome::Deny(reason) => return HookFlow::Deny(reason),
            HookOutcome::Halt(reason) => return HookFlow::Halt(reason),
        }
    }
    HookFlow::Continue(carried_request)
}

async fn drive(mut cfg: LoopConfig, tx: &mpsc::Sender<AgentEvent>) -> Result<(), AgentError> {
    let mut total_usage = Usage::default();
    let mut turn: u32 = 0;

    // UserPromptSubmit fires once at the top of the run, before the first
    // model call. Skip if there is no fresh input (e.g. seeded session).
    if !cfg.initial_input.is_empty() {
        let outcome = dispatch_hooks(
            &cfg,
            turn,
            HookEvent::UserPromptSubmit {
                input: cfg.initial_input.clone(),
            },
        )
        .await;
        match outcome {
            HookFlow::Continue(_) => {}
            HookFlow::Halt(reason) => return Err(AgentError::HookHalt(reason)),
            HookFlow::Deny(reason) => return Err(AgentError::HookHalt(reason)),
        }
    }

    loop {
        if cfg.cancel.is_cancelled() {
            return Err(AgentError::Cancelled);
        }

        let mut req = LanguageModelRequest::default();
        req.messages = cfg.session.messages.clone();
        req.system = cfg.system.clone();
        req.tools = cfg.tool_specs.clone();
        req.tool_choice = cfg.tool_choice.clone();
        req.temperature = cfg.temperature;
        req.max_tokens = cfg.max_tokens;

        // PreModel: hooks may rewrite the request before it goes upstream.
        req = match dispatch_hooks(&cfg, turn, HookEvent::PreModel { request: req }).await {
            HookFlow::Continue(Some(replaced)) => replaced,
            HookFlow::Continue(None) => unreachable!("PreModel must surface the request"),
            HookFlow::Halt(reason) => return Err(AgentError::HookHalt(reason)),
            HookFlow::Deny(reason) => return Err(AgentError::HookHalt(reason)),
        };

        let turn_outcome = run_turn(&cfg, req, turn, tx).await?;

        let TurnOutcome {
            assistant_message,
            finish_reason,
            usage,
            tool_uses,
        } = turn_outcome;

        // PostModel: notify hooks of the assistant message + usage.
        match dispatch_hooks(
            &cfg,
            turn,
            HookEvent::PostModel {
                message: assistant_message.clone(),
                usage,
            },
        )
        .await
        {
            HookFlow::Continue(_) => {}
            HookFlow::Halt(reason) | HookFlow::Deny(reason) => {
                return Err(AgentError::HookHalt(reason));
            }
        }

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
            // OnFinish: best-effort notify; ignore Deny/Halt — the loop is
            // already finished, hooks here are observers.
            let _ = dispatch_hooks(
                &cfg,
                turn,
                HookEvent::OnFinish {
                    turns: turn,
                    reason: finish_reason,
                    usage: total_usage,
                },
            )
            .await;
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
            let _ = dispatch_hooks(
                &cfg,
                turn,
                HookEvent::OnFinish {
                    turns: turn,
                    reason: finish_reason,
                    usage: total_usage,
                },
            )
            .await;
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
#[derive(Clone)]
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

    // Phase 1 (sequential): for each tool call, fire PreToolUse + permission.
    // Calls that survive both gates queue for execution; calls that get
    // denied/asked-and-denied resolve into a synthetic error ToolResult.
    let mut gated: Vec<GatedToolUse> = Vec::with_capacity(tool_uses.len());
    for tu in tool_uses {
        match gate_tool_call(cfg, turn, tu).await? {
            Some(result) => gated.push(GatedToolUse::PreResolved {
                tu: tu.clone(),
                result,
            }),
            None => gated.push(GatedToolUse::Ready(tu.clone())),
        }
    }

    // Phase 2 (parallel or sequential): execute the surviving calls.
    let mut indexed: Vec<(usize, FinalToolUse, ToolResult)> = Vec::with_capacity(gated.len());

    // First, harvest pre-resolved (denied) calls — no execution needed.
    let mut to_execute: Vec<(usize, FinalToolUse)> = Vec::new();
    for (idx, item) in gated.into_iter().enumerate() {
        match item {
            GatedToolUse::PreResolved { tu, result } => indexed.push((idx, tu, result)),
            GatedToolUse::Ready(tu) => to_execute.push((idx, tu)),
        }
    }

    if cfg.parallel_tool_calls && to_execute.len() > 1 {
        let mut set: JoinSet<(usize, FinalToolUse, Result<ToolResult, AgentError>)> =
            JoinSet::new();
        for (idx, tu) in to_execute {
            let tool = cfg.tools.get(&tu.name).expect("validated above").clone();
            let args = tu.arguments.clone();
            let cancel = cfg.cancel.clone();
            set.spawn(async move {
                let res = tool.execute(args, cancel).await;
                (idx, tu, res)
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
                Ok((idx, tu, Ok(result))) => indexed.push((idx, tu, result)),
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
    } else {
        for (idx, tu) in to_execute {
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
            indexed.push((idx, tu, result));
        }
    }

    // Phase 3: emit ToolResult events + run PostToolUse hooks in original
    // call order so observers see a stable sequence.
    indexed.sort_by_key(|(idx, _, _)| *idx);

    let mut results: Vec<(usize, ContentBlock)> = Vec::with_capacity(indexed.len());
    for (idx, tu, result) in indexed {
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

        // PostToolUse: observers only. Hook Halt still ends the loop.
        match dispatch_hooks(
            cfg,
            turn,
            HookEvent::PostToolUse {
                tool_use_id: tu.id.clone(),
                name: tu.name.clone(),
                result: result.clone(),
            },
        )
        .await
        {
            HookFlow::Continue(_) | HookFlow::Deny(_) => {}
            HookFlow::Halt(reason) => return Err(AgentError::HookHalt(reason)),
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

    Ok(results.into_iter().map(|(_, c)| c).collect())
}

/// Phase-2 outcome for a tool call after passing through the PreToolUse +
/// permission gates.
enum GatedToolUse {
    /// Hook or policy denied — a synthetic error result is already pinned.
    PreResolved { tu: FinalToolUse, result: ToolResult },
    /// Cleared to execute.
    Ready(FinalToolUse),
}

/// Run PreToolUse hooks + permission policy for a single call.
///
/// Returns:
/// - `Ok(Some(result))` — call was denied, synthetic error to send to model.
/// - `Ok(None)` — call passes, execute the tool.
/// - `Err(_)` — halt the loop (HookHalt, Cancelled).
async fn gate_tool_call(
    cfg: &LoopConfig,
    turn: u32,
    tu: &FinalToolUse,
) -> Result<Option<ToolResult>, AgentError> {
    // PreToolUse hooks first.
    match dispatch_hooks(
        cfg,
        turn,
        HookEvent::PreToolUse {
            tool_use_id: tu.id.clone(),
            name: tu.name.clone(),
            arguments: tu.arguments.clone(),
        },
    )
    .await
    {
        HookFlow::Continue(_) => {}
        HookFlow::Deny(reason) => {
            return Ok(Some(ToolResult::error(format!(
                "hook denied tool '{}': {reason}",
                tu.name
            ))));
        }
        HookFlow::Halt(reason) => return Err(AgentError::HookHalt(reason)),
    }

    // Permission policy.
    let decision = cfg.permission.check(&tu.name, &tu.arguments).await;
    let decision = match decision {
        Decision::AskUser { prompt } => match &cfg.ask_user {
            Some(cb) => cb(prompt).await,
            None => Decision::Deny(format!(
                "permission policy asked the user for '{}' but no AskUser callback is configured",
                tu.name
            )),
        },
        other => other,
    };

    match decision {
        Decision::Allow => Ok(None),
        Decision::Deny(reason) => Ok(Some(ToolResult::error(format!(
            "permission denied for '{}': {reason}",
            tu.name
        )))),
        // AskUser already resolved above; treat any residual as deny.
        Decision::AskUser { prompt } => Ok(Some(ToolResult::error(format!(
            "permission policy left AskUser unresolved for '{}': {prompt}",
            tu.name
        )))),
    }
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
