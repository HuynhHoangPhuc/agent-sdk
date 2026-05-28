//! [`AgentBuilder`] — the only constructor for [`Agent`](crate::Agent).

use std::collections::HashMap;
use std::sync::Arc;

use agent_sdk_language_model::{LanguageModel, ToolChoice, ToolSpec};

use crate::hook::Hook;
use crate::permission::{AllowAll, AskUserCallback, PermissionPolicy};
use crate::stop::{ArcStopCondition, Never, StopCondition};
use crate::{Agent, AgentError, Session, Tool};

/// Builder for [`Agent`]. Built via `Agent::builder()` / `AgentBuilder::new`.
///
/// All setters take owned values and return `self` so callers can chain. The
/// builder validates at [`build`](Self::build) — calling `build` without a
/// model returns [`AgentError::InvalidConfig`].
#[must_use = "AgentBuilder does nothing until .build() is called"]
pub struct AgentBuilder {
    model: Option<Arc<dyn LanguageModel>>,
    tools: Vec<Arc<dyn Tool>>,
    system: Option<String>,
    tool_choice: ToolChoice,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    max_turns: u32,
    parallel_tool_calls: bool,
    stop_when: ArcStopCondition,
    session: Option<Session>,
    hooks: Vec<Arc<dyn Hook>>,
    permission: Option<Arc<dyn PermissionPolicy>>,
    ask_user: Option<AskUserCallback>,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBuilder {
    /// Start a fresh builder. Defaults: `max_turns=16`, parallel tool calls
    /// on, [`Never`] stop predicate, [`ToolChoice::Auto`].
    pub fn new() -> Self {
        Self {
            model: None,
            tools: Vec::new(),
            system: None,
            tool_choice: ToolChoice::Auto,
            temperature: None,
            max_tokens: None,
            max_turns: 16,
            parallel_tool_calls: true,
            stop_when: Arc::new(Never),
            session: None,
            hooks: Vec::new(),
            permission: None,
            ask_user: None,
        }
    }

    /// Set the language model. Required.
    pub fn model<M: LanguageModel + 'static>(mut self, model: M) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Set the language model via an already-shared `Arc`. Useful when the
    /// same provider is shared across sub-agents.
    pub fn model_arc(mut self, model: Arc<dyn LanguageModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Register a single tool. Call repeatedly to add more.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Register a pre-shared tool.
    pub fn tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Replace the entire tools list.
    pub fn tools<I>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Arc<dyn Tool>>,
    {
        self.tools = tools.into_iter().collect();
        self
    }

    /// Override the model's tool-selection strategy.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Sampling temperature. Provider-clamped.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Hard cap on output tokens per turn.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Hard cap on the number of model turns in one run. Hitting this is an
    /// [`AgentError::MaxTurns`].
    pub fn max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Toggle parallel tool execution. `true` (the default) spawns each tool
    /// call on a tokio task and joins them; `false` runs them sequentially.
    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = parallel;
        self
    }

    /// Set a [`StopCondition`] consulted at the end of every turn.
    pub fn stop_when<S: StopCondition>(mut self, stop: S) -> Self {
        self.stop_when = Arc::new(stop);
        self
    }

    /// Provide an initial [`Session`] to seed the conversation. If omitted, a
    /// fresh empty session is created on `run`/`run_stream`.
    pub fn session(mut self, session: Session) -> Self {
        self.session = Some(session);
        self
    }

    /// Append a [`Hook`] to the lifecycle pipeline. Hooks fire in registration
    /// order at every event; the first hook to return `Deny`/`Halt` short-
    /// circuits the rest at that event.
    pub fn hook<H: Hook + 'static>(mut self, hook: H) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// Append a pre-shared hook.
    pub fn hook_arc(mut self, hook: Arc<dyn Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Replace the entire hooks list.
    pub fn hooks<I>(mut self, hooks: I) -> Self
    where
        I: IntoIterator<Item = Arc<dyn Hook>>,
    {
        self.hooks = hooks.into_iter().collect();
        self
    }

    /// Install a [`PermissionPolicy`] consulted at `PreToolUse`. If omitted,
    /// the loop uses [`AllowAll`].
    pub fn permission<P: PermissionPolicy + 'static>(mut self, policy: P) -> Self {
        self.permission = Some(Arc::new(policy));
        self
    }

    /// Install a pre-shared permission policy.
    pub fn permission_arc(mut self, policy: Arc<dyn PermissionPolicy>) -> Self {
        self.permission = Some(policy);
        self
    }

    /// Install the resolver invoked when the permission policy returns
    /// [`Decision::AskUser`](crate::Decision::AskUser). If omitted, `AskUser`
    /// degrades to `Deny`.
    pub fn permission_ask(mut self, cb: AskUserCallback) -> Self {
        self.ask_user = Some(cb);
        self
    }

    /// Validate and produce an [`Agent`].
    pub fn build(self) -> Result<Agent, AgentError> {
        let Some(model) = self.model else {
            return Err(AgentError::InvalidConfig(
                "Agent requires a model (call .model(..))".into(),
            ));
        };
        if self.max_turns == 0 {
            return Err(AgentError::InvalidConfig(
                "max_turns must be >= 1".into(),
            ));
        }

        let mut tools_map: HashMap<String, Arc<dyn Tool>> = HashMap::with_capacity(self.tools.len());
        let mut tool_specs: Vec<ToolSpec> = Vec::with_capacity(self.tools.len());
        for tool in self.tools {
            let name = tool.name().to_string();
            if tools_map.contains_key(&name) {
                return Err(AgentError::InvalidConfig(format!(
                    "duplicate tool name registered: {name}"
                )));
            }
            tool_specs.push(ToolSpec::new(
                name.clone(),
                tool.description().to_string(),
                tool.input_schema(),
            ));
            tools_map.insert(name, tool);
        }

        let permission: Arc<dyn PermissionPolicy> =
            self.permission.unwrap_or_else(|| Arc::new(AllowAll));

        Ok(Agent {
            model,
            tools: tools_map,
            tool_specs,
            system: self.system,
            tool_choice: self.tool_choice,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
            parallel_tool_calls: self.parallel_tool_calls,
            stop_when: self.stop_when,
            default_session: self.session,
            hooks: self.hooks,
            permission,
            ask_user: self.ask_user,
        })
    }
}
