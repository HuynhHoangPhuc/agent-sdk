//! Loop-termination predicates.
//!
//! A `StopCondition` is consulted at the end of every turn. It receives a
//! lightweight [`LoopState`] snapshot and returns `true` to stop the loop.
//! Predicates compose with [`StopCondition::or`] / [`StopCondition::and`].

use std::sync::Arc;

use agent_sdk_language_model::FinishReason;

/// Snapshot passed to a [`StopCondition`] at each turn boundary.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoopState {
    /// One-based count of turns completed so far.
    pub turns: u32,
    /// Provider's finish reason for the just-completed turn.
    pub last_reason: FinishReason,
    /// How many tool calls the last turn produced. Zero implies the model
    /// returned a plain text reply.
    pub last_tool_call_count: u32,
}

/// Composable loop-termination predicate. Implementations should be cheap;
/// they are called once per turn.
pub trait StopCondition: Send + Sync + 'static {
    /// Return `true` to stop the loop after the current turn.
    fn should_stop(&self, state: &LoopState) -> bool;
}

/// Type-erased [`StopCondition`] used inside the agent.
pub type ArcStopCondition = Arc<dyn StopCondition>;

/// Always-false predicate. The default — the loop runs until the provider
/// stops emitting tool calls or [`max_turns`](crate::AgentBuilder::max_turns)
/// is hit.
#[derive(Debug, Default)]
pub struct Never;

impl StopCondition for Never {
    fn should_stop(&self, _state: &LoopState) -> bool {
        false
    }
}

/// Stop once `state.turns >= n`.
#[derive(Debug)]
pub struct StopCountIs(pub u32);

impl StopCondition for StopCountIs {
    fn should_stop(&self, state: &LoopState) -> bool {
        state.turns >= self.0
    }
}

/// Stop the first turn the assistant returns no tool calls.
#[derive(Debug, Default)]
pub struct NoToolCalls;

impl StopCondition for NoToolCalls {
    fn should_stop(&self, state: &LoopState) -> bool {
        state.last_tool_call_count == 0
    }
}

/// Convenience: stop once `state.turns >= n`.
pub fn stop_count_is(n: u32) -> Arc<StopCountIs> {
    Arc::new(StopCountIs(n))
}

/// Convenience: stop the first turn with no tool calls.
pub fn no_tool_calls() -> Arc<NoToolCalls> {
    Arc::new(NoToolCalls)
}

/// Disjunction of two predicates.
pub struct Or<A: StopCondition, B: StopCondition>(pub A, pub B);

impl<A: StopCondition, B: StopCondition> StopCondition for Or<A, B> {
    fn should_stop(&self, state: &LoopState) -> bool {
        self.0.should_stop(state) || self.1.should_stop(state)
    }
}

/// Conjunction of two predicates.
pub struct And<A: StopCondition, B: StopCondition>(pub A, pub B);

impl<A: StopCondition, B: StopCondition> StopCondition for And<A, B> {
    fn should_stop(&self, state: &LoopState) -> bool {
        self.0.should_stop(state) && self.1.should_stop(state)
    }
}

/// Extension methods for composing predicates with `.or(..)` / `.and(..)`.
pub trait StopConditionExt: StopCondition + Sized {
    /// Compose with `other` via logical OR.
    fn or<B: StopCondition>(self, other: B) -> Or<Self, B> {
        Or(self, other)
    }
    /// Compose with `other` via logical AND.
    fn and<B: StopCondition>(self, other: B) -> And<Self, B> {
        And(self, other)
    }
}

impl<T: StopCondition + Sized> StopConditionExt for T {}

/// Blanket-impl: an `Arc<T: StopCondition>` is itself a [`StopCondition`].
impl<T: StopCondition + ?Sized> StopCondition for Arc<T> {
    fn should_stop(&self, state: &LoopState) -> bool {
        (**self).should_stop(state)
    }
}
