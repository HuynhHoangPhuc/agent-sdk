//! The [`Hook`] trait.

use async_trait::async_trait;

use super::ctx::{HookCtx, HookOutcome};

/// Lifecycle observer wired into [`Agent`](crate::Agent).
///
/// Object-safe: `&self` receivers, no generics, owned argument. Implementations
/// must be `Send + Sync` so they can be stored as `Arc<dyn Hook>` and called
/// from the loop task.
///
/// One hook handles every [`HookEvent`](super::HookEvent) variant; dispatch
/// inside the impl. Most hooks pattern-match `ctx.event` and return
/// [`HookOutcome::Continue`] for events they do not care about.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Called by the loop at each [`HookEvent`](super::HookEvent) transition.
    async fn on_event(&self, ctx: HookCtx) -> HookOutcome;
}
