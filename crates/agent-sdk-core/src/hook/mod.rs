//! Hook system: lifecycle observers wired into the agent loop.
//!
//! See [`Hook`] for the trait, [`HookEvent`] / [`HookOutcome`] for the data
//! exchanged with the loop, and [`ExternalCommandHook`] for the built-in
//! subprocess hook.

mod ctx;
mod external_command;
mod trait_;

pub use ctx::{HookCtx, HookEvent, HookOutcome};
pub use external_command::{ExternalCommandHook, DEFAULT_TIMEOUT};
pub use trait_::Hook;
