//! Permission policy: the gate consulted at `PreToolUse`.
//!
//! See [`PermissionPolicy`] for the trait, [`Decision`] for the verdict
//! variants, [`AskUserCallback`] / [`channel_callback`] for resolving the
//! `AskUser` verdict, and [`AllowAll`] / [`DenyAll`] / [`AllowList`] /
//! [`DenyList`] for built-in policies.

mod builtins;
mod decision;
mod policy;

pub use builtins::{AllowAll, AllowList, DenyAll, DenyList};
pub use decision::Decision;
pub use policy::{channel_callback, AskUserCallback, AskUserFuture, AskUserPrompt, PermissionPolicy};
