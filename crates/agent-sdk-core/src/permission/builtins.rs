//! Built-in [`PermissionPolicy`] implementations.

use std::collections::HashSet;

use async_trait::async_trait;
use serde_json::Value;

use super::decision::Decision;
use super::policy::PermissionPolicy;

/// Allow every tool call. Default when no policy is wired up.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllowAll;

#[async_trait]
impl PermissionPolicy for AllowAll {
    async fn check(&self, _tool: &str, _arguments: &Value) -> Decision {
        Decision::Allow
    }
}

/// Deny every tool call. Useful for harnesses that gate everything behind a
/// human-in-the-loop.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAll;

#[async_trait]
impl PermissionPolicy for DenyAll {
    async fn check(&self, _tool: &str, _arguments: &Value) -> Decision {
        Decision::Deny("DenyAll policy".into())
    }
}

/// Allow tools whose names appear in the list. Unknown tools route to
/// [`Decision::AskUser`].
#[derive(Debug, Clone)]
pub struct AllowList {
    allowed: HashSet<String>,
}

impl AllowList {
    /// Build from any iterable of tool-name strings.
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed: names.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl PermissionPolicy for AllowList {
    async fn check(&self, tool: &str, _arguments: &Value) -> Decision {
        if self.allowed.contains(tool) {
            Decision::Allow
        } else {
            Decision::AskUser {
                prompt: format!("Allow tool '{tool}'?"),
            }
        }
    }
}

/// Deny tools whose names appear in the list. Everything else is allowed
/// outright (no ask-user).
#[derive(Debug, Clone)]
pub struct DenyList {
    denied: HashSet<String>,
}

impl DenyList {
    /// Build from any iterable of tool-name strings.
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            denied: names.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl PermissionPolicy for DenyList {
    async fn check(&self, tool: &str, _arguments: &Value) -> Decision {
        if self.denied.contains(tool) {
            Decision::Deny(format!("tool '{tool}' is in deny list"))
        } else {
            Decision::Allow
        }
    }
}
