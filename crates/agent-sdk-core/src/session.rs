//! Conversation session state — in-memory and the trait for pluggable stores.
//!
//! v0.1 ships only the in-memory [`Session`]. The [`SessionStore`] trait is
//! defined now so the v0.2 SQLite-backed implementation can drop in without
//! API change; for v0.1 callers manually shuttle [`Session`]s around.

use std::collections::BTreeMap;

use agent_sdk_language_model::Message;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::AgentError;

/// In-memory conversation state. `serde`-round-trippable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Session {
    /// Opaque session identifier. Callers choose the format (uuid, slug, ...).
    pub id: String,
    /// Ordered conversation history.
    pub messages: Vec<Message>,
    /// Free-form metadata for callers and hooks (e.g., user id, parent run).
    /// Sorted map to keep the serialized form stable.
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl Session {
    /// Create an empty session with the given id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            messages: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Append a message.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }
}

/// Pluggable session persistence. No implementation ships with v0.1; the
/// trait is locked so v0.2 can add SQLite/sled/sqlx variants without breaking
/// the public surface.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Load a session by id. Returns `Ok(None)` if not found.
    async fn load(&self, id: &str) -> Result<Option<Session>, AgentError>;

    /// Persist (insert or replace) a session.
    async fn save(&self, session: &Session) -> Result<(), AgentError>;

    /// Delete a session by id. A missing id is not an error.
    async fn delete(&self, id: &str) -> Result<(), AgentError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_language_model::Message;

    #[test]
    fn session_round_trips_through_serde() {
        let mut s = Session::new("sess-1");
        s.push(Message::user("hello"));
        s.push(Message::assistant("hi"));
        s.metadata
            .insert("user".into(), serde_json::json!("alice"));
        let json = serde_json::to_string(&s).expect("serialize");
        let back: Session = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, s);
    }
}
