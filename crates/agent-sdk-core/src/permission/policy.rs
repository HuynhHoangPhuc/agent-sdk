//! [`PermissionPolicy`] trait + [`AskUserCallback`] resolution machinery.
//!
//! The trait is intentionally **object-safe** (no generics, owned args at the
//! boundary) so the v0.2 C-ABI layer can map it onto a `dyn` handle without
//! re-shaping the API.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use super::decision::Decision;

/// Permission gate consulted at `PreToolUse`.
///
/// One trait, one method. The policy returns a [`Decision`]; the loop
/// translates `AskUser` into a callback invocation if a resolver is wired up
/// (otherwise it degrades to `Deny`).
#[async_trait]
pub trait PermissionPolicy: Send + Sync {
    /// Decide whether a tool may run with the given arguments.
    async fn check(&self, tool: &str, arguments: &Value) -> Decision;
}

/// Boxed future returned by an [`AskUserCallback`].
pub type AskUserFuture = Pin<Box<dyn Future<Output = Decision> + Send + 'static>>;

/// Resolver invoked by the loop when a policy returns
/// [`Decision::AskUser`](super::Decision::AskUser).
///
/// Shape chosen to map cleanly onto a C-ABI `extern "C" fn(*mut c_void, *const c_char, ...)`
/// fn-pointer + opaque `user_data`. Rust callers can use the
/// [`channel_callback`] adapter below for the more idiomatic
/// "receive on a channel, reply with a oneshot" pattern.
pub type AskUserCallback =
    Arc<dyn Fn(String) -> AskUserFuture + Send + Sync + 'static>;

/// One pending question delivered to an [`AskUserChannel`] receiver.
pub struct AskUserPrompt {
    /// Prompt to show the user.
    pub prompt: String,
    /// Send the user's decision back to the loop. Dropping without sending is
    /// treated as `Deny("ask-user channel dropped")` by the loop.
    pub reply: oneshot::Sender<Decision>,
}

/// Build an [`AskUserCallback`] backed by a tokio mpsc channel.
///
/// Returns the callback (hand to `.permission_ask(..)`) and a receiver the
/// harness drains. Each prompt arrives with an attached `oneshot::Sender`; the
/// harness replies with a [`Decision`].
pub fn channel_callback() -> (AskUserCallback, mpsc::UnboundedReceiver<AskUserPrompt>) {
    let (tx, rx) = mpsc::unbounded_channel::<AskUserPrompt>();
    let cb: AskUserCallback = Arc::new(move |prompt: String| {
        let tx = tx.clone();
        Box::pin(async move {
            let (reply_tx, reply_rx) = oneshot::channel();
            if tx
                .send(AskUserPrompt {
                    prompt,
                    reply: reply_tx,
                })
                .is_err()
            {
                return Decision::Deny("ask-user channel closed".into());
            }
            match reply_rx.await {
                Ok(decision) => decision,
                Err(_) => Decision::Deny("ask-user channel dropped".into()),
            }
        }) as AskUserFuture
    });
    (cb, rx)
}
