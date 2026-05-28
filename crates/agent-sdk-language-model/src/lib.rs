//! # `agent-sdk-language-model`
//!
//! Provider-agnostic specification crate for the `agent-sdk` ecosystem. Defines
//! the [`LanguageModel`] trait plus the request/event types every provider
//! speaks. **Zero HTTP, zero runtime, zero `agent-sdk-core` dependency** — the
//! only thing a community provider author depends on.
//!
//! ## FFI-safe discipline
//!
//! All public types are deliberately constrained so the v0.2 C-ABI layer can
//! wrap them without breaking changes:
//!
//! - No GATs, no HRTB, no lifetimes in public signatures.
//! - Enums are `#[non_exhaustive]` for additive evolution.
//! - The streaming method returns a boxed stream with `'static` lifetime.
//!
//! ## Minimal provider
//!
//! ```ignore
//! use agent_sdk_language_model::{
//!     BoxedEventStream, LanguageModel, LanguageModelEvent, LanguageModelRequest,
//!     ModelError, FinishReason, Usage,
//! };
//! use async_trait::async_trait;
//! use futures_core::Stream;
//! use std::pin::Pin;
//! use tokio_util::sync::CancellationToken;
//!
//! pub struct EchoModel;
//!
//! #[async_trait]
//! impl LanguageModel for EchoModel {
//!     fn provider_id(&self) -> &str { "echo" }
//!     fn model_id(&self) -> &str { "echo-1" }
//!     async fn stream(
//!         &self,
//!         _req: LanguageModelRequest,
//!         _cancel: CancellationToken,
//!     ) -> Result<BoxedEventStream, ModelError> {
//!         let s = futures_util::stream::iter(vec![
//!             Ok(LanguageModelEvent::TextDelta { delta: "hi".into() }),
//!             Ok(LanguageModelEvent::Finish {
//!                 reason: FinishReason::Stop,
//!                 usage: Usage::default(),
//!             }),
//!         ]);
//!         Ok(Box::pin(s))
//!     }
//! }
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod error;
mod event;
mod message;
mod request;
mod tool;

use std::pin::Pin;

use async_trait::async_trait;
use futures_core::Stream;
use tokio_util::sync::CancellationToken;

pub use crate::error::ModelError;
pub use crate::event::{FinishReason, LanguageModelEvent, Usage};
pub use crate::message::{ContentBlock, Message, Role};
pub use crate::request::LanguageModelRequest;
pub use crate::tool::{ToolChoice, ToolSpec};

/// Boxed, `'static`, `Send` stream of streaming events.
///
/// Aliased so the trait signature stays FFI-friendly (no lifetimes, no GATs)
/// and so implementers do not have to spell out the long type every time.
pub type BoxedEventStream =
    Pin<Box<dyn Stream<Item = Result<LanguageModelEvent, ModelError>> + Send + 'static>>;

/// Trait every LLM provider implements.
///
/// The single required method, [`Self::stream`], yields a stream of
/// [`LanguageModelEvent`]s for one model invocation. Non-streaming usage is
/// implemented at a higher level by collecting the stream — providers MUST NOT
/// expose a second `complete`-style method here. Keeping the surface to one
/// method is the contract that makes the spec FFI-safe.
///
/// Implementations must be `Send + Sync` so they can be shared across tasks
/// and stored as `Arc<dyn LanguageModel>`.
#[async_trait]
pub trait LanguageModel: Send + Sync {
    /// Stable provider identifier, e.g. `"anthropic"`, `"openai"`, `"gemini"`.
    /// Used by hooks, logging, and the `provider_metadata` namespacing scheme.
    fn provider_id(&self) -> &str;

    /// Stable model identifier as the provider understands it,
    /// e.g. `"claude-opus-4-7"`, `"gpt-5-turbo"`.
    fn model_id(&self) -> &str;

    /// Begin a streamed model invocation.
    ///
    /// On success, the returned stream emits zero or more delta events and
    /// terminates with exactly one [`LanguageModelEvent::Finish`] or
    /// [`LanguageModelEvent::Error`]. Implementations MUST honour `cancel`
    /// promptly — when cancelled, surface a final
    /// [`LanguageModelEvent::Finish`] with [`FinishReason::Cancelled`] or
    /// return [`ModelError::Cancelled`] from this call.
    async fn stream(
        &self,
        req: LanguageModelRequest,
        cancel: CancellationToken,
    ) -> Result<BoxedEventStream, ModelError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_roundtrip<T>(value: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let bytes = serde_json::to_vec(value).expect("serialize");
        let parsed: T = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(value, &parsed);
    }

    #[test]
    fn content_block_text_roundtrip() {
        assert_roundtrip(&ContentBlock::text("hello"));
    }

    #[test]
    fn content_block_tool_use_roundtrip() {
        assert_roundtrip(&ContentBlock::ToolUse {
            id: "toolu_1".into(),
            name: "bash".into(),
            input: json!({"command": "ls"}),
        });
    }

    #[test]
    fn content_block_tool_result_roundtrip() {
        assert_roundtrip(&ContentBlock::ToolResult {
            tool_use_id: "toolu_1".into(),
            content: json!("file1\nfile2"),
            is_error: None,
        });
    }

    #[test]
    fn content_block_reasoning_roundtrip() {
        assert_roundtrip(&ContentBlock::Reasoning {
            text: "step 1".into(),
            signature: None,
        });
        assert_roundtrip(&ContentBlock::Reasoning {
            text: "step 1".into(),
            signature: Some("sig_abc".into()),
        });
    }

    #[test]
    fn message_roundtrip() {
        assert_roundtrip(&Message::user("hi"));
        assert_roundtrip(&Message::system("be helpful"));
        assert_roundtrip(&Message::assistant("hello"));
    }

    #[test]
    fn request_roundtrip_with_metadata() {
        let mut req = LanguageModelRequest {
            messages: vec![Message::user("hi")],
            system: Some("be terse".into()),
            tools: vec![ToolSpec {
                name: "echo".into(),
                description: "echoes input".into(),
                input_schema: json!({"type": "object"}),
            }],
            tool_choice: ToolChoice::Required,
            temperature: Some(0.7),
            max_tokens: Some(128),
            stop: vec!["END".into()],
            ..Default::default()
        };
        req.provider_metadata
            .insert("anthropic.cache_control".into(), json!({"ttl": "5m"}));
        assert_roundtrip(&req);
    }

    #[test]
    fn event_text_delta_roundtrip() {
        assert_roundtrip(&LanguageModelEvent::TextDelta {
            delta: "hello".into(),
        });
    }

    #[test]
    fn event_reasoning_delta_roundtrip() {
        assert_roundtrip(&LanguageModelEvent::ReasoningDelta {
            delta: "step 1".into(),
        });
    }

    #[test]
    fn event_reasoning_signature_roundtrip() {
        assert_roundtrip(&LanguageModelEvent::ReasoningSignature {
            signature: "sig_abc".into(),
        });
    }

    #[test]
    fn event_error_roundtrip() {
        assert_roundtrip(&LanguageModelEvent::Error {
            message: "upstream 500".into(),
        });
    }

    #[test]
    fn event_finish_roundtrip() {
        assert_roundtrip(&LanguageModelEvent::Finish {
            reason: FinishReason::Stop,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                cached_input_tokens: 5,
            },
        });
    }

    #[test]
    fn event_tool_call_lifecycle_roundtrip() {
        assert_roundtrip(&LanguageModelEvent::ToolCallStart {
            id: "toolu_1".into(),
            name: "bash".into(),
        });
        assert_roundtrip(&LanguageModelEvent::ToolCallDelta {
            id: "toolu_1".into(),
            arguments_delta: "{\"cmd\":".into(),
        });
        assert_roundtrip(&LanguageModelEvent::ToolCallEnd {
            id: "toolu_1".into(),
            arguments: Some(json!({"cmd": "ls"})),
        });
    }

    #[test]
    fn tool_choice_default_is_auto() {
        assert_eq!(ToolChoice::default(), ToolChoice::Auto);
    }
}
