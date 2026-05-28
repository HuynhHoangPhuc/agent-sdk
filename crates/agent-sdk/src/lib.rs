//! `agent-sdk` facade. Stub crate — full re-exports + examples land in Phase 12.
//!
//! For now, re-exports the spec crate so consumers can begin against the
//! stable `LanguageModel` trait without depending on `agent-sdk-language-model`
//! directly.

#![warn(missing_docs)]

pub use agent_sdk_language_model as language_model;
