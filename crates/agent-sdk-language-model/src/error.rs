//! Error types returned by [`LanguageModel`](crate::LanguageModel) implementations.

use thiserror::Error;

/// Errors that a [`LanguageModel`](crate::LanguageModel) implementation may return.
///
/// Variants are intentionally coarse: providers should map their wire errors onto
/// the closest variant and put provider-specific detail in the `message` payload.
/// The enum is `#[non_exhaustive]` so new variants can be added without a major bump.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelError {
    /// The request was rejected as malformed before reaching the provider.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The provider refused the request due to missing or invalid credentials.
    #[error("authentication failed: {0}")]
    Auth(String),

    /// The provider applied a rate limit. `retry_after_ms` is advisory.
    #[error("rate limited: {message}")]
    RateLimit {
        /// Human-readable rate-limit message from the provider.
        message: String,
        /// Suggested backoff in milliseconds, if the provider supplied one.
        retry_after_ms: Option<u64>,
    },

    /// The request context exceeded the model's allowed length.
    #[error("context length exceeded: {0}")]
    ContextLengthExceeded(String),

    /// The provider returned a recoverable upstream error (e.g., HTTP 5xx).
    #[error("provider error: {0}")]
    Provider(String),

    /// Network / transport failure (connection reset, DNS, TLS).
    #[error("transport error: {0}")]
    Transport(String),

    /// The request was cancelled via the supplied
    /// [`CancellationToken`](tokio_util::sync::CancellationToken).
    #[error("request cancelled")]
    Cancelled,

    /// Catch-all for anything that does not fit the variants above.
    #[error("{0}")]
    Other(String),
}
