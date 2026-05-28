//! `Anthropic` model: implements [`LanguageModel`] against `/v1/messages`.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use agent_sdk_language_model::{
    BoxedEventStream, LanguageModel, LanguageModelEvent, LanguageModelRequest, ModelError,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};

use crate::event_map::EventMapper;
use crate::models::StreamEvent;
use crate::request_map::build_request;
use crate::sse::SseParser;
use crate::{ANTHROPIC_VERSION, DEFAULT_BASE_URL};

/// Surface `LanguageModelEvent::Error` and end the stream after this many
/// consecutive SSE frames fail to parse. Guards against a stuck connection
/// that emits nothing but malformed bytes.
const MAX_CONSECUTIVE_PARSE_ERRORS: u32 = 8;

/// Construction-time config knobs for [`Anthropic`].
///
/// Most callers should use [`Anthropic::new`] or the `claude_sonnet_4_6`
/// constructor. This struct exists for tests (custom `base_url` against
/// wiremock) and for users who want a tuned `reqwest::Client`.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Anthropic API key (`x-api-key` header).
    pub api_key: String,
    /// Base URL — defaults to [`DEFAULT_BASE_URL`].
    pub base_url: String,
    /// Connect/request timeout. `None` disables.
    pub timeout: Option<Duration>,
}

impl AnthropicConfig {
    /// Build the default config for a given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: Some(Duration::from_secs(120)),
        }
    }

    /// Override the base URL (used by tests; in prod, leave as default).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

/// Anthropic Messages-API-backed [`LanguageModel`] implementation.
pub struct Anthropic {
    model_id: String,
    config: AnthropicConfig,
    http: reqwest::Client,
}

impl Anthropic {
    /// Construct a model targeting `model_id` (e.g. `"claude-sonnet-4-6"`,
    /// `"claude-opus-4-7"`) with the supplied API key. Uses Anthropic's
    /// public base URL.
    pub fn new(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::with_config(model_id, AnthropicConfig::new(api_key))
    }

    /// Construct from an explicit [`AnthropicConfig`].
    pub fn with_config(model_id: impl Into<String>, config: AnthropicConfig) -> Self {
        let mut builder = reqwest::Client::builder();
        if let Some(t) = config.timeout {
            builder = builder.timeout(t);
        }
        let http = builder.build().unwrap_or_else(|_| reqwest::Client::new());
        Self {
            model_id: model_id.into(),
            config,
            http,
        }
    }

    fn headers(&self) -> Result<HeaderMap, ModelError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        h.insert(
            "x-api-key",
            HeaderValue::from_str(&self.config.api_key)
                .map_err(|_| ModelError::Auth("invalid API key header".into()))?,
        );
        Ok(h)
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl LanguageModel for Anthropic {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn stream(
        &self,
        req: LanguageModelRequest,
        cancel: CancellationToken,
    ) -> Result<BoxedEventStream, ModelError> {
        let body = build_request(&self.model_id, &req)?;
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| ModelError::InvalidRequest(format!("serialize body: {e}")))?;
        let headers = self.headers()?;
        let url = self.endpoint();

        let send_fut = self
            .http
            .post(&url)
            .headers(headers)
            .body(body_bytes)
            .send();

        let resp = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(ModelError::Cancelled),
            r = send_fut => r.map_err(map_reqwest_err)?,
        };

        let status = resp.status();
        if !status.is_success() {
            let headers = resp.headers().clone();
            let snippet = resp.text().await.unwrap_or_default();
            return Err(map_http_error(status, &headers, snippet));
        }

        let bytes_stream = resp.bytes_stream();
        let cancel_fut = Box::pin(cancel.clone().cancelled_owned());
        Ok(Box::pin(EventStream {
            inner: bytes_stream,
            parser: SseParser::new(),
            mapper: EventMapper::new(),
            buffered: VecDeque::new(),
            done: false,
            consecutive_parse_errors: 0,
            cancel_fut,
            _cancel: cancel,
        }))
    }
}

fn map_reqwest_err(e: reqwest::Error) -> ModelError {
    if e.is_timeout() {
        ModelError::Transport(format!("timeout: {e}"))
    } else if e.is_connect() {
        ModelError::Transport(format!("connect: {e}"))
    } else {
        ModelError::Transport(e.to_string())
    }
}

fn map_http_error(status: reqwest::StatusCode, headers: &HeaderMap, body: String) -> ModelError {
    match status.as_u16() {
        401 | 403 => ModelError::Auth(body),
        429 => ModelError::RateLimit {
            message: body,
            retry_after_ms: parse_retry_after_ms(headers),
        },
        413 => ModelError::ContextLengthExceeded(body),
        400 => ModelError::InvalidRequest(body),
        408 => ModelError::Transport(format!("HTTP 408: {body}")),
        500..=599 => ModelError::Provider(format!("HTTP {}: {}", status.as_u16(), body)),
        _ => ModelError::Other(format!("HTTP {}: {}", status.as_u16(), body)),
    }
}

/// Parse the `Retry-After` header into milliseconds.
///
/// Supports the common delta-seconds form (e.g. `Retry-After: 30`). HTTP-date
/// form (RFC 7231 §7.1.3) is accepted by the spec but rare; we ignore it here
/// and rely on the caller's own backoff.
fn parse_retry_after_ms(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|secs| secs.saturating_mul(1000))
}

pin_project_lite::pin_project! {
    /// Adapter from a `reqwest` byte stream into the spec's
    /// `LanguageModelEvent` stream — drives SSE parse, event mapping, and
    /// cancellation in a single `Stream` impl.
    ///
    /// Cancellation is polled alongside the byte stream so a mid-body cancel
    /// drops the connection at the next poll wake-up instead of waiting for
    /// the next chunk to arrive.
    struct EventStream<S> {
        #[pin]
        inner: S,
        parser: SseParser,
        mapper: EventMapper,
        buffered: VecDeque<LanguageModelEvent>,
        done: bool,
        consecutive_parse_errors: u32,
        // Pinned cancellation future — its waker is registered on every poll
        // so cancel() wakes us even while `inner.poll_next` is parked.
        cancel_fut: Pin<Box<WaitForCancellationFutureOwned>>,
        // Owned clone kept alive for callers that may still hold the token;
        // also lets us inspect cancellation state without a stale waker.
        _cancel: CancellationToken,
    }
}

impl<S> Stream for EventStream<S>
where
    S: Stream<Item = reqwest::Result<Bytes>>,
{
    type Item = Result<LanguageModelEvent, ModelError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            if *this.done {
                return Poll::Ready(None);
            }
            if let Some(ev) = this.buffered.pop_front() {
                // Spec contract: no events follow `Error`. End the stream
                // immediately after surfacing one.
                if matches!(ev, LanguageModelEvent::Error { .. }) {
                    *this.done = true;
                }
                return Poll::Ready(Some(Ok(ev)));
            }
            // Poll cancel first so its waker is registered — if the token
            // is cancelled mid-poll, cx will be woken even while `inner` is
            // parked on the network.
            if this.cancel_fut.as_mut().poll(cx).is_ready() {
                *this.done = true;
                return Poll::Ready(Some(Err(ModelError::Cancelled)));
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    *this.done = true;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => {
                    *this.done = true;
                    return Poll::Ready(Some(Err(map_reqwest_err(e))));
                }
                Poll::Ready(Some(Ok(chunk))) => {
                    this.parser.push(&chunk);
                    for frame in this.parser.drain() {
                        if frame.data.trim() == "[DONE]" || frame.data.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<StreamEvent>(&frame.data) {
                            Ok(ev) => {
                                *this.consecutive_parse_errors = 0;
                                for mapped in this.mapper.map(ev) {
                                    this.buffered.push_back(mapped);
                                }
                            }
                            Err(e) => {
                                tracing::debug!(
                                    target: "agent_sdk_anthropic",
                                    error = %e,
                                    data = %frame.data,
                                    "dropping malformed SSE frame"
                                );
                                *this.consecutive_parse_errors += 1;
                                if *this.consecutive_parse_errors >= MAX_CONSECUTIVE_PARSE_ERRORS {
                                    let msg = format!(
                                        "{} consecutive malformed SSE frames; last error: {e}",
                                        *this.consecutive_parse_errors
                                    );
                                    this.buffered
                                        .push_back(LanguageModelEvent::Error { message: msg });
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_sdk_language_model::Message;

    #[test]
    fn endpoint_strips_trailing_slash() {
        let cfg = AnthropicConfig::new("k").with_base_url("http://x/");
        let a = Anthropic::with_config("m", cfg);
        assert_eq!(a.endpoint(), "http://x/v1/messages");
    }

    #[test]
    fn provider_and_model_ids() {
        let a = Anthropic::new("claude-sonnet-4-6", "k");
        assert_eq!(a.provider_id(), "anthropic");
        assert_eq!(a.model_id(), "claude-sonnet-4-6");
    }

    #[tokio::test]
    async fn cancel_before_send_returns_cancelled() {
        let a = Anthropic::with_config(
            "m",
            AnthropicConfig::new("k").with_base_url("http://127.0.0.1:1"),
        );
        let cancel = CancellationToken::new();
        cancel.cancel();
        let mut req = LanguageModelRequest::default();
        req.messages.push(Message::user("hi"));
        let res = a.stream(req, cancel).await;
        match res {
            Err(ModelError::Cancelled) => {}
            Err(other) => panic!("expected Cancelled, got {other:?}"),
            Ok(_) => panic!("expected Cancelled, got stream"),
        }
    }
}
