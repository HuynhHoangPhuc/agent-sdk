//! OpenAI HTTP client + [`LanguageModel`] implementation.
//!
//! One [`OpenAI`] struct services both the Chat Completions and Responses APIs
//! via the [`OpenAIApi`] selector — request building and event mapping are
//! dispatched per-API in `stream`, but the HTTP, SSE, and cancellation
//! plumbing is shared.

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
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};

use crate::chat::{self, ChatStreamChunk};
use crate::event_map_chat::ChatEventMapper;
use crate::event_map_responses::ResponsesEventMapper;
use crate::responses;
use crate::sse::SseParser;
use crate::DEFAULT_BASE_URL;

/// End the stream after this many consecutive SSE parse errors. Same
/// reasoning as the Anthropic provider's identical knob.
const MAX_CONSECUTIVE_PARSE_ERRORS: u32 = 8;

/// Which OpenAI API a given [`OpenAI`] instance targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAIApi {
    /// `POST /v1/chat/completions`.
    Chat,
    /// `POST /v1/responses`.
    Responses,
}

/// Construction-time config for [`OpenAI`].
///
/// `api_key` is redacted in `Debug` to keep it out of logs.
#[derive(Clone)]
pub struct OpenAIConfig {
    /// OpenAI API key sent as `Authorization: Bearer <key>`.
    pub api_key: String,
    /// Base URL — defaults to [`DEFAULT_BASE_URL`].
    pub base_url: String,
    /// Connect/request timeout. `None` disables.
    pub timeout: Option<Duration>,
    /// Optional `OpenAI-Organization` header value.
    pub organization: Option<String>,
}

impl std::fmt::Debug for OpenAIConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIConfig")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("organization", &self.organization)
            .finish()
    }
}

impl OpenAIConfig {
    /// Build the default config for a given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: Some(Duration::from_secs(120)),
            organization: None,
        }
    }

    /// Override the base URL (used by tests; production should keep the default).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the `OpenAI-Organization` header.
    #[must_use]
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
}

/// OpenAI-backed [`LanguageModel`] for either Chat Completions or Responses.
#[derive(Debug, Clone)]
pub struct OpenAI {
    model_id: String,
    api: OpenAIApi,
    config: OpenAIConfig,
    http: reqwest::Client,
}

impl OpenAI {
    /// Construct a [`OpenAI`] for the Chat Completions API.
    pub fn chat(
        model_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self, ModelError> {
        Self::with_config(model_id, OpenAIApi::Chat, OpenAIConfig::new(api_key))
    }

    /// Construct a [`OpenAI`] for the Responses API.
    pub fn responses(
        model_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self, ModelError> {
        Self::with_config(model_id, OpenAIApi::Responses, OpenAIConfig::new(api_key))
    }

    /// Construct from an explicit [`OpenAIConfig`] and [`OpenAIApi`] selection.
    pub fn with_config(
        model_id: impl Into<String>,
        api: OpenAIApi,
        config: OpenAIConfig,
    ) -> Result<Self, ModelError> {
        let mut builder = reqwest::Client::builder();
        if let Some(t) = config.timeout {
            builder = builder.timeout(t);
        }
        let http = builder
            .build()
            .map_err(|e| ModelError::Other(format!("reqwest client build: {e}")))?;
        Ok(Self {
            model_id: model_id.into(),
            api,
            config,
            http,
        })
    }

    /// API selector this instance targets.
    pub fn api(&self) -> OpenAIApi {
        self.api
    }

    fn headers(&self) -> Result<HeaderMap, ModelError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let bearer = format!("Bearer {}", self.config.api_key);
        h.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&bearer)
                .map_err(|_| ModelError::Auth("invalid API key header".into()))?,
        );
        if let Some(org) = self.config.organization.as_ref() {
            h.insert(
                "openai-organization",
                HeaderValue::from_str(org)
                    .map_err(|_| ModelError::InvalidRequest("invalid organization header".into()))?,
            );
        }
        Ok(h)
    }

    fn endpoint(&self) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        match self.api {
            OpenAIApi::Chat => format!("{base}/v1/chat/completions"),
            OpenAIApi::Responses => format!("{base}/v1/responses"),
        }
    }

    fn build_body(&self, req: &LanguageModelRequest) -> Result<Vec<u8>, ModelError> {
        match self.api {
            OpenAIApi::Chat => {
                let body = chat::build_request(&self.model_id, req)?;
                serde_json::to_vec(&body)
                    .map_err(|e| ModelError::InvalidRequest(format!("serialize body: {e}")))
            }
            OpenAIApi::Responses => {
                let body = responses::build_request(&self.model_id, req)?;
                serde_json::to_vec(&body)
                    .map_err(|e| ModelError::InvalidRequest(format!("serialize body: {e}")))
            }
        }
    }
}

#[async_trait]
impl LanguageModel for OpenAI {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn stream(
        &self,
        req: LanguageModelRequest,
        cancel: CancellationToken,
    ) -> Result<BoxedEventStream, ModelError> {
        let body = self.build_body(&req)?;
        let headers = self.headers()?;
        let url = self.endpoint();

        let send_fut = self
            .http
            .post(&url)
            .headers(headers)
            .body(body)
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
        let cancel_fut = Box::pin(cancel.cancelled_owned());
        let mapper = match self.api {
            OpenAIApi::Chat => Mapper::Chat(ChatEventMapper::new()),
            OpenAIApi::Responses => Mapper::Responses(ResponsesEventMapper::new()),
        };
        Ok(Box::pin(EventStream {
            inner: bytes_stream,
            parser: SseParser::new(),
            mapper,
            buffered: VecDeque::new(),
            done: false,
            consecutive_parse_errors: 0,
            cancel_fut,
        }))
    }
}

/// Routes parsed SSE frames into either the Chat or Responses mapper.
enum Mapper {
    Chat(ChatEventMapper),
    Responses(ResponsesEventMapper),
}

impl Mapper {
    fn handle_frame(&mut self, event: &str, data: &str) -> Result<Vec<LanguageModelEvent>, String> {
        match self {
            Mapper::Chat(m) => {
                let chunk = serde_json::from_str::<ChatStreamChunk>(data)
                    .map_err(|e| e.to_string())?;
                Ok(m.map(chunk))
            }
            Mapper::Responses(m) => {
                let ev = responses::parse_event(event, data).map_err(|e| e.to_string())?;
                Ok(m.map(ev))
            }
        }
    }

    fn flush(&mut self) -> Vec<LanguageModelEvent> {
        match self {
            Mapper::Chat(m) => m.flush(),
            Mapper::Responses(m) => m.flush(),
        }
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

fn parse_retry_after_ms(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|secs| secs.saturating_mul(1000))
}

pin_project_lite::pin_project! {
    /// Stream adapter from `reqwest` bytes → spec events.
    ///
    /// Identical wake-up + cancel preempt discipline as the Anthropic
    /// provider's EventStream — see that crate for the rationale. Notable
    /// difference: when the byte stream ends, we drain the mapper's
    /// `flush()` so OpenAI's terminator-less Chat protocol still emits a
    /// terminal `Finish`.
    struct EventStream<S> {
        #[pin]
        inner: S,
        parser: SseParser,
        mapper: Mapper,
        buffered: VecDeque<LanguageModelEvent>,
        done: bool,
        consecutive_parse_errors: u32,
        cancel_fut: Pin<Box<WaitForCancellationFutureOwned>>,
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
            if this.cancel_fut.as_mut().poll(cx).is_ready() {
                *this.done = true;
                return Poll::Ready(Some(Err(ModelError::Cancelled)));
            }
            if let Some(ev) = this.buffered.pop_front() {
                if matches!(ev, LanguageModelEvent::Error { .. }) {
                    *this.done = true;
                }
                return Poll::Ready(Some(Ok(ev)));
            }
            match this.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    // Drain any synthetic terminal events from the mapper
                    // (Chat: ToolCallEnd + Finish; Responses: Finish if upstream
                    // didn't send response.completed).
                    for ev in this.mapper.flush() {
                        this.buffered.push_back(ev);
                    }
                    if this.buffered.is_empty() {
                        *this.done = true;
                        return Poll::Ready(None);
                    }
                    // Loop to drain buffered.
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
                        match this.mapper.handle_frame(&frame.event, &frame.data) {
                            Ok(events) => {
                                *this.consecutive_parse_errors = 0;
                                for ev in events {
                                    this.buffered.push_back(ev);
                                }
                            }
                            Err(err) => {
                                tracing::debug!(
                                    target: "agent_sdk_openai",
                                    error = %err,
                                    event = %frame.event,
                                    data = %frame.data,
                                    "dropping malformed SSE frame"
                                );
                                *this.consecutive_parse_errors += 1;
                                if *this.consecutive_parse_errors >= MAX_CONSECUTIVE_PARSE_ERRORS {
                                    let msg = format!(
                                        "{} consecutive malformed SSE frames; last error: {err}",
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
    fn endpoint_chat_strips_trailing_slash() {
        let cfg = OpenAIConfig::new("k").with_base_url("http://x/");
        let m = OpenAI::with_config("gpt-x", OpenAIApi::Chat, cfg).unwrap();
        assert_eq!(m.endpoint(), "http://x/v1/chat/completions");
    }

    #[test]
    fn endpoint_responses() {
        let cfg = OpenAIConfig::new("k").with_base_url("http://x");
        let m = OpenAI::with_config("gpt-x", OpenAIApi::Responses, cfg).unwrap();
        assert_eq!(m.endpoint(), "http://x/v1/responses");
    }

    #[test]
    fn provider_and_model_ids() {
        let m = OpenAI::chat("gpt-x", "k").unwrap();
        assert_eq!(m.provider_id(), "openai");
        assert_eq!(m.model_id(), "gpt-x");
    }

    #[test]
    fn debug_redacts_api_key() {
        let cfg = OpenAIConfig::new("sk-super-secret");
        let s = format!("{cfg:?}");
        assert!(!s.contains("sk-super-secret"), "key leaked: {s}");
        assert!(s.contains("<redacted>"));
    }

    #[tokio::test]
    async fn cancel_before_send_returns_cancelled() {
        let cfg = OpenAIConfig::new("k").with_base_url("http://127.0.0.1:1");
        let m = OpenAI::with_config("gpt-x", OpenAIApi::Chat, cfg).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let mut req = LanguageModelRequest::default();
        req.messages.push(Message::user("hi"));
        match m.stream(req, cancel).await {
            Err(ModelError::Cancelled) => {}
            Err(other) => panic!("expected Cancelled, got {other:?}"),
            Ok(_) => panic!("expected Cancelled, got stream"),
        }
    }
}
