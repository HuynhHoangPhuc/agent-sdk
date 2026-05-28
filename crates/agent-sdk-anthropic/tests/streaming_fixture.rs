//! Integration tests: drive the provider against recorded SSE fixtures
//! served by `wiremock`. No live API calls.

use std::time::Duration;

use agent_sdk_anthropic::{Anthropic, AnthropicConfig};
use agent_sdk_language_model::{
    FinishReason, LanguageModel, LanguageModelEvent, LanguageModelRequest, Message,
};
use futures_util::StreamExt;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixture(name: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

fn req() -> LanguageModelRequest {
    let mut r = LanguageModelRequest::default();
    r.messages.push(Message::user("hi"));
    r
}

async fn mount_sse(server: &MockServer, body: String) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(server)
        .await;
}

fn provider(server: &MockServer) -> Anthropic {
    let cfg = AnthropicConfig::new("test-key").with_base_url(server.uri());
    Anthropic::with_config("claude-sonnet-4-6", cfg)
}

async fn collect(model: &Anthropic, cancel: CancellationToken) -> Vec<LanguageModelEvent> {
    let mut stream = model
        .stream(req(), cancel)
        .await
        .expect("stream should open");
    let mut out = Vec::new();
    while let Some(item) = timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("stream did not yield in time")
    {
        out.push(item.expect("event err"));
    }
    out
}

#[tokio::test]
async fn streams_text_deltas_and_usage_from_fixture() {
    let server = MockServer::start().await;
    mount_sse(&server, fixture("text_only.sse")).await;
    let p = provider(&server);

    let events = collect(&p, CancellationToken::new()).await;

    let text: String = events
        .iter()
        .filter_map(|e| match e {
            LanguageModelEvent::TextDelta { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello, world");

    let finish = events
        .iter()
        .find_map(|e| match e {
            LanguageModelEvent::Finish { reason, usage } => Some((*reason, *usage)),
            _ => None,
        })
        .expect("finish event");
    assert_eq!(finish.0, FinishReason::Stop);
    assert_eq!(finish.1.input_tokens, 12);
    assert_eq!(finish.1.output_tokens, 7);
}

#[tokio::test]
async fn surfaces_tool_use_lifecycle_with_full_args() {
    let server = MockServer::start().await;
    mount_sse(&server, fixture("tool_use.sse")).await;
    let p = provider(&server);

    let events = collect(&p, CancellationToken::new()).await;

    // Collect tool-call lifecycle for id `toolu_01`.
    let mut start_seen = false;
    let mut end_seen = false;
    let mut args = String::new();
    for e in &events {
        match e {
            LanguageModelEvent::ToolCallStart { id, name } if id == "toolu_01" => {
                assert_eq!(name, "bash");
                start_seen = true;
            }
            LanguageModelEvent::ToolCallDelta {
                id,
                arguments_delta,
            } if id == "toolu_01" => {
                args.push_str(arguments_delta);
            }
            LanguageModelEvent::ToolCallEnd { id, .. } if id == "toolu_01" => end_seen = true,
            _ => {}
        }
    }
    assert!(start_seen, "missing ToolCallStart");
    assert!(end_seen, "missing ToolCallEnd");

    let parsed: serde_json::Value =
        serde_json::from_str(&args).expect("concatenated tool args must parse");
    assert_eq!(parsed["command"], "ls");

    let finish = events
        .iter()
        .find_map(|e| match e {
            LanguageModelEvent::Finish { reason, .. } => Some(*reason),
            _ => None,
        })
        .unwrap();
    assert_eq!(finish, FinishReason::ToolUse);
}

#[tokio::test]
async fn surfaces_reasoning_deltas() {
    let server = MockServer::start().await;
    mount_sse(&server, fixture("thinking.sse")).await;
    let p = provider(&server);

    let events = collect(&p, CancellationToken::new()).await;

    let reasoning: String = events
        .iter()
        .filter_map(|e| match e {
            LanguageModelEvent::ReasoningDelta { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(reasoning.contains("Considering"));
    assert!(reasoning.contains("done thinking"));

    let text: String = events
        .iter()
        .filter_map(|e| match e {
            LanguageModelEvent::TextDelta { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Result text.");
}

#[tokio::test]
async fn http_401_maps_to_auth_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad key"))
        .mount(&server)
        .await;
    let p = provider(&server);
    let res = p.stream(req(), CancellationToken::new()).await;
    match res {
        Err(agent_sdk_language_model::ModelError::Auth(msg)) => {
            assert!(msg.contains("bad key"));
        }
        Err(e) => panic!("expected Auth error, got {e}"),
        Ok(_) => panic!("expected Auth error, got opened stream"),
    }
}

#[tokio::test]
async fn cancel_mid_stream_stops_polling() {
    let server = MockServer::start().await;
    // Body intentionally truncated so the SSE stream does not terminate by
    // itself; we rely on cancellation to end the stream.
    let body = fixture("text_only.sse")
        .split("event: message_stop")
        .next()
        .unwrap()
        .to_string();
    mount_sse(&server, body).await;
    let p = provider(&server);

    let cancel = CancellationToken::new();
    let mut stream = p.stream(req(), cancel.clone()).await.expect("open stream");

    // Pull one event, then cancel.
    let _first = timeout(Duration::from_secs(2), stream.next())
        .await
        .expect("first event")
        .expect("event present");
    cancel.cancel();

    // After cancel, the stream must terminate (either Err(Cancelled) or None)
    // within a bounded time — never hang.
    let mut saw_cancelled = false;
    while let Some(ev) = timeout(Duration::from_secs(3), stream.next())
        .await
        .expect("stream must end after cancel")
    {
        if matches!(ev, Err(agent_sdk_language_model::ModelError::Cancelled)) {
            saw_cancelled = true;
            break;
        }
    }
    assert!(
        saw_cancelled,
        "cancel should surface as ModelError::Cancelled before stream ends"
    );
}

#[tokio::test]
async fn http_429_populates_retry_after() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "7")
                .set_body_string("slow down"),
        )
        .mount(&server)
        .await;
    let p = provider(&server);
    let res = p.stream(req(), CancellationToken::new()).await;
    match res {
        Err(agent_sdk_language_model::ModelError::RateLimit {
            message,
            retry_after_ms,
        }) => {
            assert!(message.contains("slow down"));
            assert_eq!(retry_after_ms, Some(7000));
        }
        Err(e) => panic!("expected RateLimit, got {e}"),
        Ok(_) => panic!("expected RateLimit error, got opened stream"),
    }
}

#[tokio::test]
async fn malformed_sse_run_surfaces_error_and_terminates() {
    let server = MockServer::start().await;
    // 8 garbage frames in a row — exactly the threshold.
    let mut body = String::new();
    for i in 0..10 {
        body.push_str(&format!("event: garbage\ndata: not-json-{i}\n\n"));
    }
    mount_sse(&server, body).await;
    let p = provider(&server);

    let mut stream = p
        .stream(req(), CancellationToken::new())
        .await
        .expect("open stream");

    let mut saw_error = false;
    let mut saw_event_after_error = false;
    while let Some(item) = timeout(Duration::from_secs(2), stream.next())
        .await
        .expect("stream must terminate")
    {
        match item {
            Ok(LanguageModelEvent::Error { .. }) if !saw_error => saw_error = true,
            Ok(_) if saw_error => saw_event_after_error = true,
            _ => {}
        }
    }
    assert!(
        saw_error,
        "expected an Error event after malformed-frame run"
    );
    assert!(
        !saw_event_after_error,
        "spec contract: no events may follow Error"
    );
}
