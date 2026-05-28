//! Integration tests for the OpenAI **Responses API** stream.

use std::time::Duration;

use agent_sdk_language_model::{
    FinishReason, LanguageModel, LanguageModelEvent, LanguageModelRequest, Message,
};
use agent_sdk_openai::{OpenAI, OpenAIApi, OpenAIConfig};
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
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(server)
        .await;
}

fn provider(server: &MockServer) -> OpenAI {
    let cfg = OpenAIConfig::new("test-key").with_base_url(server.uri());
    OpenAI::with_config("gpt-x", OpenAIApi::Responses, cfg).expect("build provider")
}

async fn collect(model: &OpenAI, cancel: CancellationToken) -> Vec<LanguageModelEvent> {
    let mut stream = model.stream(req(), cancel).await.expect("stream open");
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
async fn streams_text_deltas_and_usage_from_responses_fixture() {
    let server = MockServer::start().await;
    mount_sse(&server, fixture("responses_text.sse")).await;
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
    assert_eq!(finish.1.cached_input_tokens, 2);
}

#[tokio::test]
async fn parallel_tool_calls_surface_via_responses_api() {
    let server = MockServer::start().await;
    mount_sse(&server, fixture("responses_tool_call.sse")).await;
    let p = provider(&server);

    let events = collect(&p, CancellationToken::new()).await;

    let starts: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LanguageModelEvent::ToolCallStart { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert!(starts.contains(&"call_xyz"), "missing call_xyz: {starts:?}");
    assert!(starts.contains(&"call_two"), "missing call_two: {starts:?}");

    let args_xyz: String = events
        .iter()
        .filter_map(|e| match e {
            LanguageModelEvent::ToolCallDelta {
                id,
                arguments_delta,
            } if id == "call_xyz" => Some(arguments_delta.as_str()),
            _ => None,
        })
        .collect();
    let parsed: serde_json::Value =
        serde_json::from_str(&args_xyz).expect("call_xyz args concat must parse");
    assert_eq!(parsed["command"], "ls");

    // ToolCallEnd should carry fully-parsed arguments when the upstream
    // `output_item.done` payload included the assembled `arguments` string.
    let end_xyz = events
        .iter()
        .find_map(|e| match e {
            LanguageModelEvent::ToolCallEnd { id, arguments } if id == "call_xyz" => {
                Some(arguments.clone())
            }
            _ => None,
        })
        .expect("ToolCallEnd for call_xyz");
    assert_eq!(end_xyz.unwrap()["command"], "ls");
    let ends: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LanguageModelEvent::ToolCallEnd { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert!(ends.contains(&"call_xyz"));
    assert!(ends.contains(&"call_two"));

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
async fn responses_cancel_mid_stream_terminates() {
    let server = MockServer::start().await;
    let body = fixture("responses_text.sse")
        .split("event: response.completed")
        .next()
        .unwrap()
        .to_string();
    mount_sse(&server, body).await;
    let p = provider(&server);

    let cancel = CancellationToken::new();
    let mut stream = p.stream(req(), cancel.clone()).await.expect("open stream");

    let _first = timeout(Duration::from_secs(2), stream.next())
        .await
        .expect("first event")
        .expect("event present");
    cancel.cancel();

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
    assert!(saw_cancelled, "expected Cancelled before stream ends");
}

#[tokio::test]
async fn responses_http_429_populates_retry_after() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "3")
                .set_body_string("slow down"),
        )
        .mount(&server)
        .await;
    let p = provider(&server);
    match p.stream(req(), CancellationToken::new()).await {
        Err(agent_sdk_language_model::ModelError::RateLimit {
            message,
            retry_after_ms,
        }) => {
            assert!(message.contains("slow down"));
            assert_eq!(retry_after_ms, Some(3000));
        }
        Err(e) => panic!("expected RateLimit, got {e}"),
        Ok(_) => panic!("expected RateLimit, got stream"),
    }
}
