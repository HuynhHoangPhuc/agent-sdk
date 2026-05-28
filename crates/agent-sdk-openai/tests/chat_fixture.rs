//! Integration tests for the OpenAI **Chat Completions** stream.
//! Drives the provider against recorded SSE fixtures served by `wiremock`.

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
        .and(path("/v1/chat/completions"))
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
    OpenAI::with_config("gpt-x", OpenAIApi::Chat, cfg).expect("build provider")
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
async fn streams_text_deltas_and_usage_from_chat_fixture() {
    let server = MockServer::start().await;
    mount_sse(&server, fixture("chat_text.sse")).await;
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
    assert_eq!(finish.1.cached_input_tokens, 3);
}

#[tokio::test]
async fn parallel_tool_calls_surface_as_multiple_ids() {
    let server = MockServer::start().await;
    mount_sse(&server, fixture("chat_tool_call.sse")).await;
    let p = provider(&server);

    let events = collect(&p, CancellationToken::new()).await;

    // Two distinct ToolCallStart ids — proves parallel tool calls are surfaced.
    let mut starts: Vec<&str> = Vec::new();
    for e in &events {
        if let LanguageModelEvent::ToolCallStart { id, .. } = e {
            starts.push(id);
        }
    }
    assert!(starts.contains(&"call_abc"), "missing call_abc: {starts:?}");
    assert!(starts.contains(&"call_def"), "missing call_def: {starts:?}");

    // Reconstruct call_abc arguments.
    let args_abc: String = events
        .iter()
        .filter_map(|e| match e {
            LanguageModelEvent::ToolCallDelta {
                id,
                arguments_delta,
            } if id == "call_abc" => Some(arguments_delta.as_str()),
            _ => None,
        })
        .collect();
    let parsed: serde_json::Value =
        serde_json::from_str(&args_abc).expect("call_abc args concat must parse");
    assert_eq!(parsed["command"], "ls");

    // Finish reason is ToolUse.
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
async fn chat_http_401_maps_to_auth_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("bad key"))
        .mount(&server)
        .await;
    let p = provider(&server);
    match p.stream(req(), CancellationToken::new()).await {
        Err(agent_sdk_language_model::ModelError::Auth(msg)) => {
            assert!(msg.contains("bad key"));
        }
        Err(e) => panic!("expected Auth, got {e}"),
        Ok(_) => panic!("expected Auth, got stream"),
    }
}

#[tokio::test]
async fn chat_cancel_mid_stream_terminates() {
    let server = MockServer::start().await;
    // Truncate before [DONE] so the connection won't terminate by itself.
    let body = fixture("chat_text.sse")
        .split("data: [DONE]")
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
