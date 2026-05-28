//! grep + glob + ls behavior; .gitignore is honored.

use agent_sdk_core::Tool;
use agent_sdk_tools::{GlobTool, GrepTool, LsTool, SandboxRoot};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn populate(tmp: &std::path::Path) {
    std::fs::write(tmp.join(".gitignore"), "ignored.txt\n").unwrap();
    std::fs::write(tmp.join("kept.rs"), "fn marker_kept() {}\n").unwrap();
    std::fs::write(tmp.join("ignored.txt"), "marker_ignored\n").unwrap();
    std::fs::create_dir(tmp.join("sub")).unwrap();
    std::fs::write(tmp.join("sub/inner.rs"), "fn marker_inner() {}\n").unwrap();
}

#[tokio::test]
async fn grep_finds_match_and_skips_gitignored() {
    let tmp = tempfile::tempdir().unwrap();
    populate(tmp.path());
    let sb = SandboxRoot::at(tmp.path()).unwrap();

    let result = GrepTool::new(sb)
        .execute(
            json!({"pattern": "marker_"}),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    let hits = result.content["hits"].as_array().unwrap();
    let texts: Vec<&str> = hits.iter().map(|h| h["text"].as_str().unwrap()).collect();
    assert!(texts.iter().any(|t| t.contains("marker_kept")));
    assert!(texts.iter().any(|t| t.contains("marker_inner")));
    assert!(
        texts.iter().all(|t| !t.contains("marker_ignored")),
        "gitignored file should be skipped: {texts:?}"
    );
}

#[tokio::test]
async fn glob_matches_recursive_pattern() {
    let tmp = tempfile::tempdir().unwrap();
    populate(tmp.path());
    let sb = SandboxRoot::at(tmp.path()).unwrap();

    let result = GlobTool::new(sb)
        .execute(json!({"pattern": "**/*.rs"}), CancellationToken::new())
        .await
        .unwrap();

    let paths = result.content["paths"].as_array().unwrap();
    assert!(paths.iter().any(|p| p.as_str().unwrap().ends_with("kept.rs")));
    assert!(paths.iter().any(|p| p.as_str().unwrap().ends_with("inner.rs")));
}

#[tokio::test]
async fn ls_respects_depth_cap() {
    let tmp = tempfile::tempdir().unwrap();
    populate(tmp.path());
    let sb = SandboxRoot::at(tmp.path()).unwrap();

    let result = LsTool::new(sb)
        .execute(json!({"depth": 1}), CancellationToken::new())
        .await
        .unwrap();

    let entries = result.content["entries"].as_array().unwrap();
    // depth=1 should include "sub/" but not "sub/inner.rs"
    assert!(entries.iter().any(|e| e["path"].as_str().unwrap().ends_with("sub")));
    assert!(entries
        .iter()
        .all(|e| !e["path"].as_str().unwrap().ends_with("inner.rs")));
}
