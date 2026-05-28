//! Compile-fail diagnostics for `#[tool]`.
//!
//! Each `.rs` file under `tests/trybuild/` deliberately misuses the macro;
//! `trybuild` compiles it and diffs the captured `stderr` against the
//! sibling `.stderr` file. Run `cargo test --test compile_fail` to refresh
//! (`TRYBUILD=overwrite` to accept new output).

#[test]
fn compile_fail_cases() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/trybuild/non_async_fn.rs");
    t.compile_fail("tests/trybuild/reference_param.rs");
    t.compile_fail("tests/trybuild/generic_fn.rs");
}
