//! Procedural macros for `agent-sdk`.
//!
//! - [`macro@tool`] turns an `async fn` into a unit struct implementing
//!   [`agent_sdk_core::Tool`], with the JSON Schema derived from the fn
//!   parameters and doc comments.
//! - [`macro@hook`] is a thin placeholder stabilised in Phase 4; the full
//!   `Hook` trait and event dispatch land in Phase 5.
//!
//! Both macros emit absolute paths through `::agent_sdk_core::__macros::*` so
//! the consuming crate only needs `agent-sdk-core` in scope. See
//! `agent-sdk-core/src/lib.rs` for the re-export module.

use proc_macro::TokenStream;

mod docs;
mod hook;
mod tool;

/// Annotate an `async fn` to register it as a tool the agent can call.
///
/// ```ignore
/// use agent_sdk_core::{tool, ToolResult, AgentError};
///
/// /// Echo a string back to the caller.
/// #[tool]
/// async fn echo(
///     /// Text the model wants to echo.
///     text: String,
/// ) -> Result<ToolResult, AgentError> {
///     Ok(ToolResult::text(text))
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    tool::expand(attr.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Annotate an `async fn` as a hook event handler.
///
/// Phase 4 ships a forward-compatible placeholder; the full Hook trait lands
/// in Phase 5. Today the macro validates the fn is async and emits it
/// unchanged plus a hidden marker constant.
#[proc_macro_attribute]
pub fn hook(attr: TokenStream, item: TokenStream) -> TokenStream {
    hook::expand(attr.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
