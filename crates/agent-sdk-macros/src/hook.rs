//! `#[hook]` proc-macro expansion.
//!
//! Phase 4 ships a *thin* placeholder: it validates that the annotated item is
//! an `async fn` and emits it unchanged plus a `#[doc(hidden)]` marker constant
//! so downstream tooling can detect annotated hook fns. The full `Hook` trait
//! and event dispatch land in Phase 5; the public macro path is stabilised
//! here so user code that imports `#[hook]` will not break across phases.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse2, spanned::Spanned, ItemFn};

pub(crate) fn expand(_attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let func: ItemFn = parse2(item)?;
    if func.sig.asyncness.is_none() {
        return Err(syn::Error::new(
            func.sig.fn_token.span(),
            "#[hook] requires an `async fn`",
        ));
    }
    let marker = format_ident!("__{}_HOOK_MARKER", func.sig.ident);
    Ok(quote! {
        #func

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        const #marker: () = ();
    })
}
