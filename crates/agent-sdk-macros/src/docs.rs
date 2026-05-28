//! Extract doc-comment text from `#[doc = "..."]` attributes.
//!
//! `///` and `//!` desugar to `#[doc = "..."]` attributes that `syn` exposes
//! on items, struct fields, and fn parameters. We concatenate the strings and
//! trim a single leading space (the convention rustdoc inserts) per line.

use syn::{Attribute, Expr, ExprLit, Lit, Meta};

/// Concatenate `#[doc = "..."]` attribute values into a single string,
/// separated by newlines. Returns `None` if no doc attributes are present.
pub(crate) fn extract(attrs: &[Attribute]) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) = &nv.value
        else {
            continue;
        };
        let raw = s.value();
        // Strip a single leading space if present (rustdoc convention).
        let trimmed = raw.strip_prefix(' ').unwrap_or(&raw);
        lines.push(trimmed.to_string());
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n").trim().to_string())
    }
}
