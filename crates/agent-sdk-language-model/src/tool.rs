//! Tool/function specifications advertised to the model.

use serde::{Deserialize, Serialize};

/// Description of a tool the model is permitted to call.
///
/// `input_schema` is a JSON Schema object describing the tool's argument shape.
/// Providers translate this to their native function/tool definition format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolSpec {
    /// Tool name. Must be unique within a request.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema (draft 2020-12 compatible) describing the input arguments.
    pub input_schema: serde_json::Value,
}

impl ToolSpec {
    /// Construct a [`ToolSpec`] from name, description, and a JSON Schema.
    ///
    /// Use this in cross-crate code (provider crates, user code) because the
    /// struct is `#[non_exhaustive]` — struct literals are forbidden outside
    /// this crate, but `ToolSpec::new(..)` works everywhere.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// How aggressively the model should select tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// Default: the model decides whether to call a tool.
    #[default]
    Auto,
    /// The model must call at least one tool (any tool).
    Required,
    /// The model must call the named tool.
    Tool {
        /// Name of the specific tool to force-call.
        name: String,
    },
    /// The model must NOT call any tool this turn.
    None,
}
