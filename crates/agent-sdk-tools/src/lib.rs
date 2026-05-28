//! # `agent-sdk-tools`
//!
//! Batteries-included coding toolkit: `bash`, `read`, `write`, `edit`, `grep`,
//! `glob`, `ls`. Every tool implements [`agent_sdk_core::Tool`] and shares a
//! single [`SandboxRoot`] so paths and shell cwds cannot escape a chosen
//! directory unless the caller explicitly opts out via
//! [`SandboxRoot::unrestricted`].
//!
//! Minimal usage:
//!
//! ```ignore
//! use agent_sdk_tools::{coding_toolkit, SandboxRoot};
//!
//! let sandbox = SandboxRoot::cwd()?;
//! let tools = coding_toolkit(sandbox);
//! // pass `tools` into AgentBuilder::tools(...)
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod bash;
mod edit;
mod glob;
mod grep;
mod ls;
mod read;
mod sandbox;
mod write;

use std::sync::Arc;

use agent_sdk_core::Tool;

pub use crate::bash::BashTool;
pub use crate::edit::EditTool;
pub use crate::glob::GlobTool;
pub use crate::grep::GrepTool;
pub use crate::ls::LsTool;
pub use crate::read::ReadTool;
pub use crate::sandbox::{SandboxError, SandboxRoot};
pub use crate::write::WriteTool;

/// Build the full coding toolkit (`bash`, `read`, `write`, `edit`, `grep`,
/// `glob`, `ls`) sharing the given [`SandboxRoot`].
pub fn coding_toolkit(sandbox: SandboxRoot) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(BashTool::new(sandbox.clone())) as Arc<dyn Tool>,
        Arc::new(ReadTool::new(sandbox.clone())),
        Arc::new(WriteTool::new(sandbox.clone())),
        Arc::new(EditTool::new(sandbox.clone())),
        Arc::new(GrepTool::new(sandbox.clone())),
        Arc::new(GlobTool::new(sandbox.clone())),
        Arc::new(LsTool::new(sandbox)),
    ]
}
