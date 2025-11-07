//! # Arm Bot Library
//!
//! Shared types and utilities for the robotic arm controller system.
//! This library is used by all nodes in the dora-rs dataflow.

pub mod types;
pub mod utils;

// Re-export everything for convenience
pub use types::*;
pub use utils::*;
