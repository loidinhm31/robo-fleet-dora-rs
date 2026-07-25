//! Versioned, actor-free wire contracts for workload-level power coordination.

mod command;
mod demand;
mod status;
pub(crate) mod validation;

pub use command::*;
pub use demand::*;
pub use status::*;

pub const POWER_PROTOCOL_VERSION: u8 = 1;
pub const MAX_POWER_DEMANDS: usize = 128;
