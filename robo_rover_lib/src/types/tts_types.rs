//! Version-stable edge-voice contracts shared by Dora and Zenoh nodes.

mod command;
mod config;
mod lifecycle;
mod lifecycle_validation;
mod validation;

pub use command::*;
pub use config::*;
pub use lifecycle::*;
pub use validation::sanitize_external_detail;

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod tests;
