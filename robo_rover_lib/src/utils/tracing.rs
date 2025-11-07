//! Centralized tracing initialization for all Dora nodes.
//!
//! This module provides a standardized way to initialize tracing for all nodes
//! in the robotic rover system. It uses thread-local subscribers to avoid conflicts
//! with Dora's own tracing infrastructure.

use tracing::subscriber::DefaultGuard;
use tracing_subscriber::EnvFilter;

/// Initialize tracing with thread-local subscriber.
///
/// This function sets up tracing with a thread-local subscriber that:
/// - Respects RUST_LOG environment variable (defaults to "info")
/// - Outputs clean, compact logs without extra metadata
/// - Avoids conflicts with Dora's global subscriber
///
/// # Returns
/// A `DefaultGuard` that keeps the subscriber active. The guard must be kept
/// in scope for the duration of the program.
///
/// # Example
/// ```no_run
/// use robo_rover_lib::init_tracing;
///
/// fn main() {
///     let _guard = init_tracing();
///     // Your node code here
/// }
/// ```
pub fn init_tracing() -> DefaultGuard {
    use tracing_subscriber::layer::SubscriberExt;

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_file(false)
        .with_line_number(false);

    let subscriber = tracing_subscriber::Registry::default()
        .with(env_filter)
        .with(fmt_layer);

    tracing::subscriber::set_default(subscriber)
}
