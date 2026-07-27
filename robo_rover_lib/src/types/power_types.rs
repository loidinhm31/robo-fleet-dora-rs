//! Versioned, actor-free wire contracts for workload-level power coordination.

mod command;
mod demand;
mod signed_command;
mod status;
pub(crate) mod validation;

pub use command::*;
pub use demand::*;
pub use signed_command::*;
pub use status::*;

pub const POWER_PROTOCOL_VERSION: u8 = 1;
pub const MAX_POWER_DEMANDS: usize = 128;

/// Fixed V1 Zenoh channels for coordinator-to-coordinator power transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerTopic {
    Command,
    Status,
    Snapshot,
    SnapshotRequest,
    Event,
    EventAck,
}

pub fn power_v1_topic(entity_id: &str, topic: PowerTopic) -> String {
    let suffix = match topic {
        PowerTopic::Command => "command",
        PowerTopic::Status => "status",
        PowerTopic::Snapshot => "snapshot",
        PowerTopic::SnapshotRequest => "snapshot-request",
        PowerTopic::Event => "event",
        PowerTopic::EventAck => "event-ack",
    };
    format!("rover/{entity_id}/power/v1/{suffix}")
}
