use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Fleet status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetStatus {
    /// Currently selected rover entity ID
    pub selected_entity: String,
    /// List of all available rover entity IDs
    pub fleet_roster: Vec<String>,
    /// Timestamp when status was generated
    pub timestamp: u64,
}

impl FleetStatus {
    pub fn new(selected_entity: String, fleet_roster: Vec<String>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            selected_entity,
            fleet_roster,
            timestamp,
        }
    }
}

/// Fleet selection command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetSelectCommand {
    /// Entity ID to select
    pub entity_id: String,
    /// Timestamp of the command
    pub timestamp: u64,
}

impl FleetSelectCommand {
    pub fn new(entity_id: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            entity_id,
            timestamp,
        }
    }
}

/// Individual rover status in the fleet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoverStatus {
    /// Rover entity ID
    pub entity_id: String,
    /// Whether this rover is currently selected
    pub is_selected: bool,
    /// Connection status (connected/disconnected)
    pub is_connected: bool,
    /// Last telemetry received timestamp
    pub last_seen: Option<u64>,
    /// Battery level (0.0-100.0)
    pub battery_level: Option<f32>,
    /// Signal strength (0.0-100.0)
    pub signal_strength: Option<f32>,
}

impl RoverStatus {
    pub fn new(entity_id: String, is_selected: bool) -> Self {
        Self {
            entity_id,
            is_selected,
            is_connected: false,
            last_seen: None,
            battery_level: None,
            signal_strength: None,
        }
    }

    pub fn with_connection(mut self, is_connected: bool) -> Self {
        self.is_connected = is_connected;
        self
    }

    pub fn with_last_seen(mut self, timestamp: u64) -> Self {
        self.last_seen = Some(timestamp);
        self
    }
}

/// Fleet roster update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetRosterUpdate {
    /// List of rover statuses
    pub rovers: Vec<RoverStatus>,
    /// Timestamp when update was generated
    pub timestamp: u64,
}

impl FleetRosterUpdate {
    pub fn new(rovers: Vec<RoverStatus>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            rovers,
            timestamp,
        }
    }
}

/// Fleet subscription command for controlling which rovers to subscribe to
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FleetSubscriptionCommand {
    /// Activate subscription to a specific rover
    ActivateRover {
        entity_id: String,
        timestamp: u64,
    },
    /// Deactivate subscription to a specific rover
    DeactivateRover {
        entity_id: String,
        timestamp: u64,
    },
    /// Set the complete list of active rovers (replaces current subscriptions)
    SetActiveRovers {
        entity_ids: Vec<String>,
        timestamp: u64,
    },
}

impl FleetSubscriptionCommand {
    pub fn activate_rover(entity_id: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self::ActivateRover {
            entity_id,
            timestamp,
        }
    }

    pub fn deactivate_rover(entity_id: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self::DeactivateRover {
            entity_id,
            timestamp,
        }
    }

    pub fn set_active_rovers(entity_ids: Vec<String>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self::SetActiveRovers {
            entity_ids,
            timestamp,
        }
    }
}

/// Active rovers status (which rovers are currently subscribed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveRoversStatus {
    /// List of currently active (subscribed) rover entity IDs
    pub active_rovers: Vec<String>,
    /// Timestamp when status was generated
    pub timestamp: u64,
}

impl ActiveRoversStatus {
    pub fn new(active_rovers: Vec<String>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            active_rovers,
            timestamp,
        }
    }
}
