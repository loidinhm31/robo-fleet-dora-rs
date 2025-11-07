use crate::CommandMetadata;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid;

/// Rover Command - supports both legacy throttle/steering and modern velocity control
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoverCommand {
    /// Legacy command for simple forward/backward/turn control
    Legacy {
        throttle: f64,
        brake: f64,
        steering_angle: f64,
        timestamp: u64,
        command_id: String,
    },

    /// Modern velocity command for omnidirectional Mecanum wheel control
    /// Based on Modern Robotics body twist: (ω_z, v_x, v_y)
    Velocity {
        omega_z: f64,  // Angular velocity about z-axis (rad/s)
        v_x: f64,      // Linear velocity in x direction (m/s)
        v_y: f64,      // Linear velocity in y direction (m/s)
        timestamp: u64,
        command_id: String,
    },

    /// Direct joint position command for 3 mecanum wheels
    /// Positions are in radians
    JointPositions {
        wheel1: f64,  // ST3215_Servo_Motor-v1-2_Revolute-60
        wheel2: f64,  // ST3215_Servo_Motor-v1-1_Revolute-62
        wheel3: f64,  // ST3215_Servo_Motor-v1_Revolute-64
        timestamp: u64,
        command_id: String,
    },

    /// Stop command
    Stop {
        timestamp: u64,
        command_id: String,
    },
}

impl RoverCommand {
    /// Create a new legacy command
    pub fn new_legacy(throttle: f64, brake: f64, steering_angle: f64) -> Self {
        Self::Legacy {
            throttle: throttle.clamp(-1.0, 1.0),
            brake: brake.clamp(0.0, 1.0),
            steering_angle: steering_angle.clamp(-15.0, 15.0),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            command_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Create a new velocity command for omnidirectional control
    pub fn new_velocity(omega_z: f64, v_x: f64, v_y: f64) -> Self {
        Self::Velocity {
            omega_z,
            v_x,
            v_y,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            command_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Create a new joint position command
    pub fn new_joint_positions(wheel1: f64, wheel2: f64, wheel3: f64) -> Self {
        Self::JointPositions {
            wheel1,
            wheel2,
            wheel3,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            command_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Create a stop command
    pub fn new_stop() -> Self {
        Self::Stop {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            command_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

/// Rover Telemetry - feedback from simulation/hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoverTelemetry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,

    // Position and orientation
    pub position: (f64, f64),  // (x, y) in meters
    pub yaw: f64,              // Rotation about z-axis (rad)
    pub pitch: f64,            // Rotation about y-axis (rad)
    pub roll: f64,             // Rotation about x-axis (rad)

    // Velocity
    pub velocity: f64,         // Linear velocity magnitude (m/s)
    pub velocity_x: Option<f64>,   // X-component of velocity
    pub velocity_y: Option<f64>,   // Y-component of velocity
    pub angular_velocity: Option<f64>,  // Angular velocity about z

    // Wheel states (for 3 mecanum wheels)
    pub wheel_positions: Option<[f64; 3]>,  // Current wheel angles (rad)
    pub wheel_velocities: Option<[f64; 3]>, // Current wheel speeds (rad/s)

    // Navigation sensors (if available)
    pub nav_angles: Option<Vec<f64>>,
    pub nav_dists: Option<Vec<f64>>,

    pub timestamp: u64,
}

impl RoverTelemetry {
    pub fn new() -> Self {
        Self {
            entity_id: None,
            position: (0.0, 0.0),
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            velocity: 0.0,
            velocity_x: None,
            velocity_y: None,
            angular_velocity: None,
            wheel_positions: None,
            wheel_velocities: None,
            nav_angles: None,
            nav_dists: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RoverCommandWithMetadata {
   pub command: RoverCommand,
   pub metadata: CommandMetadata,
}