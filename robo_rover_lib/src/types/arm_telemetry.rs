use serde::{Deserialize, Serialize};

/// Arm telemetry data received from Unity simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmTelemetry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    /// End effector position and orientation [x, y, z, roll, pitch, yaw]
    pub end_effector_pose: [f64; 6],

    /// Whether the arm is currently moving
    pub is_moving: bool,

    /// Timestamp in milliseconds since Unix epoch
    pub timestamp: u64,

    /// Optional joint angles if available from Unity
    pub joint_angles: Option<Vec<f64>>,

    /// Optional joint velocities if available from Unity
    pub joint_velocities: Option<Vec<f64>>,

    /// Source identifier (typically "unity_simulation")
    pub source: String,
}

impl ArmTelemetry {
    pub fn new() -> Self {
        Self {
            entity_id: None,
            end_effector_pose: [0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
            is_moving: false,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            joint_angles: Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            joint_velocities: Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            source: "mock_simulation".to_string(),
        }
    }
}