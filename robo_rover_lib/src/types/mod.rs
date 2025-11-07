pub mod arm_types;
pub mod config;
pub mod rover_types;
pub mod arm_telemetry;
pub mod video_types;
pub mod detection_types;
pub mod speech_types;
pub mod nlu_types;
pub mod tts_types;
pub mod performance_types;
pub mod fleet_types;

use serde::{Deserialize, Serialize};
pub use arm_types::*;
pub use config::*;
pub use rover_types::*;
pub use arm_telemetry::*;
pub use video_types::*;
pub use detection_types::*;
pub use speech_types::*;
pub use nlu_types::*;
pub use tts_types::*;
pub use performance_types::*;
pub use fleet_types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandMetadata {
    pub command_id: String,
    pub timestamp: u64,
    pub source: InputSource,
    pub priority: CommandPriority,
}

/// Complete URDF joint state (3 mecanum wheels + 6 arm joints)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteJointState {
    pub names: Vec<String>,
    pub positions: Vec<f64>,
}

impl CompleteJointState {
    /// Create a new complete joint state
    pub fn new() -> Self {
        Self {
            names: vec![
                // Mecanum wheels (3)
                "ST3215_Servo_Motor-v1-2_Revolute-60".to_string(),
                "ST3215_Servo_Motor-v1-1_Revolute-62".to_string(),
                "ST3215_Servo_Motor-v1_Revolute-64".to_string(),
                // Arm joints (6)
                "STS3215_03a-v1_Revolute-45".to_string(),
                "STS3215_03a-v1-1_Revolute-49".to_string(),
                "STS3215_03a-v1-2_Revolute-51".to_string(),
                "STS3215_03a-v1-3_Revolute-53".to_string(),
                "STS3215_03a_Wrist_Roll-v1_Revolute-55".to_string(),
                "STS3215_03a-v1-4_Revolute-57".to_string(),
            ],
            positions: vec![0.0; 9],
        }
    }

    /// Set rover wheel positions (first 3 elements)
    pub fn set_rover_positions(&mut self, wheel1: f64, wheel2: f64, wheel3: f64) {
        if self.positions.len() >= 3 {
            self.positions[0] = wheel1;
            self.positions[1] = wheel2;
            self.positions[2] = wheel3;
        }
    }

    /// Set arm joint positions (elements 3-8)
    pub fn set_arm_positions(&mut self, arm_joints: &[f64]) {
        for (i, &pos) in arm_joints.iter().enumerate() {
            if i + 3 < self.positions.len() {
                self.positions[i + 3] = pos;
            }
        }
    }

    /// Get rover wheel positions
    pub fn get_rover_positions(&self) -> [f64; 3] {
        [
            self.positions.get(0).copied().unwrap_or(0.0),
            self.positions.get(1).copied().unwrap_or(0.0),
            self.positions.get(2).copied().unwrap_or(0.0),
        ]
    }

    /// Get arm joint positions
    pub fn get_arm_positions(&self) -> Vec<f64> {
        self.positions.iter().skip(3).copied().collect()
    }
}