use eyre::Result;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmConfig {
    pub name: String,
    pub dof: usize,
    pub joint_limits: Vec<JointLimit>,
    pub kinematics: KinematicsConfig,
    pub control: ControlConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimit {
    pub min_angle: f64,
    pub max_angle: f64,
    pub max_velocity: f64,
    pub max_acceleration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinematicsConfig {
    pub link_lengths: Vec<f64>,
    pub dh_parameters: Vec<DHParameter>,
    pub base_offset: [f64; 3], // x, y, z
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DHParameter {
    pub a: f64,      // link length
    pub alpha: f64,  // link twist
    pub d: f64,      // link offset
    pub theta: f64,  // joint angle offset
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlConfig {
    pub max_cartesian_velocity: f64,
    pub max_cartesian_acceleration: f64,
    pub position_tolerance: f64,
    pub orientation_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub unity_websocket_port: u16,
    pub update_rate_hz: f64,
    pub physics_timestep: f64,
}

impl ArmConfig {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: ArmConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.joint_limits.len() != self.dof {
            return Err(eyre::eyre!(
                "Joint limits count ({}) doesn't match DOF ({})",
                self.joint_limits.len(),
                self.dof
            ));
        }

        if self.kinematics.dh_parameters.len() != self.dof {
            return Err(eyre::eyre!(
                "DH parameters count ({}) doesn't match DOF ({})",
                self.kinematics.dh_parameters.len(),
                self.dof
            ));
        }

        Ok(())
    }
}

impl SimulationConfig {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: SimulationConfig = toml::from_str(&content)?;
        Ok(config)
    }
}