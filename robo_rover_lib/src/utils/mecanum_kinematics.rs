// Mecanum Wheel Kinematics for 3-Wheel Configuration

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

/// Body twist representation (ω_z, v_x, v_y) in the body frame
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BodyTwist {
    pub omega_z: f64,  // Angular velocity about z-axis (rad/s)
    pub v_x: f64,      // Linear velocity in x direction (m/s)
    pub v_y: f64,      // Linear velocity in y direction (m/s)
}

impl BodyTwist {
    pub fn new(omega_z: f64, v_x: f64, v_y: f64) -> Self {
        Self { omega_z, v_x, v_y }
    }

    pub fn zero() -> Self {
        Self {
            omega_z: 0.0,
            v_x: 0.0,
            v_y: 0.0,
        }
    }

    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.omega_z, self.v_x, self.v_y)
    }

    pub fn from_vector(v: Vector3<f64>) -> Self {
        Self {
            omega_z: v[0],
            v_x: v[1],
            v_y: v[2],
        }
    }
}

/// Mecanum Wheel Configuration Parameters
#[derive(Debug, Clone)]
pub struct MecanumConfig {
    pub wheel_radius: f64,        // r: radius of the wheels (m)
    pub chassis_radius: f64,       // d: distance from center to wheel (m)
    pub gamma: [f64; 3],          // γ: sliding angles for each wheel (rad)
    pub beta: [f64; 3],           // β: wheel position angles (rad)
}

impl Default for MecanumConfig {
    /// Default configuration for 3 mecanum wheels in triangular arrangement
    /// Based on Modern Robotics Figure 13.5 - three omniwheels layout
    /// We adapt this for mecanum wheels with γ = ±45°
    fn default() -> Self {
        use std::f64::consts::PI;
        
        Self {
            wheel_radius: 0.05,        // 5cm wheel radius
            chassis_radius: 0.15,      // 15cm from center to wheel
            // Mecanum wheels typically use γ = ±45° for sliding angles
            gamma: [
                45.0_f64.to_radians(),   // Wheel 1: +45°
                -45.0_f64.to_radians(),  // Wheel 2: -45°  
                45.0_f64.to_radians(),   // Wheel 3: +45°
            ],
            // Wheels arranged in triangle: 0°, 120°, 240°
            beta: [
                0.0,                     // Wheel 1 at 0°
                2.0 * PI / 3.0,          // Wheel 2 at 120°
                4.0 * PI / 3.0,          // Wheel 3 at 240°
            ],
        }
    }
}

/// Mecanum Wheel Kinematics Calculator
/// 
/// Implements the kinematic model from Modern Robotics Eq. (13.6):
/// 
/// u_i = h_i(0) * V_b
/// 
/// where:
/// - u_i is the driving angular velocity of wheel i
/// - V_b = [ω_bz, v_bx, v_by]^T is the body twist
/// - h_i(0) is the kinematic relationship for wheel i
pub struct MecanumKinematics {
    config: MecanumConfig,
    h_matrix: Matrix3<f64>,  // H(0) matrix mapping V_b to wheel velocities
}

impl MecanumKinematics {
    pub fn new(config: MecanumConfig) -> Self {
        let h_matrix = Self::compute_h_matrix(&config);
        Self { config, h_matrix }
    }

    /// Compute the H(0) matrix from Modern Robotics Eq. (13.6)
    /// 
    /// For wheel i: h_i(0) = (1 / (r_i * cos(γ_i))) * 
    ///     [x_i*sin(β_i + γ_i) - y_i*cos(β_i + γ_i), cos(β_i + γ_i), sin(β_i + γ_i)]^T
    /// 
    /// For wheels arranged in a circle: x_i = d*cos(β_i), y_i = d*sin(β_i)
    fn compute_h_matrix(config: &MecanumConfig) -> Matrix3<f64> {
        let r = config.wheel_radius;
        let d = config.chassis_radius;
        
        let mut h = Matrix3::zeros();
        
        for i in 0..3 {
            let beta_i = config.beta[i];
            let gamma_i = config.gamma[i];
            
            // Wheel position in body frame
            let x_i = d * beta_i.cos();
            let y_i = d * beta_i.sin();
            
            // Compute h_i(0) components from Eq. (13.6)
            let denom = r * gamma_i.cos();
            
            // First component: contribution from ω_bz
            h[(i, 0)] = (x_i * (beta_i + gamma_i).sin() - y_i * (beta_i + gamma_i).cos()) / denom;
            
            // Second component: contribution from v_bx
            h[(i, 1)] = (beta_i + gamma_i).cos() / denom;
            
            // Third component: contribution from v_by
            h[(i, 2)] = (beta_i + gamma_i).sin() / denom;
        }
        
        h
    }

    /// Convert body twist to wheel angular velocities
    /// 
    /// Given desired chassis velocity V_b, compute required wheel speeds u
    /// u = H(0) * V_b
    pub fn body_twist_to_wheel_speeds(&self, twist: &BodyTwist) -> Vector3<f64> {
        self.h_matrix * twist.to_vector()
    }

    /// Convert wheel angular velocities to body twist (inverse kinematics)
    /// 
    /// Given wheel speeds u, compute the body twist V_b
    /// V_b = H(0)^† * u  (using pseudoinverse)
    pub fn wheel_speeds_to_body_twist(&self, wheel_speeds: &Vector3<f64>) -> BodyTwist {
        // Use pseudoinverse for better numerical stability
        let h_pinv = self.h_matrix.clone().pseudo_inverse(1e-10).unwrap();
        let twist_vec = h_pinv * wheel_speeds;
        BodyTwist::from_vector(twist_vec)
    }

    /// Convert wheel speeds (rad/s) to joint positions for a time step
    /// This integrates the wheel velocities over dt to get position changes
    pub fn wheel_speeds_to_positions(&self, wheel_speeds: &Vector3<f64>, dt: f64) -> [f64; 3] {
        [
            wheel_speeds[0] * dt,
            wheel_speeds[1] * dt,
            wheel_speeds[2] * dt,
        ]
    }

    /// Get the H matrix
    pub fn h_matrix(&self) -> &Matrix3<f64> {
        &self.h_matrix
    }

    /// Get configuration
    pub fn config(&self) -> &MecanumConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_motion() {
        let kinematics = MecanumKinematics::new(MecanumConfig::default());
        let twist = BodyTwist::new(0.0, 1.0, 0.0); // Move forward at 1 m/s
        let wheel_speeds = kinematics.body_twist_to_wheel_speeds(&twist);
        
        println!("Forward motion wheel speeds: {:?}", wheel_speeds);
        // All wheels should drive forward for forward motion
    }

    #[test]
    fn test_lateral_motion() {
        let kinematics = MecanumKinematics::new(MecanumConfig::default());
        let twist = BodyTwist::new(0.0, 0.0, 1.0); // Move sideways at 1 m/s
        let wheel_speeds = kinematics.body_twist_to_wheel_speeds(&twist);
        
        println!("Lateral motion wheel speeds: {:?}", wheel_speeds);
    }

    #[test]
    fn test_rotation() {
        let kinematics = MecanumKinematics::new(MecanumConfig::default());
        let twist = BodyTwist::new(1.0, 0.0, 0.0); // Rotate at 1 rad/s
        let wheel_speeds = kinematics.body_twist_to_wheel_speeds(&twist);
        
        println!("Rotation wheel speeds: {:?}", wheel_speeds);
    }

    #[test]
    fn test_inverse_kinematics() {
        let kinematics = MecanumKinematics::new(MecanumConfig::default());
        let original_twist = BodyTwist::new(0.5, 1.0, 0.3);
        
        // Forward
        let wheel_speeds = kinematics.body_twist_to_wheel_speeds(&original_twist);
        
        // Inverse
        let recovered_twist = kinematics.wheel_speeds_to_body_twist(&wheel_speeds);
        
        println!("Original: {:?}", original_twist);
        println!("Recovered: {:?}", recovered_twist);
        
        assert!((original_twist.omega_z - recovered_twist.omega_z).abs() < 1e-6);
        assert!((original_twist.v_x - recovered_twist.v_x).abs() < 1e-6);
        assert!((original_twist.v_y - recovered_twist.v_y).abs() < 1e-6);
    }
}
