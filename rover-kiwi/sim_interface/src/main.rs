use dora_node_api::{
    arrow::array::{Array, AsArray, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use robo_rover_lib::{init_tracing, ArmCommand, ArmCommandWithMetadata, ArmTelemetry, CompleteJointState, RoverCommand, RoverCommandWithMetadata, RoverTelemetry};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio;

// Joint positions structure for LeKiwi arm (6 DOF)
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ArmJointPositions {
    pub shoulder_pan: f64,
    pub shoulder_lift: f64,
    pub elbow_flex: f64,
    pub wrist_flex: f64,
    pub wrist_roll: f64,
    pub gripper: f64,
}

impl ArmJointPositions {
    pub fn home() -> Self {
        Self {
            shoulder_pan: 0.0,
            shoulder_lift: 0.0,
            elbow_flex: 0.0,
            wrist_flex: 0.0,
            wrist_roll: 0.0,
            gripper: 0.0,
        }
    }

    pub fn to_array(&self) -> [f64; 6] {
        [
            self.shoulder_pan,
            self.shoulder_lift,
            self.elbow_flex,
            self.wrist_flex,
            self.wrist_roll,
            self.gripper,
        ]
    }

    pub fn from_array(arr: &[f64]) -> Self {
        assert!(arr.len() >= 6, "Array must have at least 6 elements");
        Self {
            shoulder_pan: arr[0],
            shoulder_lift: arr[1],
            elbow_flex: arr[2],
            wrist_flex: arr[3],
            wrist_roll: arr[4],
            gripper: arr[5],
        }
    }
}

// urdf-viz response structure
#[derive(Deserialize, Debug)]
struct UrdfVizResponse {
    pub is_ok: bool,
    pub reason: String,
}

/// Unified Robot Controller for urdf-viz simulation
/// Handles both arm (6 joints) and rover (3 mecanum wheels)
pub struct UnifiedRobotController {
    url: String,
    client: reqwest::Client,
    current_arm_position: ArmJointPositions,
    current_rover_position: [f64; 3],  // 3 mecanum wheel positions
}

impl UnifiedRobotController {
    /// Create a new controller with default urdf-viz URL
    pub fn new() -> Self {
        Self {
            url: "http://127.0.0.1:7777".to_string(),
            client: reqwest::Client::new(),
            current_arm_position: ArmJointPositions::home(),
            current_rover_position: [0.0, 0.0, 0.0],
        }
    }

    /// Create with custom URL
    pub fn with_url(url: String) -> Self {
        Self {
            url,
            client: reqwest::Client::new(),
            current_arm_position: ArmJointPositions::home(),
            current_rover_position: [0.0, 0.0, 0.0],
        }
    }

    /// Send complete joint positions (3 rover wheels + 6 arm joints) to urdf-viz
    pub async fn send_complete_state(&self, state: &CompleteJointState) -> Result<()> {
        tracing::debug!("Sending complete joint state to urdf-viz:");
        tracing::debug!("Rover wheels:");
        for i in 0..3 {
            tracing::debug!("{}: {:.3} rad ({:.1}°)",
                     state.names[i], state.positions[i], state.positions[i].to_degrees());
        }
        tracing::debug!("Arm joints:");
        for i in 3..9 {
            tracing::debug!("{}: {:.3} rad ({:.1}°)",
                     state.names[i], state.positions[i], state.positions[i].to_degrees());
        }

        let endpoint = format!("{}/set_joint_positions", self.url);

        let response = self.client
            .post(&endpoint)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&state)
            .send()
            .await?;

        if response.status().is_success() {
            let result: UrdfVizResponse = response.json().await?;
            if result.is_ok {
                tracing::debug!("Joint positions sent successfully to urdf-viz");
            } else {
                eyre::bail!("urdf-viz error: {}", result.reason);
            }
        } else {
            eyre::bail!("HTTP error: {}", response.status());
        }

        Ok(())
    }

    /// Update arm position and send to urdf-viz
    pub async fn update_arm(&mut self, positions: ArmJointPositions) -> Result<()> {
        self.current_arm_position = positions;
        let mut state = CompleteJointState::new();
        state.set_rover_positions(
            self.current_rover_position[0],
            self.current_rover_position[1],
            self.current_rover_position[2],
        );
        state.set_arm_positions(&positions.to_array());
        self.send_complete_state(&state).await
    }

    /// Update rover wheel positions and send to urdf-viz
    pub async fn update_rover(&mut self, wheel1: f64, wheel2: f64, wheel3: f64) -> Result<()> {
        self.current_rover_position = [wheel1, wheel2, wheel3];
        let mut state = CompleteJointState::new();
        state.set_rover_positions(wheel1, wheel2, wheel3);
        state.set_arm_positions(&self.current_arm_position.to_array());
        self.send_complete_state(&state).await
    }

    /// Move to home position (both arm and rover)
    pub async fn home(&mut self) -> Result<()> {
        self.current_arm_position = ArmJointPositions::home();
        self.current_rover_position = [0.0, 0.0, 0.0];
        let state = CompleteJointState::new();  // Default is home
        self.send_complete_state(&state).await
    }

    /// Generate telemetry from current state
    pub fn generate_arm_telemetry(&self) -> ArmTelemetry {
        // Simplified forward kinematics for end-effector position
        let l1 = 0.3; // Link lengths (should match LeKiwi URDF)
        let l2 = 0.25;

        let theta1 = self.current_arm_position.shoulder_pan;
        let theta2 = self.current_arm_position.shoulder_lift;
        let theta3 = self.current_arm_position.elbow_flex;

        // Simplified planar forward kinematics (2D projection)
        let x = l1 * theta2.cos() + l2 * (theta2 + theta3).cos();
        let y = theta1.sin() * (l1 * theta2.cos() + l2 * (theta2 + theta3).cos());
        let z = l1 * theta2.sin() + l2 * (theta2 + theta3).sin() + 0.1;

        ArmTelemetry {
            entity_id: None,
            end_effector_pose: [
                x,
                y,
                z,
                self.current_arm_position.wrist_roll,
                self.current_arm_position.wrist_flex,
                theta1,
            ],
            is_moving: false,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            joint_angles: Some(self.current_arm_position.to_array().to_vec()),
            joint_velocities: Some(vec![0.0; 6]),
            source: "urdf_viz_simulation".to_string(),
        }
    }

    /// Generate rover telemetry from current state
    pub fn generate_rover_telemetry(&self) -> RoverTelemetry {
        let mut telemetry = RoverTelemetry::new();
        telemetry.wheel_positions = Some(self.current_rover_position);
        telemetry.wheel_velocities = Some([0.0, 0.0, 0.0]);  // Simplified - not tracking velocities
        telemetry
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing();

    tracing::info!("Starting Unified Simulation Interface for LeKiwi Robot");
    tracing::info!("3 Mecanum wheels for omnidirectional movement");
    tracing::info!("6 DOF robotic arm");
    tracing::info!("Connecting to urdf-viz at http://127.0.0.1:7777");

    let (mut node, mut events) = DoraNode::init_from_env()?;
    let arm_telemetry_output = DataId::from("arm_telemetry".to_owned());
    let rover_telemetry_output = DataId::from("rover_telemetry".to_owned());

    // Get urdf-viz URL from environment or use default
    let urdf_viz_url = std::env::var("URDFVIZ_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:7777".to_string());

    let mut controller = UnifiedRobotController::with_url(urdf_viz_url);

    tracing::info!("Simulation interface initialized");
    tracing::info!("Sending robot to home position");

    // Initialize to home position
    if let Err(e) = controller.home().await {
        tracing::warn!("Failed to send home position to urdf-viz: {}", e);
        tracing::info!("Make sure urdf-viz is running with: urdf-viz model/LeKiwi.urdf");
    } else {
        tracing::info!("Robot initialized at home position");
    }

    tracing::info!("Ready to receive commands");

    // Event loop
    loop {
        if let Some(event) = events.recv() {
            match event {
                Event::Input { id, data, .. } => {
                    let id_str = id.as_str();

                    match id_str {
                        "arm_command" => {
                            tracing::debug!("Received arm command");

                            if let Some(binary_array) = data.as_binary_opt::<i32>() {
                                if binary_array.len() > 0 {
                                    let bytes = binary_array.value(0);

                                    match serde_json::from_slice::<ArmCommandWithMetadata>(bytes) {
                                        Ok(cmd_with_metadata) => {
                                            if let Some(arm_cmd) = cmd_with_metadata.command {
                                                tracing::debug!("Processing arm command: {:?}", arm_cmd);

                                                match arm_cmd {
                                                    ArmCommand::JointPosition { joint_angles, .. } => {
                                                        if joint_angles.len() >= 6 {
                                                            let positions = ArmJointPositions::from_array(&joint_angles);

                                                            match controller.update_arm(positions).await {
                                                                Ok(_) => {
                                                                    tracing::debug!("Arm positions sent successfully");

                                                                    // Send telemetry
                                                                    let telemetry = controller.generate_arm_telemetry();
                                                                    if let Ok(serialized) = serde_json::to_vec(&telemetry) {
                                                                        let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                                                                        let _ = node.send_output(
                                                                            arm_telemetry_output.clone(),
                                                                            Default::default(),
                                                                            arrow_data,
                                                                        );
                                                                    }
                                                                }
                                                                Err(e) => {
                                                                    tracing::error!("Failed to send arm positions: {}", e);
                                                                }
                                                            }
                                                        } else {
                                                            tracing::error!("Invalid joint angles length: {}", joint_angles.len());
                                                        }
                                                    }
                                                    ArmCommand::Home => {
                                                        if let Err(e) = controller.home().await {
                                                            tracing::error!("Failed to move to home: {}", e);
                                                        }
                                                    }
                                                    ArmCommand::Stop | ArmCommand::EmergencyStop => {
                                                        tracing::info!("Stop command received for arm");
                                                    }
                                                    _ => {
                                                        tracing::warn!("Unsupported arm command type for simulation");
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            tracing::error!("Failed to parse arm command: {}", e);
                                        }
                                    }
                                }
                            }
                        }

                        "rover_command" => {
                            tracing::debug!("Received rover command");

                            if let Some(binary_array) = data.as_binary_opt::<i32>() {
                                if binary_array.len() > 0 {
                                    let bytes = binary_array.value(0);

                                    match serde_json::from_slice::<RoverCommandWithMetadata>(bytes) {
                                        Ok(cmd_with_metadata) => {
                                            let rover_cmd = cmd_with_metadata.command;
                                            tracing::debug!("Processing rover command: {:?}", rover_cmd);

                                            match rover_cmd {
                                                RoverCommand::JointPositions { wheel1, wheel2, wheel3, .. } => {
                                                    match controller.update_rover(wheel1, wheel2, wheel3).await {
                                                        Ok(_) => {
                                                            tracing::debug!("Rover wheel positions sent: [{:.3}, {:.3}, {:.3}]",
                                                                     wheel1, wheel2, wheel3);

                                                            // Send telemetry
                                                            let telemetry = controller.generate_rover_telemetry();
                                                            if let Ok(serialized) = serde_json::to_vec(&telemetry) {
                                                                let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                                                                let _ = node.send_output(
                                                                    rover_telemetry_output.clone(),
                                                                    Default::default(),
                                                                    arrow_data,
                                                                );
                                                            }
                                                        }
                                                        Err(e) => {
                                                            tracing::error!("Failed to send rover positions: {}", e);
                                                        }
                                                    }
                                                }
                                                RoverCommand::Stop { .. } => {
                                                    tracing::info!("Stop command received for rover");
                                                    // Keep current position, don't update
                                                    }
                                                    _ => {
                                                        tracing::debug!("Rover command type will be converted to joint positions by rover-controller");
                                                    }
                                                }
                                        }
                                        Err(e) => {
                                            tracing::error!("Failed to parse rover command: {}", e);
                                        }
                                    }
                                }
                            }
                        }

                        _ => {
                            tracing::warn!("Unknown input: {}", id_str);
                        }
                    }
                }

                Event::Stop(_) => {
                    tracing::info!("Stop event received");
                    break;
                }

                _ => {}
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    tracing::info!("Simulation interface shutting down");
    Ok(())
}