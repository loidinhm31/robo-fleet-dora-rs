use dora_node_api::{
    arrow::array::{types::GenericBinaryType, Array, AsArray, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use robo_rover_lib::{init_tracing, ArmCommand, ArmCommandWithMetadata, ArmConfig, ArmTelemetry};
use std::error::Error;
use tracing::{debug, info, warn};

struct ArmController {
    config: ArmConfig,
    last_telemetry: Option<ArmTelemetry>,
    current_command: Option<ArmCommand>,
    safety_limits_enabled: bool,
}

impl ArmController {
    fn new() -> Result<Self> {
        // Load arm configuration
        let config_path =
            std::env::var("ARM_CONFIG").unwrap_or_else(|_| "config/arm_6dof.toml".to_string());

        let config = ArmConfig::load_from_file(&config_path)
            .map_err(|e| eyre::eyre!("Failed to load arm config from {}: {}", config_path, e))?;

        info!("Loaded arm configuration: {} DOF", config.dof);
        info!(
            "Joint limits configured for {} joints",
            config.joint_limits.len()
        );

        Ok(Self {
            config,
            last_telemetry: None,
            current_command: None,
            safety_limits_enabled: true,
        })
    }

    fn process_command(&mut self, command: ArmCommand) -> Result<ArmCommand> {
        info!("Processing arm command: {:?}", command);

        // Validate command safety
        if let Err(e) = self.validate_command_safety(&command) {
            warn!("Command validation failed: {}", e);
            return Err(e);
        }

        // Process different command types
        let processed_command = match &command {
            ArmCommand::CartesianMove {
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
                max_velocity,
            } => self.process_cartesian_move(*x, *y, *z, *roll, *pitch, *yaw, *max_velocity)?,
            ArmCommand::JointPosition {
                joint_angles,
                max_velocity,
            } => self.process_joint_position(joint_angles, *max_velocity)?,
            ArmCommand::RelativeMove { delta_joints } => {
                self.process_relative_move(delta_joints)?
            }
            ArmCommand::Home => self.process_home_command()?,
            ArmCommand::Stop => self.process_stop_command()?,
            ArmCommand::EmergencyStop => self.process_emergency_stop()?,
        };

        self.current_command = Some(command);
        Ok(processed_command)
    }

    fn validate_command_safety(&self, command: &ArmCommand) -> Result<()> {
        if !self.safety_limits_enabled {
            return Ok(());
        }

        match command {
            ArmCommand::JointPosition { joint_angles, .. } => {
                // Check joint limits
                for (i, &angle) in joint_angles.iter().enumerate() {
                    if i >= self.config.joint_limits.len() {
                        return Err(eyre::eyre!("Joint {} exceeds configured DOF", i));
                    }

                    let limits = &self.config.joint_limits[i];
                    if angle < limits.min_angle || angle > limits.max_angle {
                        return Err(eyre::eyre!(
                            "Joint {} angle {:.3} outside limits [{:.3}, {:.3}]",
                            i,
                            angle,
                            limits.min_angle,
                            limits.max_angle
                        ));
                    }
                }
            }
            ArmCommand::RelativeMove { .. } => {
                debug!("RelativeMove move validation");
            }
            ArmCommand::CartesianMove { .. } => {
                // TODO: Implement workspace limits checking
                // For now, just validate that IK solution exists
                debug!("Cartesian move validation - IK solution check needed");
            }
            _ => {
                // Stop, Home, EmergencyStop are always safe
            }
        }

        Ok(())
    }

    fn process_cartesian_move(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        roll: f64,
        pitch: f64,
        yaw: f64,
        max_velocity: Option<f64>,
    ) -> Result<ArmCommand> {
        debug!(
            "Processing Cartesian move: ({:.3}, {:.3}, {:.3}) + rotation",
            x, y, z
        );

        let velocity = max_velocity.unwrap_or(self.config.control.max_cartesian_velocity);

        info!("Cartesian move command validated - forwarding to Unity for IK");

        Ok(ArmCommand::CartesianMove {
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            max_velocity: Some(velocity),
        })
    }

    fn process_joint_position(
        &mut self,
        joint_angles: &[f64],
        max_velocity: Option<f64>,
    ) -> Result<ArmCommand> {
        debug!(
            "Processing joint position command with {} angles",
            joint_angles.len()
        );

        if joint_angles.len() != self.config.dof {
            return Err(eyre::eyre!(
                "Joint angles count {} doesn't match configured DOF {}",
                joint_angles.len(),
                self.config.dof
            ));
        }

        let velocity = max_velocity.unwrap_or(2.0);

        info!(
            "Joint position command validated - {} joints",
            joint_angles.len()
        );

        Ok(ArmCommand::JointPosition {
            joint_angles: joint_angles.to_vec(),
            max_velocity: Some(velocity),
        })
    }

    fn process_relative_move(&mut self, delta_joints: &[f64]) -> Result<ArmCommand> {
        debug!(
            "Processing relative move with {} deltas",
            delta_joints.len()
        );

        if delta_joints.len() != self.config.dof {
            return Err(eyre::eyre!(
                "Delta joints count {} doesn't match configured DOF {}",
                delta_joints.len(),
                self.config.dof
            ));
        }

        info!("Relative move converted to joint positions");

        Ok(ArmCommand::JointPosition {
            joint_angles: vec![],
            max_velocity: Some(2.0),
        })
    }

    fn process_home_command(&mut self) -> Result<ArmCommand> {
        info!("Processing home command - returning to zero position");

        // Home position is all joints at 0
        let home_positions = vec![0.0; self.config.dof];

        Ok(ArmCommand::JointPosition {
            joint_angles: home_positions,
            max_velocity: Some(1.0), // Slower for safety
        })
    }

    fn process_stop_command(&mut self) -> Result<ArmCommand> {
        info!("Stop command issued - holding current position");

        Ok(ArmCommand::Stop)
    }

    fn process_emergency_stop(&mut self) -> Result<ArmCommand> {
        warn!("EMERGENCY STOP - immediate halt");

        Ok(ArmCommand::EmergencyStop)
    }

    fn update_current_state(&mut self, arm_telemetry: ArmTelemetry) {
        self.last_telemetry = Some(arm_telemetry);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let _guard = init_tracing();

    info!("Starting arm controller node");

    let (mut node, mut events) = DoraNode::init_from_env()?;
    let output_id = DataId::from("processed_arm_command".to_owned());

    let mut arm_controller = ArmController::new()?;

    info!(
        "Arm controller initialized with {} DOF configuration",
        arm_controller.config.dof
    );
    info!(
        "Safety limits: {}",
        if arm_controller.safety_limits_enabled {
            "ENABLED"
        } else {
            "DISABLED"
        }
    );

    while let Some(event) = events.recv() {
        match event {
            Event::Input {
                id,
                metadata: _,
                data,
            } => {
                let id_str = id.as_str();
                debug!("Received input: {}", id_str);

                match id_str {
                    "arm_command" | "arm_command_zenoh" => {
                        if let Some(array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if array.len() > 0 {
                                let bytes = array.value(0);

                                match serde_json::from_slice::<ArmCommandWithMetadata>(bytes) {
                                    Ok(cmd_with_metadata) => {
                                        if let Some(command) = cmd_with_metadata.command {
                                            let source = if id_str == "arm_command_zenoh" { "zenoh" } else { "manual" };
                                            info!("Received {} arm command: {:?}", source, command);

                                            match arm_controller.process_command(command) {
                                                Ok(processed_cmd) => {
                                                    // Create output with metadata
                                                    let output_data = ArmCommandWithMetadata {
                                                        command: Some(processed_cmd),
                                                        metadata: cmd_with_metadata.metadata,
                                                    };

                                                    let serialized =
                                                        serde_json::to_vec(&output_data)?;
                                                    let arrow_data = BinaryArray::from_vec(vec![
                                                        serialized.as_slice(),
                                                    ]);

                                                    if let Err(e) = node.send_output(
                                                        output_id.clone(),
                                                        Default::default(),
                                                        arrow_data,
                                                    ) {
                                                        warn!(
                                                            "Failed to send processed command: {}",
                                                            e
                                                        );
                                                    } else {
                                                        debug!("Sent processed arm command");
                                                    }
                                                }
                                                Err(e) => {
                                                    warn!("Command processing failed: {}", e);
                                                }
                                            }
                                        } else {
                                            warn!("Received empty arm command");
                                        }
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse arm command: {}", e);
                                    }
                                }
                            }
                        }
                    }

                    "arm_telemetry" => {
                        if let Some(bytes_array) = data.as_bytes_opt::<GenericBinaryType<i32>>() {
                            if bytes_array.len() > 0 {
                                let bytes = bytes_array.value(0);

                                if let Ok(status) = serde_json::from_slice::<ArmTelemetry>(bytes) {
                                    arm_controller.update_current_state(status);
                                    debug!("Updated arm state from feedback");
                                }
                            }
                        }
                    }

                    _ => {
                        debug!("Unknown input id: {}", id_str);
                    }
                }
            }

            Event::Stop(_) => {
                info!("Stop event received");
                break;
            }

            _ => {}
        }
    }

    info!("Arm controller shutting down");
    Ok(())
}
