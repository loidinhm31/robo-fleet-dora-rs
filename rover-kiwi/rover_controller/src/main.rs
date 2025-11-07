use dora_node_api::arrow::array::Array;
use dora_node_api::{arrow::array::BinaryArray, dora_core::config::DataId, DoraNode, Event};
use eyre::Result;
use robo_rover_lib::{
    init_tracing, BodyTwist, CommandMetadata, MecanumConfig,
    MecanumKinematics, RoverCommand, RoverTelemetry,
};
use std::collections::HashMap;
use std::error::Error;
use tracing::{debug, info, warn};

fn main() -> Result<(), Box<dyn Error>> {
    let _guard = init_tracing();

    info!("Starting rover controller node...");

    let (mut node, mut events) = DoraNode::init_from_env()?;
    let output_id = DataId::from("processed_rover_command".to_owned());

    // Initialize Mecanum kinematics
    let mecanum_config = MecanumConfig::default();
    info!("Mecanum Configuration:");
    info!("  Wheel radius: {:.3} m", mecanum_config.wheel_radius);
    info!("  Chassis radius: {:.3} m", mecanum_config.chassis_radius);
    info!("  Sliding angles: [{:.1}°, {:.1}°, {:.1}°]",
        mecanum_config.gamma[0].to_degrees(),
        mecanum_config.gamma[1].to_degrees(),
        mecanum_config.gamma[2].to_degrees()
    );

    let mut rover_controller = RoverController::new(mecanum_config);

    info!("Rover controller initialized successfully");
    info!("Ready to process:");
    info!("  - Velocity commands (v_x, v_y, omega_z)");
    info!("  - Joint position commands");
    info!("  - Legacy throttle/steering commands");

    let mut event_count = 0;
    let mut input_stats: HashMap<String, usize> = HashMap::new();

    while let Some(event) = events.recv() {
        event_count += 1;
        debug!("Event #{}: {:?}", event_count, event);

        match event {
            Event::Input {
                id,
                metadata: _,
                data,
            } => {
                let count = input_stats.entry(id.as_str().to_string()).or_insert(0);
                *count += 1;

                match id.as_str() {
                    "rover_command" | "rover_command_voice" | "rover_command_zenoh" => {
                        if let Some(array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if array.len() > 0 {
                                let bytes = array.value(0);

                                match serde_json::from_slice::<RoverCommandWithMetadata>(bytes) {
                                    Ok(cmd_with_metadata) => {
                                        let source = match id.as_str() {
                                            "rover_command_voice" => "voice",
                                            "rover_command_zenoh" => "zenoh",
                                            _ => "manual"
                                        };
                                        info!("Received {} rover command (priority {:?})", source, cmd_with_metadata.metadata.priority);
                                        rover_controller.manual_command = Some(cmd_with_metadata);

                                        // Process arbitrated command
                                        if let Err(e) = rover_controller.process_arbitrated_command(&mut node, &output_id) {
                                            warn!("Failed to process arbitrated command: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse rover command: {}", e);
                                    }
                                }
                            }
                        }
                    }

                    "servo_command" => {
                        if let Some(array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if array.len() > 0 {
                                let bytes = array.value(0);

                                match serde_json::from_slice::<RoverCommandWithMetadata>(bytes) {
                                    Ok(cmd_with_metadata) => {
                                        debug!("Received servo command (priority {:?})", cmd_with_metadata.metadata.priority);
                                        rover_controller.servo_command = Some(cmd_with_metadata);

                                        // Process arbitrated command
                                        if let Err(e) = rover_controller.process_arbitrated_command(&mut node, &output_id) {
                                            warn!("Failed to process arbitrated command: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse servo command: {}", e);
                                    }
                                }
                            }
                        }
                    }

                    "rover_telemetry" => {
                        if let Some(array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if array.len() > 0 {
                                let bytes = array.value(0);
                                match serde_json::from_slice::<RoverTelemetry>(bytes) {
                                    Ok(telemetry) => {
                                        debug!("Received rover telemetry: pos=({:.2}, {:.2}), vel={:.2}",
                                               telemetry.position.0, telemetry.position.1, telemetry.velocity);
                                        rover_controller.update_telemetry(telemetry);
                                    }
                                    Err(e) => {
                                        warn!("Failed to deserialize rover telemetry: {}", e);
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        debug!("Unknown input ID: '{}'", id.as_str());
                    }
                }
            }
            Event::Stop(_) => {
                info!("\nSTOP EVENT RECEIVED");
                info!("Final statistics:");
                info!("   Total events processed: {}", event_count);
                info!("   Input breakdown:");
                for (input_id, count) in &input_stats {
                    info!("      {}: {} events", input_id, count);
                }
                info!("Rover controller stopping gracefully");
                break;
            }

            other_event => {
                debug!("Other event type: {:?}", other_event);
            }
        }
    }

    info!(
        "Rover controller finished after processing {} events",
        event_count
    );
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct RoverCommandWithMetadata {
    command: Option<RoverCommand>,
    metadata: CommandMetadata,
}

struct RoverController {
    current_command: Option<RoverCommand>,
    current_wheel_positions: [f64; 3],  // Current accumulated wheel positions
    last_telemetry: Option<RoverTelemetry>,
    safety_limits: SafetyLimits,
    command_history: Vec<RoverCommand>,
    kinematics: MecanumKinematics,
    control_rate_dt: f64,  // Control loop time step (seconds)

    // Command arbitration
    manual_command: Option<RoverCommandWithMetadata>,
    servo_command: Option<RoverCommandWithMetadata>,
}

#[derive(Debug)]
struct SafetyLimits {
    max_linear_velocity: f64,   // m/s
    max_angular_velocity: f64,  // rad/s
    max_wheel_speed: f64,       // rad/s
}

impl Default for SafetyLimits {
    fn default() -> Self {
        Self {
            max_linear_velocity: 2.0,   // 2 m/s max linear speed
            max_angular_velocity: 2.0,  // 2 rad/s max rotation
            max_wheel_speed: 50.0,      // 50 rad/s max wheel speed
        }
    }
}

impl RoverController {
    fn new(mecanum_config: MecanumConfig) -> Self {
        Self {
            current_command: None,
            current_wheel_positions: [0.0, 0.0, 0.0],
            last_telemetry: None,
            safety_limits: SafetyLimits::default(),
            command_history: Vec::new(),
            kinematics: MecanumKinematics::new(mecanum_config),
            control_rate_dt: 0.05,  // 50ms = 20Hz control rate
            manual_command: None,
            servo_command: None,
        }
    }

    /// Select the highest priority command from available sources
    /// Priority: Emergency (4) > High/Autonomous (3) > Normal/Manual (2) > Low (1)
    fn select_command(&self) -> Option<&RoverCommandWithMetadata> {
        match (&self.manual_command, &self.servo_command) {
            (Some(manual), Some(servo)) => {
                // Compare priorities - higher priority wins
                if servo.metadata.priority >= manual.metadata.priority {
                    info!("Using servo command (priority {:?} >= {:?})",
                          servo.metadata.priority, manual.metadata.priority);
                    Some(servo)
                } else {
                    info!("Manual override - using manual command (priority {:?} > {:?})",
                          manual.metadata.priority, servo.metadata.priority);
                    Some(manual)
                }
            }
            (Some(manual), None) => {
                debug!("Using manual command (no servo command available)");
                Some(manual)
            }
            (None, Some(servo)) => {
                debug!("Using servo command (no manual command available)");
                Some(servo)
            }
            (None, None) => {
                debug!("No commands available");
                None
            }
        }
    }

    /// Process the arbitrated command (highest priority) and send output
    fn process_arbitrated_command(&mut self, node: &mut DoraNode, output_id: &DataId) -> Result<()> {
        if let Some(cmd_with_metadata) = self.select_command() {
            if let Some(command) = &cmd_with_metadata.command {
                debug!("Processing arbitrated command: {:?}", command);

                self.process_command(command.clone())?;

                let processed_cmd = self.get_processed_command();
                let serialized = serde_json::to_vec(&processed_cmd)?;
                let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);

                node.send_output(
                    output_id.clone(),
                    Default::default(),
                    arrow_data
                )?;

                debug!("Sent processed rover command");
            }
        }
        Ok(())
    }

    fn process_command(&mut self, command: RoverCommand) -> Result<()> {
        match &command {
            RoverCommand::Velocity { omega_z, v_x, v_y, .. } => {
                debug!("Processing velocity command: v_x={:.2} m/s, v_y={:.2} m/s, ω_z={:.2} rad/s",
                     v_x, v_y, omega_z);

                // Apply safety limits
                let mut twist = BodyTwist::new(*omega_z, *v_x, *v_y);

                // Limit linear velocity
                let linear_speed = (twist.v_x.powi(2) + twist.v_y.powi(2)).sqrt();
                if linear_speed > self.safety_limits.max_linear_velocity {
                    let scale = self.safety_limits.max_linear_velocity / linear_speed;
                    twist.v_x *= scale;
                    twist.v_y *= scale;
                    info!("SAFETY: Limited linear velocity from {:.2} to {:.2} m/s",
                         linear_speed, self.safety_limits.max_linear_velocity);
                }

                // Limit angular velocity
                if twist.omega_z.abs() > self.safety_limits.max_angular_velocity {
                    twist.omega_z = twist.omega_z.signum() * self.safety_limits.max_angular_velocity;
                    info!("SAFETY: Limited angular velocity to {:.2} rad/s",
                         self.safety_limits.max_angular_velocity);
                }

                // Convert to wheel speeds using Mecanum kinematics
                let wheel_speeds = self.kinematics.body_twist_to_wheel_speeds(&twist);

                debug!("Computed wheel speeds: [{:.2}, {:.2}, {:.2}] rad/s",
                     wheel_speeds[0], wheel_speeds[1], wheel_speeds[2]);

                // Check wheel speed limits
                for (i, &speed) in wheel_speeds.iter().enumerate() {
                    if speed.abs() > self.safety_limits.max_wheel_speed {
                        warn!("SAFETY: Wheel {} speed {:.2} exceeds limit {:.2} rad/s",
                             i, speed.abs(), self.safety_limits.max_wheel_speed);
                    }
                }

                // Integrate wheel speeds to get position changes
                let delta_positions = self.kinematics.wheel_speeds_to_positions(&wheel_speeds, self.control_rate_dt);

                // Update accumulated wheel positions
                for i in 0..3 {
                    self.current_wheel_positions[i] += delta_positions[i];
                }

                debug!("Updated wheel positions: [{:.3}, {:.3}, {:.3}] rad",
                     self.current_wheel_positions[0],
                     self.current_wheel_positions[1],
                     self.current_wheel_positions[2]);

                // Create a joint position command with the updated positions
                let joint_cmd = RoverCommand::new_joint_positions(
                    self.current_wheel_positions[0],
                    self.current_wheel_positions[1],
                    self.current_wheel_positions[2],
                );

                self.current_command = Some(joint_cmd);
            }

            RoverCommand::JointPositions { wheel1, wheel2, wheel3, .. } => {
                info!("Processing direct joint position command: [{:.3}, {:.3}, {:.3}] rad",
                     wheel1, wheel2, wheel3);

                // Update current positions
                self.current_wheel_positions = [*wheel1, *wheel2, *wheel3];
                self.current_command = Some(command);
            }

            RoverCommand::Legacy { throttle, brake, steering_angle, .. } => {
                info!("Processing legacy command: throttle={:.2}, brake={:.2}, steer={:.2}°",
                     throttle, brake, steering_angle);

                // Convert legacy command to velocity command
                // This is a simplified conversion - adjust based on your robot
                let v_x = if *brake > 0.5 { 0.0 } else { throttle * 1.0 }; // Scale to m/s
                let omega_z = steering_angle.to_radians() * 0.5; // Convert to rad/s

                let velocity_cmd = RoverCommand::new_velocity(omega_z, v_x, 0.0);
                return self.process_command(velocity_cmd);
            }

            RoverCommand::Stop { .. } => {
                info!("Processing stop command");
                // Don't change wheel positions, just stop
                self.current_command = Some(command);
            }
        }

        // Store command in history
        if let Some(cmd) = &self.current_command {
            self.command_history.push(cmd.clone());
            if self.command_history.len() > 10 {
                self.command_history.remove(0);
            }
        }

        Ok(())
    }

    fn update_telemetry(&mut self, telemetry: RoverTelemetry) {
        debug!(
            "Updated rover telemetry: pos=({:.2}, {:.2}), vel={:.2}",
            telemetry.position.0, telemetry.position.1, telemetry.velocity
        );

        // Update wheel positions from telemetry if available
        if let Some(wheel_pos) = telemetry.wheel_positions {
            self.current_wheel_positions = wheel_pos;
        }

        self.last_telemetry = Some(telemetry);
    }

    fn get_processed_command(&self) -> RoverCommandWithMetadata {
        let command = self.current_command.clone()
            .unwrap_or_else(|| RoverCommand::new_stop());

        RoverCommandWithMetadata {
            command: Some(command),
            metadata: CommandMetadata {
                command_id: uuid::Uuid::new_v4().to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                source: robo_rover_lib::InputSource::RoverController,
                priority: robo_rover_lib::CommandPriority::Normal,
            },
        }
    }
}