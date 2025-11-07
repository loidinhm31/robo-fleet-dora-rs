use dora_node_api::{self, arrow::array::BinaryArray, dora_core::config::DataId, DoraNode, Event};
use dora_node_api::arrow::array::Array;
use eyre::Result;
use robo_rover_lib::{
    init_tracing, BoundingBox, ControlMode, ControlOutput, RoverCommand, RoverCommandWithMetadata,
    TrackingTelemetry, TrackingState, CommandMetadata, InputSource, CommandPriority,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

mod pid;
use pid::PIDController;

/// Camera and mounting configuration
struct CameraConfig {
    focal_length_pixels: f32,  // Focal length in pixels (calibrated)
    image_width: u32,
    image_height: u32,
    camera_height: f32,  // Camera mounting height above ground (meters)
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            focal_length_pixels: 500.0,  // Typical for 640x480 webcam (needs calibration)
            image_width: 640,
            image_height: 480,
            camera_height: 0.5,  // 50cm above ground
        }
    }
}

/// Visual servoing controller configuration
struct ServoConfig {
    // PID parameters for lateral control (centering)
    lateral_kp: f64,
    lateral_ki: f64,
    lateral_kd: f64,

    // PID parameters for longitudinal control (distance)
    longitudinal_kp: f64,
    longitudinal_ki: f64,
    longitudinal_kd: f64,

    // Safety and physical constraints
    min_distance: f32,  // Minimum safe distance (meters)
    max_velocity: f64,  // Maximum linear velocity (m/s)
    max_angular_velocity: f64,  // Maximum angular velocity (rad/s)

    // Control parameters
    target_bbox_height: f32,  // Target size for distance control (normalized)
    dead_zone: f32,  // Dead zone for centering (normalized coordinates)

    // Typical object heights for distance estimation (meters)
    person_height: f32,
    dog_height: f32,
    cat_height: f32,
    default_object_height: f32,
}

impl Default for ServoConfig {
    fn default() -> Self {
        Self {
            // Lateral PID (centering control)
            lateral_kp: 1.5,
            lateral_ki: 0.0,
            lateral_kd: 0.2,

            // Longitudinal PID (distance control)
            longitudinal_kp: 0.8,
            longitudinal_ki: 0.0,
            longitudinal_kd: 0.15,

            // Safety constraints
            min_distance: 1.0,
            max_velocity: 0.5,
            max_angular_velocity: 1.0,

            // Control parameters
            target_bbox_height: 0.3,  // Target 30% of frame height
            dead_zone: 0.05,  // 5% dead zone

            // Object heights for distance estimation
            person_height: 1.7,
            dog_height: 0.5,
            cat_height: 0.3,
            default_object_height: 0.5,
        }
    }
}

impl ServoConfig {
    /// Load configuration from environment variables
    fn from_env() -> Self {
        let mut config = Self::default();

        // Lateral PID
        if let Ok(val) = std::env::var("LATERAL_PID_KP") {
            config.lateral_kp = val.parse().unwrap_or(config.lateral_kp);
        }
        if let Ok(val) = std::env::var("LATERAL_PID_KI") {
            config.lateral_ki = val.parse().unwrap_or(config.lateral_ki);
        }
        if let Ok(val) = std::env::var("LATERAL_PID_KD") {
            config.lateral_kd = val.parse().unwrap_or(config.lateral_kd);
        }

        // Longitudinal PID
        if let Ok(val) = std::env::var("LONGITUDINAL_PID_KP") {
            config.longitudinal_kp = val.parse().unwrap_or(config.longitudinal_kp);
        }
        if let Ok(val) = std::env::var("LONGITUDINAL_PID_KI") {
            config.longitudinal_ki = val.parse().unwrap_or(config.longitudinal_ki);
        }
        if let Ok(val) = std::env::var("LONGITUDINAL_PID_KD") {
            config.longitudinal_kd = val.parse().unwrap_or(config.longitudinal_kd);
        }

        // Safety constraints
        if let Ok(val) = std::env::var("MIN_DISTANCE") {
            config.min_distance = val.parse().unwrap_or(config.min_distance);
        }
        if let Ok(val) = std::env::var("MAX_VELOCITY") {
            config.max_velocity = val.parse().unwrap_or(config.max_velocity);
        }
        if let Ok(val) = std::env::var("MAX_ANGULAR_VELOCITY") {
            config.max_angular_velocity = val.parse().unwrap_or(config.max_angular_velocity);
        }

        // Control parameters
        if let Ok(val) = std::env::var("TARGET_BBOX_HEIGHT") {
            config.target_bbox_height = val.parse().unwrap_or(config.target_bbox_height);
        }
        if let Ok(val) = std::env::var("DEAD_ZONE") {
            config.dead_zone = val.parse().unwrap_or(config.dead_zone);
        }

        config
    }
}

/// Visual servoing controller state
struct ServoController {
    config: ServoConfig,
    camera_config: CameraConfig,
    lateral_pid: PIDController,
    longitudinal_pid: PIDController,
    last_command_time: Option<SystemTime>,
}

impl ServoController {
    fn new() -> Self {
        let config = ServoConfig::from_env();

        info!("Visual Servo Controller Configuration:");
        info!("  Lateral PID: Kp={}, Ki={}, Kd={}", config.lateral_kp, config.lateral_ki, config.lateral_kd);
        info!("  Longitudinal PID: Kp={}, Ki={}, Kd={}", config.longitudinal_kp, config.longitudinal_ki, config.longitudinal_kd);
        info!("  Min Distance: {}m", config.min_distance);
        info!("  Max Velocity: {}m/s", config.max_velocity);
        info!("  Target BBox Height: {}", config.target_bbox_height);

        Self {
            lateral_pid: PIDController::new(
                config.lateral_kp,
                config.lateral_ki,
                config.lateral_kd,
                -config.max_angular_velocity,
                config.max_angular_velocity,
            ),
            longitudinal_pid: PIDController::new(
                config.longitudinal_kp,
                config.longitudinal_ki,
                config.longitudinal_kd,
                -config.max_velocity,
                config.max_velocity,
            ),
            config,
            camera_config: CameraConfig::default(),
            last_command_time: None,
        }
    }

    /// Estimate distance to object based on bounding box height
    /// Uses pinhole camera model: distance = (real_height * focal_length) / image_height
    fn estimate_distance(&self, bbox: &BoundingBox, class_name: &str) -> f32 {
        // Get expected real-world height of object
        let real_height = match class_name {
            "person" => self.config.person_height,
            "dog" => self.config.dog_height,
            "cat" => self.config.cat_height,
            _ => self.config.default_object_height,
        };

        // Calculate image height in pixels
        let bbox_height_pixels = bbox.height() * self.camera_config.image_height as f32;

        // Avoid division by zero
        if bbox_height_pixels < 1.0 {
            return 10.0;  // Return large distance if bbox too small
        }

        // Distance = (real_height * focal_length) / image_height
        let distance = (real_height * self.camera_config.focal_length_pixels) / bbox_height_pixels;

        // Clamp to reasonable range
        distance.clamp(0.5, 10.0)
    }

    /// Process tracking telemetry and generate servo command + enhanced telemetry
    fn process_tracking(&mut self, telemetry: TrackingTelemetry, dt: f64) -> (Option<RoverCommandWithMetadata>, TrackingTelemetry) {
        // Only generate commands when actively tracking a target
        if telemetry.state != TrackingState::Tracking {
            // Reset PIDs when not tracking
            self.lateral_pid.reset();
            self.longitudinal_pid.reset();

            // Return original telemetry with Manual mode
            let enhanced_telemetry = TrackingTelemetry {
                control_mode: ControlMode::Manual,
                ..telemetry
            };
            return (None, enhanced_telemetry);
        }

        let target = match &telemetry.target {
            Some(t) => t.clone(),
            None => {
                warn!("Tracking state is Tracking but no target present");
                let enhanced_telemetry = TrackingTelemetry {
                    control_mode: ControlMode::Manual,
                    ..telemetry
                };
                return (None, enhanced_telemetry);
            }
        };

        // Get target center in normalized coordinates
        let (center_x, _center_y) = target.bbox.center();

        // Calculate lateral error (horizontal offset from center)
        // Negative error = target on left, positive = target on right
        let error_x = center_x - 0.5;

        // Apply dead zone to reduce oscillation
        let error_x = if error_x.abs() < self.config.dead_zone {
            0.0
        } else {
            error_x
        };

        // Estimate distance to target
        let estimated_distance = self.estimate_distance(&target.bbox, &target.class_name);

        // Calculate longitudinal error (size-based distance control)
        // Positive error = too far, negative = too close
        let current_bbox_height = target.bbox.height();
        let error_size = self.config.target_bbox_height - current_bbox_height;

        // Compute PID outputs
        let omega_z = -self.lateral_pid.update(error_x as f64, dt);  // Negative for correct rotation direction
        let mut v_x = self.longitudinal_pid.update(error_size as f64, dt);

        // Safety: Don't move forward if too close
        if estimated_distance < self.config.min_distance {
            v_x = v_x.min(0.0);  // Only allow backward motion
            warn!("Target too close ({}m < {}m), limiting forward motion", estimated_distance, self.config.min_distance);
        }

        // Apply velocity limits
        let v_x = v_x.clamp(-self.config.max_velocity, self.config.max_velocity);
        let omega_z = omega_z.clamp(-self.config.max_angular_velocity, self.config.max_angular_velocity);

        debug!("Servo: error_x={:.3}, error_size={:.3}, distance={:.2}m, omega_z={:.3}, v_x={:.3}",
              error_x, error_size, estimated_distance, omega_z, v_x);

        // Create enhanced telemetry with distance and mode
        let control_output = ControlOutput::new(omega_z, v_x, error_x, error_size);
        let enhanced_telemetry = TrackingTelemetry {
            distance_estimate: Some(estimated_distance),
            control_output: Some(control_output),
            control_mode: ControlMode::Autonomous,
            ..telemetry
        };

        // Create rover command with metadata
        // Priority High (3) = Autonomous Tracking, below Emergency but above Manual Control
        let command = RoverCommand::new_velocity(omega_z, v_x, 0.0);
        let metadata = CommandMetadata {
            command_id: uuid::Uuid::new_v4().to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            source: InputSource::VisualServo,
            priority: CommandPriority::High,
        };

        self.last_command_time = Some(SystemTime::now());

        (Some(RoverCommandWithMetadata { command, metadata }), enhanced_telemetry)
    }
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    info!("Starting visual servo controller node");

    let mut controller = ServoController::new();
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let servo_cmd_output = DataId::from("servo_command".to_owned());
    let servo_telem_output = DataId::from("servo_telemetry".to_owned());

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata: _ } => {
                match id.as_str() {
                    "tracking_telemetry" => {
                        // Parse tracking telemetry
                        if let Some(array) = data.as_any().downcast_ref::<BinaryArray>() {
                            if array.is_empty() {
                                continue;
                            }

                            let bytes = array.value(0);
                            let telemetry: TrackingTelemetry = match serde_json::from_slice(bytes) {
                                Ok(t) => t,
                                Err(e) => {
                                    warn!("Failed to deserialize TrackingTelemetry: {}", e);
                                    continue;
                                }
                            };

                            // Calculate time delta
                            let dt = match controller.last_command_time {
                                Some(last_time) => {
                                    SystemTime::now()
                                        .duration_since(last_time)
                                        .unwrap_or(Duration::from_millis(100))
                                        .as_secs_f64()
                                }
                                None => 0.1,  // Default 100ms
                            };

                            // Process tracking and generate servo command + enhanced telemetry
                            let (command_opt, enhanced_telemetry) = controller.process_tracking(telemetry, dt);

                            // Send servo command if generated
                            if let Some(command_with_metadata) = command_opt {
                                let serialized = serde_json::to_vec(&command_with_metadata)?;
                                let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                                node.send_output(servo_cmd_output.clone(), Default::default(), arrow_data)?;
                            }

                            // Always send enhanced telemetry (with distance and mode)
                            let serialized = serde_json::to_vec(&enhanced_telemetry)?;
                            let arrow_data = BinaryArray::from_vec(vec![serialized.as_slice()]);
                            node.send_output(servo_telem_output.clone(), Default::default(), arrow_data)?;
                        }
                    }
                    other => {
                        warn!("Unexpected input: {}", other);
                    }
                }
            }
            Event::InputClosed { id } => {
                info!("Input {} closed", id);
                break;
            }
            Event::Stop(_) => {
                info!("Received stop event");
                break;
            }
            other => {
                warn!("Unexpected event: {:?}", other);
            }
        }
    }

    Ok(())
}
