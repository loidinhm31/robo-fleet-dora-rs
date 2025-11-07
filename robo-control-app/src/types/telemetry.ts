// Telemetry data received from the rover

export interface ArmTelemetry {
  end_effector_pose: number[];
  joint_angles?: number[];
  joint_velocities?: number[];
  is_moving: boolean;
  source: string;
  timestamp: number;
}

export interface RoverTelemetry {
  position: [number, number];
  yaw: number;
  pitch: number;
  roll: number;
  velocity: number;
  timestamp: number;
  near_sample: boolean;
  picking_up: boolean;
  nav_angles?: number[];
  nav_dists?: number[];
}

export interface VideoFrame {
  timestamp: number;
  frame_id: number;
  width: number;
  height: number;
  format: string;
  quality: number;
  data: string;
  overlay_data?: OverlayData;
}

export interface OverlayData {
  rover_position?: [number, number];
  rover_velocity?: number;
  arm_position?: number[];
  battery_level?: number;
  signal_strength?: number;
  timestamp_text: string;
}

export interface VideoStats {
  timestamp: number;
  frames_processed: number;
  frames_dropped: number;
  avg_frame_size_kb: number;
  avg_processing_time_ms: number;
  current_fps: number;
  bandwidth_kbps: number;
}

export interface CameraStatus {
  is_active: boolean;
  fps: number;
  dropped_frames: number;
  capture_errors: number;
  last_frame_timestamp: number;
}
