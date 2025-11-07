// Command types for controlling the rover and arm

export interface JointPositions {
  shoulder_pan: number;
  shoulder_lift: number;
  elbow_flex: number;
  wrist_flex: number;
  wrist_roll: number;
  gripper: number;
  wheel1?: number;
  wheel2?: number;
  wheel3?: number;
}

export interface WebArmCommand {
  command_type: "joint_position" | "cartesian" | "home" | "stop";
  joint_positions?: JointPositions;
  max_velocity?: number;
}

export interface WebRoverCommand {
  command_type: "velocity" | "joint_positions" | "stop";
  v_x?: number;
  v_y?: number;
  omega_z?: number;
  wheel1?: number;
  wheel2?: number;
  wheel3?: number;
}

export interface VideoControl {
  command: VideoCommand;
  quality?: VideoQuality;
  max_fps?: number;
}

export type VideoCommand = "start" | "stop" | "pause" | "resume" | "change_quality";

export enum VideoQuality {
  Low = "low",
  Medium = "medium",
  High = "high",
  UltraHigh = "ultra_high",
}

export interface WebTrackingCommand {
  command_type: "enable" | "disable" | "select_target" | "clear_target";
  tracking_id?: number;
  detection_index?: number;
}

// Joint limits
export const JOINT_LIMITS = {
  shoulder_pan: { min: -3.14, max: 3.14 },
  shoulder_lift: { min: -1.57, max: 1.57 },
  elbow_flex: { min: -2.09, max: 2.09 },
  wrist_flex: { min: -3.14, max: 3.14 },
  wrist_roll: { min: -1.57, max: 1.57 },
  gripper: { min: -3.14, max: 3.14 },
};

// Helper functions
export function createHomePosition(): JointPositions {
  return {
    shoulder_pan: 0.0,
    shoulder_lift: 0.0,
    elbow_flex: 0.0,
    wrist_flex: 0.0,
    wrist_roll: 0.0,
    gripper: 0.0,
  };
}

export function validateJointPositions(positions: JointPositions): string | null {
  const checks: Array<[keyof JointPositions, { min: number; max: number }]> = [
    ["shoulder_pan", JOINT_LIMITS.shoulder_pan],
    ["shoulder_lift", JOINT_LIMITS.shoulder_lift],
    ["elbow_flex", JOINT_LIMITS.elbow_flex],
    ["wrist_flex", JOINT_LIMITS.wrist_flex],
    ["wrist_roll", JOINT_LIMITS.wrist_roll],
    ["gripper", JOINT_LIMITS.gripper],
  ];

  for (const [joint, limits] of checks) {
    const value = positions[joint];
    if (value! < limits.min || value! > limits.max) {
      return `${joint} out of range: ${value!.toFixed(3)} (expected ${limits.min.toFixed(2)} to ${limits.max.toFixed(2)})`;
    }
  }

  return null;
}
