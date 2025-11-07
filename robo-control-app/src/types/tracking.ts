// Object detection and tracking types

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionResult {
  bbox: BoundingBox;
  class_id: number;
  class_name: string;
  confidence: number;
  tracking_id?: number;
}

export interface DetectionFrame {
  frame_id: number;
  timestamp: number;
  width: number;
  height: number;
  detections: DetectionResult[];
}

export interface DetectionDisplaySettings {
  enabled: boolean;
  showLabels: boolean;
  showConfidence: boolean;
  showBoundingBoxes: boolean;
  minConfidence: number;
  classColors: Record<string, string>;
}

export const DEFAULT_CLASS_COLORS: Record<string, string> = {
  person: "#00ff00",
  dog: "#ff00ff",
  cat: "#ff8800",
  car: "#0088ff",
  bicycle: "#ffff00",
  motorcycle: "#ff0088",
  bus: "#8800ff",
  truck: "#00ffff",
  bird: "#88ff00",
};

export function getClassColor(className: string): string {
  return DEFAULT_CLASS_COLORS[className] || "#ffffff";
}

export type TrackingState = "Disabled" | "Enabled" | "Tracking" | "TargetLost";
export type ControlMode = "Manual" | "Autonomous";

export interface TrackingTarget {
  tracking_id: number;
  class_name: string;
  bbox: BoundingBox;
  last_seen: number;
  confidence: number;
  lost_frames: number;
}

export interface TrackingTelemetry {
  state: TrackingState;
  target: TrackingTarget | null;
  distance_estimate: number | null;
  control_output: ControlOutput | null;
  control_mode: ControlMode;
  timestamp: number;
}

export interface ControlOutput {
  omega_z: number;
  v_x: number;
  error_x: number;
  error_size: number;
}
