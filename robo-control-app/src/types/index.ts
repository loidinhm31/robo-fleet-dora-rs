// Central export for all types - provides backwards compatibility

// Commands
export type {
  JointPositions,
  WebArmCommand,
  WebRoverCommand,
  VideoControl,
  VideoCommand,
  WebTrackingCommand,
} from "./commands";
export { VideoQuality, JOINT_LIMITS, createHomePosition, validateJointPositions } from "./commands";

// Telemetry
export type {
  ArmTelemetry,
  RoverTelemetry,
  VideoFrame,
  OverlayData,
  VideoStats,
  CameraStatus,
} from "./telemetry";

// Tracking
export type {
  BoundingBox,
  DetectionResult,
  DetectionFrame,
  DetectionDisplaySettings,
  TrackingState,
  ControlMode,
  TrackingTarget,
  TrackingTelemetry,
  ControlOutput,
} from "./tracking";
export { DEFAULT_CLASS_COLORS, getClassColor } from "./tracking";

// Voice
export type {
  SpeechTranscription,
  SpeechStats,
} from "./voice";

// Performance
export type {
  NodeMetrics,
  SystemMetrics,
} from "./performance";

// UI
export type {
  LogEntry,
  ConnectionState,
} from "./ui";

// Socket
export type {
  ServerToClientEvents,
  ClientToServerEvents,
} from "./socket";

// Fleet
export type {
  FleetStatus,
  FleetSelectCommand,
  RoverStatus,
  FleetRosterUpdate,
} from "./fleet";
export { createFleetSelectCommand } from "./fleet";

// Keep robo.ts for backwards compatibility by re-exporting everything
export * from "./commands";
export * from "./telemetry";
export * from "./tracking";
export * from "./voice";
export * from "./performance";
export * from "./ui";
export * from "./socket";
export * from "./fleet";
