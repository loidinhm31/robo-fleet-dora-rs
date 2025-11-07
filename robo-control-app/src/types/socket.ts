// Socket.IO event types

import { ArmTelemetry, RoverTelemetry, VideoFrame, VideoStats } from "./telemetry";
import { WebArmCommand, WebRoverCommand, VideoControl } from "./commands";

export interface ServerToClientEvents {
  video_frame: (frame: VideoFrame) => void;
  video_stats: (stats: VideoStats) => void;
  video_status: (status: { streaming: boolean; fps?: number }) => void;
  arm_telemetry: (telemetry: ArmTelemetry) => void;
  rover_telemetry: (telemetry: RoverTelemetry) => void;
}

export interface ClientToServerEvents {
  arm_command: (command: WebArmCommand) => void;
  rover_command: (command: WebRoverCommand) => void;
  video_control: (control: VideoControl) => void;
}
