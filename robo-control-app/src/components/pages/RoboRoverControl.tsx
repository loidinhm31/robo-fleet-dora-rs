import React, {
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { io, Socket } from "socket.io-client";
import { Joystick } from "react-joystick-component";

import { IJoystickUpdateEvent } from "react-joystick-component/build/lib/Joystick.js";
import {
  Activity,
  AlertTriangle,
  Camera,
  Eye,
  EyeOff,
  Gauge,
  Home,
  Radio,
  Zap,
} from "lucide-react";
import { CameraViewer } from "../CameraViewer.tsx";
import { RobotLocationMap } from "../LocationMap.tsx";
import { TranscriptionDisplay } from "../TranscriptionDisplay.tsx";
import { VoiceControls } from "../VoiceControls.tsx";
import { FloatingMetrics } from "../FloatingMetrics.tsx";
import {
  ArmTelemetry,
  ConnectionState,
  createHomePosition,
  createFleetSelectCommand,
  FleetStatus,
  JointPositions,
  LogEntry,
  RoverTelemetry,
  SpeechTranscription,
  SystemMetrics,
  TrackingTelemetry,
  validateJointPositions,
  WebArmCommand,
  WebRoverCommand,
} from "../../types";
import { IconBadge } from "../atoms";
import { CollapsibleSection } from "../molecules";
import { FleetSelector, JointControlPanel } from "../organisms";

// Load configuration from environment variables
const SOCKET_URL = import.meta.env.VITE_SOCKET_IO_URL || "http://localhost:3030";
const THROTTLE_DELAY = 100; // ms between updates

// Authentication credentials - loaded from environment variables
// CRITICAL: These must be set in .env file and match web_bridge configuration
const AUTH_USERNAME = import.meta.env.VITE_AUTH_USERNAME;
const AUTH_PASSWORD = import.meta.env.VITE_AUTH_PASSWORD;

// Validate that credentials are configured
if (!AUTH_USERNAME || !AUTH_PASSWORD) {
  console.error("CRITICAL: Authentication credentials not configured. Please set VITE_AUTH_USERNAME and VITE_AUTH_PASSWORD in .env file");
}

// Extended JointPositions with wheel visualization
interface ExtendedJointPositions extends JointPositions {
  wheel1: number;
  wheel2: number;
  wheel3: number;
}

const RoboRoverController: React.FC = () => {
  // Connection state
  const [connection, setConnection] = useState<ConnectionState>({
    isConnected: false,
    clientId: null,
    commandsSent: 0,
    commandsReceived: 0,
  });

  // Telemetry state
  const [, setArmTelemetry] = useState<ArmTelemetry | null>(null);
  const [roverTelemetry, setRoverTelemetry] = useState<RoverTelemetry | null>(
    null,
  );
  const [servoTelemetry, setServoTelemetry] = useState<TrackingTelemetry | null>(
    null,
  );

  // Speech recognition state
  const [transcription, setTranscription] = useState<SpeechTranscription | null>(
    null,
  );
  const [isAudioActive, setIsAudioActive] = useState(false);

  // Performance metrics state
  const [performanceMetrics, setPerformanceMetrics] = useState<SystemMetrics | null>(
    null,
  );

  // Fleet status state
  const [fleetStatus, setFleetStatus] = useState<FleetStatus | null>(null);

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [showCamera, setShowCamera] = useState(false);
  const [showLocationMap, setShowLocationMap] = useState(false);

  // LeKiwi joint position controls (now includes wheels)
  const [jointPositions, setJointPositions] =
    useState<ExtendedJointPositions>({
      ...createHomePosition(),
      wheel1: 0.0,
      wheel2: 0.0,
      wheel3: 0.0,
    });

  // Rover velocity controls
  const [roverVelocity, setRoverVelocity] = useState({
    v_x: 0.0,
    v_y: 0.0,
    omega_z: 0.0,
  });

  // UI state for collapsible sections
  const [expandedSections, setExpandedSections] = useState({
    armJoints: true,
    logs: false,
  });

  const socketRef = useRef<Socket | null>(null);
  const lastCommandTime = useRef<number>(0);
  const lastUpdateTime = useRef<number>(Date.now());
  const MAX_LOGS = 50;

  // Add log entry
  const addLog = useCallback(
    (message: string, type: LogEntry["type"] = "info") => {
      const entry: LogEntry = {
        timestamp: new Date().toISOString(),
        message,
        type,
      };
      setLogs((prev) => [entry, ...prev].slice(0, MAX_LOGS));
    },
    [],
  );

  // Throttled command sender
  const sendThrottled = useCallback((callback: () => void) => {
    const now = Date.now();
    if (now - lastCommandTime.current >= THROTTLE_DELAY) {
      callback();
      lastCommandTime.current = now;
    }
  }, []);

  // Connect to Socket.IO server
  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      addLog("Already connected", "warning");
      return;
    }

    const socket = io(SOCKET_URL, {
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      auth: {
        username: AUTH_USERNAME,
        password: AUTH_PASSWORD,
      },
    });

    socket.on("connect", () => {
      addLog(`Connected (ID: ${socket.id})`, "success");
      setConnection((prev) => ({
        ...prev,
        isConnected: true,
        clientId: socket.id as string,
      }));
    });

    socket.on("disconnect", (reason) => {
      addLog(`Disconnected: ${reason}`, "warning");
      setConnection((prev) => ({
        ...prev,
        isConnected: false,
        clientId: null,
      }));
    });

    socket.on("connect_error", (error) => {
      addLog(`Connection error: ${error.message}`, "error");
      // Check if it's an authentication error
      if (error.message.includes("authentication") || error.message.includes("auth")) {
        addLog("Authentication failed - check credentials", "error");
      }
    });

    socket.on("command_ack", () => {
      setConnection((prev) => ({
        ...prev,
        commandsReceived: prev.commandsReceived + 1,
      }));
    });

    socket.on("rover_core_telemetry", (data: RoverTelemetry) => {
      setRoverTelemetry(data);
    });

    socket.on("arm_telemetry", (data: ArmTelemetry) => {
      setArmTelemetry(data);
      if (data.joint_angles && data.joint_angles.length === 6) {
        setJointPositions((prev) => ({
          shoulder_pan: data.joint_angles![0] as number,
          shoulder_lift: data.joint_angles![1] as number,
          elbow_flex: data.joint_angles![2] as number,
          wrist_flex: data.joint_angles![3] as number,
          wrist_roll: data.joint_angles![4] as number,
          gripper: data.joint_angles![5] as number,
          // Keep wheel positions
          wheel1: prev.wheel1 as number,
          wheel2: prev.wheel2 as number,
          wheel3: prev.wheel3 as number,
        }));
      }
    });

    // Listen for servo telemetry (includes distance and control mode)
    socket.on("servo_telemetry", (data: TrackingTelemetry) => {
      setServoTelemetry(data);
    });

    // Listen for speech transcriptions
    socket.on("transcription", (data: SpeechTranscription) => {
      setTranscription(data);
      addLog(`Transcription: "${data.text}" (${(data.confidence * 100).toFixed(0)}%)`, "info");
    });

    // Listen for performance metrics
    socket.on("performance_metrics", (data: SystemMetrics) => {
      setPerformanceMetrics(data);
    });

    // Listen for fleet status updates
    socket.on("fleet_status", (data: FleetStatus) => {
      setFleetStatus(data);
      addLog(`Fleet status: Selected rover is ${data.selected_entity}`, "info");
    });

    socketRef.current = socket;
  }, [addLog]);

  // Disconnect from Socket.IO server
  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      addLog("Manually disconnected", "info");
    }
  }, [addLog]);

  // Select rover from fleet
  const selectRover = useCallback(
    (entityId: string) => {
      if (!connection.isConnected || !socketRef.current) {
        addLog("Cannot select rover - not connected", "error");
        return;
      }

      const selectCommand = createFleetSelectCommand(entityId);
      socketRef.current.emit("fleet_select", selectCommand);
      addLog(`Switching to rover: ${entityId}`, "info");
    },
    [connection.isConnected, addLog],
  );

  // Send ARM command
  const sendArmCommand = useCallback(
    (command: WebArmCommand) => {
      if (!connection.isConnected || !socketRef.current) {
        addLog("Cannot send command - not connected", "error");
        return;
      }

      socketRef.current.emit("arm_command", command);
      setConnection((prev) => ({
        ...prev,
        commandsSent: prev.commandsSent + 1,
      }));
    },
    [connection.isConnected, addLog],
  );

  // Send ROVER command
  const sendRoverCommand = useCallback(
    (command: WebRoverCommand) => {
      if (!connection.isConnected || !socketRef.current) {
        addLog("Cannot send rover command - not connected", "error");
        return;
      }

      socketRef.current.emit("rover_command", command);
      setConnection((prev) => ({
        ...prev,
        commandsSent: prev.commandsSent + 1,
      }));
    },
    [connection.isConnected, addLog],
  );

  // Audio control functions
  const startAudio = useCallback(() => {
    if (!connection.isConnected || !socketRef.current) {
      addLog("Cannot start audio - not connected", "error");
      return;
    }

    socketRef.current.emit("audio_control", { command: "start" });
    setIsAudioActive(true);
    addLog("Audio capture started", "success");
  }, [connection.isConnected, addLog]);

  const stopAudio = useCallback(() => {
    if (!connection.isConnected || !socketRef.current) {
      addLog("Cannot stop audio - not connected", "error");
      return;
    }

    socketRef.current.emit("audio_control", { command: "stop" });
    setIsAudioActive(false);
    addLog("Audio capture stopped", "info");
  }, [connection.isConnected, addLog]);

  // Update joint position
  const updateJoint = useCallback((joint: keyof JointPositions, value: number) => {
    setJointPositions((prev) => {
      const newPositions = { ...prev, [joint]: value };
      const error = validateJointPositions(newPositions);
      if (error) {
        console.warn(error);
      }
      return newPositions as ExtendedJointPositions;
    });
  }, []);

  // Real-time ARM joint control
  useEffect(() => {
    if (!connection.isConnected) return;

    const sendJointUpdate = () => {
      const command: WebArmCommand = {
        command_type: "joint_position",
        joint_positions: {
          shoulder_pan: jointPositions.shoulder_pan,
          shoulder_lift: jointPositions.shoulder_lift,
          elbow_flex: jointPositions.elbow_flex,
          wrist_flex: jointPositions.wrist_flex,
          wrist_roll: jointPositions.wrist_roll,
          gripper: jointPositions.gripper,
        },
      };
      sendArmCommand(command);
    };

    sendThrottled(sendJointUpdate);
  }, [
    jointPositions.shoulder_pan,
    jointPositions.shoulder_lift,
    jointPositions.elbow_flex,
    jointPositions.wrist_flex,
    jointPositions.wrist_roll,
    jointPositions.gripper,
    connection.isConnected,
    sendArmCommand,
    sendThrottled,
  ]);

  // Real-time ROVER velocity control
  useEffect(() => {
    if (!connection.isConnected) return;

    const sendVelocityUpdate = () => {
      const command: WebRoverCommand = {
        command_type: "velocity",
        v_x: roverVelocity.v_x,
        v_y: roverVelocity.v_y,
        omega_z: roverVelocity.omega_z,
      };
      sendRoverCommand(command);
    };

    sendThrottled(sendVelocityUpdate);
  }, [
    roverVelocity,
    connection.isConnected,
    sendRoverCommand,
    sendThrottled,
  ]);

  // **NEW: Integrate wheel velocities into wheel positions for visualization**
  useEffect(() => {
    const intervalId = setInterval(() => {
      const now = Date.now();
      const dt = (now - lastUpdateTime.current) / 1000; // Convert to seconds
      lastUpdateTime.current = now;

      // 3-wheel omnidirectional kinematics (120° apart)
      // Wheel radius (approximate, adjust if needed)
      const WHEEL_RADIUS = 0.05; // 5cm radius
      const ROBOT_RADIUS = 0.15; // Distance from center to wheel

      // Convert linear velocities to wheel angular velocities
      // For 3-wheel omni with 120° spacing:
      const { v_x, v_y, omega_z } = roverVelocity;

      // Wheel 1 (bottom, 0°): axis along [0, 0, -1]
      const omega1 = (v_y / WHEEL_RADIUS) + (omega_z * ROBOT_RADIUS / WHEEL_RADIUS);

      // Wheel 2 (right, 120°): axis along [0.866, 0, 0.5]
      const omega2 = ((-0.5 * v_y + 0.866 * v_x) / WHEEL_RADIUS) + (omega_z * ROBOT_RADIUS / WHEEL_RADIUS);

      // Wheel 3 (left, 240°): axis along [-0.866, 0, 0.5]
      const omega3 = ((-0.5 * v_y - 0.866 * v_x) / WHEEL_RADIUS) + (omega_z * ROBOT_RADIUS / WHEEL_RADIUS);

      // Integrate: position += velocity * dt
      setJointPositions((prev) => ({
        ...prev,
        wheel1: prev.wheel1 + omega1 * dt,
        wheel2: prev.wheel2 + omega2 * dt,
        wheel3: prev.wheel3 + omega3 * dt,
      }));
    }, 50); // Update at 20Hz

    return () => clearInterval(intervalId);
  }, [roverVelocity]);

  // Joystick move handler
  const handleJoystickMove = useCallback((event: IJoystickUpdateEvent) => {
    if (!event.x || !event.y) return;

    // Normalize joystick values (-100 to 100) to velocity range (-1 to 1)
    const v_y = event.x / 100; // Left/Right (x maps to v_y)
    const v_x = -event.y / 100; // Forward/Back (y maps to v_x, inverted)

    setRoverVelocity((prev) => ({
      ...prev,
      v_x: v_x,
      v_y: v_y,
    }));
  }, []);

  // Joystick stop handler
  const handleJoystickStop = useCallback(() => {
    setRoverVelocity((prev) => ({
      ...prev,
      v_x: 0,
      v_y: 0,
    }));
  }, []);

  // Home position
  const sendHome = useCallback(() => {
    const command: WebArmCommand = { command_type: "home" };
    sendArmCommand(command);
    setJointPositions({
      ...createHomePosition(),
      wheel1: 0.0,
      wheel2: 0.0,
      wheel3: 0.0,
    });
  }, [sendArmCommand]);

  // Emergency stop
  const emergencyStop = useCallback(() => {
    sendArmCommand({ command_type: "stop" });
    sendRoverCommand({ command_type: "stop" });
    setRoverVelocity({ v_x: 0, v_y: 0, omega_z: 0 });
    addLog("EMERGENCY STOP ACTIVATED", "warning");
  }, [sendArmCommand, sendRoverCommand, addLog]);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  return (
    <div className="min-h-screen gradient-bg relative">
      {/* Animated background elements - fixed positioning */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div
          className="cube-decoration top-20 left-10 bg-cyan-400"
          style={{ animationDelay: "0s" }}
        ></div>
        <div
          className="cube-decoration top-40 right-20 bg-blue-500"
          style={{ animationDelay: "2s" }}
        ></div>
        <div
          className="cube-decoration bottom-20 left-1/4 bg-orange-400"
          style={{ animationDelay: "4s" }}
        ></div>
        <div
          className="cube-decoration bottom-40 right-1/3 bg-yellow-400"
          style={{ animationDelay: "1s" }}
        ></div>
        <div
          className="cube-decoration top-1/2 left-1/2 bg-pink-400"
          style={{ animationDelay: "3s" }}
        ></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto">
        {/* Header - Sticky on scroll - Compact design */}
        <div className="sticky top-0 z-50 glass-card shadow-xl p-2 md:p-3 backdrop-blur-xl border-b border-white/10">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-2">
            {/* Left: Title and Status Indicators */}
            <div className="flex items-center gap-2 md:gap-3 flex-wrap w-full md:w-auto">
              <Zap className="w-6 h-6 text-yellow-400 animate-pulse" />
              <h1 className="text-lg md:text-xl font-bold text-white tracking-tight">
                LEKIWI ROBOT
              </h1>

              {/* Connection Status - Inline */}
              <div className="glass-card-light rounded-lg px-2 py-1 flex items-center gap-1.5">
                {connection.isConnected ? (
                  <>
                    <Radio className="w-3 h-3 text-green-400 animate-pulse" />
                    <span className="text-xs font-semibold text-green-300">
                      ONLINE
                    </span>
                  </>
                ) : (
                  <>
                    <Radio className="w-3 h-3 text-red-400" />
                    <span className="text-xs font-semibold text-red-300">
                      OFFLINE
                    </span>
                  </>
                )}
              </div>

              {/* Control Mode - Inline */}
              {servoTelemetry && (
                <div className="glass-card-light rounded-lg px-2 py-1 flex items-center gap-1.5">
                  {servoTelemetry.control_mode === "Autonomous" ? (
                    <>
                      <Zap className="w-3 h-3 text-blue-400 animate-pulse" />
                      <span className="text-xs font-semibold text-blue-300">
                        AUTO
                      </span>
                    </>
                  ) : (
                    <>
                      <Gauge className="w-3 h-3 text-purple-400" />
                      <span className="text-xs font-semibold text-purple-300">
                        MANUAL
                      </span>
                    </>
                  )}
                  {servoTelemetry.distance_estimate !== null && (
                    <span className="text-xs text-white/80 font-mono ml-1">
                      {servoTelemetry.distance_estimate.toFixed(1)}m
                    </span>
                  )}
                </div>
              )}

              {/* Commands Count */}
              <div className="text-xs text-white/50 font-mono hidden md:block">
                {connection.commandsSent} cmd
              </div>
            </div>

            {/* Right: Emergency Stop and Connect Button */}
            <div className="flex items-center gap-2 w-full md:w-auto">
              {!connection.isConnected && (
                <button
                  onClick={connect}
                  className="btn-gradient-cyan px-4 py-2 rounded-xl text-sm font-semibold whitespace-nowrap"
                >
                  CONNECT
                </button>
              )}

              {/* Emergency Stop Button - Compact */}
              <button
                onClick={emergencyStop}
                disabled={!connection.isConnected}
                className="group relative px-4 md:px-6 py-2 bg-gradient-to-br from-red-600 via-red-500 to-orange-500 text-white rounded-xl font-black text-sm md:text-base shadow-[0_0_20px_rgba(239,68,68,0.4)] hover:shadow-[0_0_30px_rgba(239,68,68,0.7)] transition-all duration-300 hover:scale-105 disabled:opacity-40 disabled:hover:scale-100 disabled:shadow-none border border-red-300/50 active:scale-95 flex-1 md:flex-none"
                style={{
                  animation: connection.isConnected ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none'
                }}
              >
                <div className="flex items-center gap-2 justify-center">
                  <AlertTriangle className="w-4 h-4 animate-pulse" />
                  <span className="tracking-wide">E-STOP</span>
                </div>
                {/* Shine effect */}
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
              </button>
            </div>
          </div>
        </div>

        <div className="p-3 md:p-4 space-y-3 md:space-y-4 pt-3 md:pt-4">
          {/* Fleet Selector */}
          <FleetSelector
            fleetStatus={fleetStatus}
            onSelectRover={selectRover}
            className="max-w-md"
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Location Map Viewer */}
            {showLocationMap && (
              <div className="glass-card rounded-3xl shadow-2xl p-4 md:p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Eye className="w-6 h-6 text-purple-400" />
                    <h2 className="text-2xl md:text-3xl font-bold text-white">
                      LOCATION MAP
                    </h2>
                  </div>
                  <button
                    onClick={() => setShowLocationMap(false)}
                    className="btn-gradient px-4 py-2 rounded-xl text-sm flex items-center gap-2"
                  >
                    <EyeOff className="w-4 h-4" />
                    Hide
                  </button>
                </div>
                <div className="glass-card-light rounded-2xl p-2 md:p-4">
                  <Suspense
                    fallback={
                      <div className="h-96 flex items-center justify-center text-white/60">
                        Loading Location Map...
                      </div>
                    }
                  >
                    <RobotLocationMap telemetry={roverTelemetry} />


                  </Suspense>
                </div>
                <div className="mt-3 text-xs text-white/60 text-center">
                  Use mouse to rotate • Scroll to zoom • Drag to pan
                </div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                  <div className="glass-card-light p-2 rounded-lg">
                    <div className="text-white/50">Wheel 1 (Bottom)</div>
                    <div className="text-cyan-300 font-mono">
                      {(jointPositions.wheel1 % (2 * Math.PI)).toFixed(2)} rad
                    </div>
                  </div>
                  <div className="glass-card-light p-2 rounded-lg">
                    <div className="text-white/50">Wheel 2 (Right)</div>
                    <div className="text-cyan-300 font-mono">
                      {(jointPositions.wheel2 % (2 * Math.PI)).toFixed(2)} rad
                    </div>
                  </div>
                  <div className="glass-card-light p-2 rounded-lg">
                    <div className="text-white/50">Wheel 3 (Left)</div>
                    <div className="text-cyan-300 font-mono">
                      {(jointPositions.wheel3 % (2 * Math.PI)).toFixed(2)} rad
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Camera Viewer */}
            {showCamera && (
              <CameraViewer
                isConnected={connection.isConnected}
                socket={socketRef.current}
                onClose={() => setShowCamera(false)}
              />
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {!showLocationMap && (
              <button
                onClick={() => setShowLocationMap(true)}
                className="w-full py-3 glass-card-light rounded-2xl text-white/80 hover:text-white transition-all hover:scale-105 flex items-center justify-center gap-2"
              >
                <Eye className="w-5 h-5" />
                Show Location Map
              </button>
            )}
            {!showCamera && (
              <button
                onClick={() => setShowCamera(true)}
                className="w-full py-3 glass-card-light rounded-2xl text-white/80 hover:text-white transition-all hover:scale-105 flex items-center justify-center gap-2"
              >
                <Camera className="w-5 h-5" />
                Show Camera Feed
              </button>
            )}
          </div>

          {/* Speech Transcription Display */}
          <div className="mt-3">
            <TranscriptionDisplay
              transcription={transcription}
              isAudioActive={isAudioActive}
              maxHistory={5}
              onStartAudio={startAudio}
              onStopAudio={stopAudio}
            />
          </div>

          {/* Voice Communication Controls */}
          <VoiceControls
            socket={socketRef.current}
            isConnected={connection.isConnected}
            onLog={addLog}
          />

          {/* Main Control Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 md:gap-4">
            {/* LEFT COLUMN: ROVER CONTROL */}
            <div className="space-y-3 md:space-y-4">
              <div className="glass-card rounded-3xl shadow-2xl p-4 md:p-6">
                <div className="flex items-center gap-2 mb-4 md:mb-6">
                  <Activity className="w-6 h-6 md:w-8 md:h-8 text-cyan-400" />
                  <h2 className="text-2xl md:text-3xl font-bold text-white">
                    ROVER
                  </h2>
                </div>

                {/* Joystick Control */}
                <div className="flex flex-col items-center space-y-4">
                  <div className="glass-card-light rounded-full p-4 md:p-6 relative">
                    <Joystick
                      size={window.innerWidth < 768 ? 180 : 240}
                      baseColor="rgba(255, 255, 255, 0.1)"
                      stickColor="linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)"
                      move={handleJoystickMove}
                      stop={handleJoystickStop}
                      throttle={50}
                    />
                    <div className="absolute inset-0 rounded-full border-4 border-cyan-400/30 pointer-events-none"></div>
                  </div>

                  <div className="text-white/90 text-center font-medium text-sm">
                    Drag to control rover movement
                  </div>
                </div>

                {/* Rotation Control */}
                <div className="mt-4 glass-card-light rounded-2xl p-4 md:p-5 space-y-3">
                  <div className="flex justify-between text-xs md:text-sm font-semibold text-white">
                    <span>Rotation (ω)</span>
                    <span className="text-cyan-300 font-mono">
                      {roverVelocity.omega_z.toFixed(2)} rad/s
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-1.0"
                    max="1.0"
                    step="0.05"
                    value={roverVelocity.omega_z}
                    onChange={(e) =>
                      setRoverVelocity((prev) => ({
                        ...prev,
                        omega_z: parseFloat(e.target.value),
                      }))
                    }
                    className="glass-slider w-full"
                  />
                  <div className="flex justify-between text-xs text-white/50 font-mono">
                    <span>-1.0</span>
                    <span>0.0</span>
                    <span>+1.0</span>
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN: ARM CONTROL */}
            <div className="space-y-3 md:space-y-4">
              <CollapsibleSection
                title="ARM JOINTS"
                isExpanded={expandedSections.armJoints}
                onToggle={() =>
                  setExpandedSections((prev) => ({
                    ...prev,
                    armJoints: !prev.armJoints,
                  }))
                }
                headerRight={
                  <IconBadge icon={Gauge} color="text-purple-400" size="md" />
                }
              >
                <JointControlPanel
                  jointPositions={jointPositions}
                  onJointChange={updateJoint}
                  disabled={!connection.isConnected}
                />
                <button
                  onClick={sendHome}
                  disabled={!connection.isConnected}
                  className="w-full py-3 md:py-4 btn-gradient rounded-2xl font-semibold text-base md:text-lg flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed mt-4"
                >
                  <Home className="w-5 h-5" />
                  HOME POSITION
                </button>
              </CollapsibleSection>
            </div>
          </div>

          {/* Activity Logs */}
          <CollapsibleSection
            title={`ACTIVITY LOG (${logs.length})`}
            isExpanded={expandedSections.logs}
            onToggle={() =>
              setExpandedSections((prev) => ({
                ...prev,
                logs: !prev.logs,
              }))
            }
            headerRight={
              <IconBadge icon={Activity} color="text-cyan-400" size="md" />
            }
            contentClassName="backdrop-blur-md bg-black/40 rounded-2xl p-3 md:p-4 max-h-48 overflow-y-auto font-mono text-xs space-y-1 border border-white/10"
          >
            {logs.length === 0 ? (
              <div className="text-white/30 text-center py-8">
                No activity yet...
              </div>
            ) : (
              logs.slice(0, 10).map((log, idx) => (
                <div
                  key={idx}
                  className={`${
                    log.type === "error"
                      ? "text-red-300"
                      : log.type === "success"
                        ? "text-green-300"
                        : log.type === "warning"
                          ? "text-yellow-300"
                          : "text-cyan-200"
                  }`}
                >
                  <span className="text-white/40">
                    [{new Date(log.timestamp).toLocaleTimeString()}]
                  </span>{" "}
                  {log.message}
                </div>
              ))
            )}
          </CollapsibleSection>

          {/* Quick Info */}
          <div className="glass-card rounded-3xl shadow-2xl p-3 md:p-4">
            <div className="flex flex-col md:flex-row items-center justify-center gap-3 md:gap-6 text-xs md:text-sm text-white/80">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                <span>
                  <span className="font-bold text-white">Real-time Control</span>{" "}
                  - Arm + Wheels active
                </span>
              </div>
              <div className="hidden md:block w-px h-6 bg-white/20"></div>
              <div>
                <span className="font-bold text-white">Throttle:</span>{" "}
                {THROTTLE_DELAY}ms
              </div>
              <div className="hidden md:block w-px h-6 bg-white/20"></div>
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4" />
                <span className="font-bold text-white">
                  Location Map:
                </span>{" "}
                {showLocationMap ? "Active" : "Hidden"}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Floating Performance Metrics */}
      <FloatingMetrics metrics={performanceMetrics} socket={socketRef.current} />
    </div>
  );
};

export default RoboRoverController;