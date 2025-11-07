import {useEffect, useRef, useState} from "react";
import {
  Activity,
  Camera,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Crosshair,
  Eye,
  EyeOff,
  Layers,
  Maximize2,
  Minimize2,
  Power,
  Target,
  Volume2,
  VolumeX,
  X,
  XCircle
} from "lucide-react";
import {Socket} from "socket.io-client";
import {DetectionFrame, getClassColor, TrackingTelemetry, WebTrackingCommand} from "../types/robo.ts";

type ViewMode = "camera" | "camera_with_detections" | "detections_only";

interface JPEGVideoFrame {
  timestamp: number;
  frame_id: number;
  width: number;
  height: number;
  codec: "jpeg";
  data: number[]; // JPEG image as byte array
}

interface AudioFrame {
  timestamp: number;
  frame_id: number;
  sample_rate: number;
  channels: number;
  format: string; // "s16le", "f32le", etc.
  data: number[]; // PCM audio data as byte array
}

interface StreamStats {
  video_frames_received: number;
  video_fps: number;
  video_bitrate_kbps: number;
  audio_frames_received: number;
  audio_buffer_ms: number;
  detections_received: number;
  detection_fps: number;
  total_objects_detected: number;
}

interface CameraViewerProps {
  isConnected: boolean;
  socket: Socket | null;
  onClose?: () => void;
}

export const CameraViewer: React.FC<CameraViewerProps> = ({
  isConnected,
  socket,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(new Image());

  const [streamEnabled, setStreamEnabled] = useState(false);
  const [videoEnabled, setVideoEnabled] = useState(true);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("camera_with_detections");
  const [latestDetections, setLatestDetections] = useState<DetectionFrame | null>(null);
  const [trackedDetections, setTrackedDetections] = useState<DetectionFrame | null>(null);
  const [trackingTelemetry, setTrackingTelemetry] = useState<TrackingTelemetry | null>(null);
  const [showStats, setShowStats] = useState(true);
  const [showDetections, setShowDetections] = useState(true);
  const [showTracking, setShowTracking] = useState(true);
  const [showControls, setShowControls] = useState(true);
  const [stats, setStats] = useState<StreamStats>({
    video_frames_received: 0,
    video_fps: 0,
    video_bitrate_kbps: 0,
    audio_frames_received: 0,
    audio_buffer_ms: 0,
    detections_received: 0,
    detection_fps: 0,
    total_objects_detected: 0,
  });

  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());
  const bytesReceivedRef = useRef(0);
  const detectionCountRef = useRef(0);
  const lastDetectionFpsUpdateRef = useRef(Date.now());

  // Audio playback references
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<AudioBuffer[]>([]);
  const nextPlayTimeRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(false);
  const audioBufferThreshold = useRef<number>(5); // Minimum buffers before starting playback (increased for stability)
  const lowPassFilterRef = useRef<BiquadFilterNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const maxBufferQueueSize = useRef<number>(20); // Max queue size to prevent excessive latency

  // Draw detection bounding boxes on canvas
  const drawDetections = (ctx: CanvasRenderingContext2D, detections: DetectionFrame, canvasWidth: number, canvasHeight: number, overlay: boolean = true) => {
    detections.detections.forEach((detection) => {
      const { bbox, class_name, confidence, tracking_id } = detection;

      // Convert normalized coordinates to pixel coordinates
      const x1 = bbox.x1 * canvasWidth;
      const y1 = bbox.y1 * canvasHeight;
      const x2 = bbox.x2 * canvasWidth;
      const y2 = bbox.y2 * canvasHeight;
      const width = x2 - x1;
      const height = y2 - y1;

      // Check if this is the currently tracked object
      const isTracked = trackingTelemetry?.target?.tracking_id === tracking_id;
      const hasTrackingId = tracking_id !== undefined;

      // Get color for this class
      const color = getClassColor(class_name);

      // Draw bounding box with different styles for tracked objects
      if (isTracked) {
        // Tracked object: thicker, pulsing border
        ctx.strokeStyle = "#00ff00"; // Bright green for tracked target
        ctx.lineWidth = overlay ? 5 : 6;
        ctx.setLineDash([10, 5]); // Dashed line for emphasis
      } else {
        ctx.strokeStyle = color;
        ctx.lineWidth = overlay ? 3 : 4;
        ctx.setLineDash([]); // Solid line
      }
      ctx.strokeRect(x1, y1, width, height);
      ctx.setLineDash([]); // Reset dash

      // Draw tracking ID badge if available
      if (hasTrackingId) {
        const idBadge = `ID: ${tracking_id}`;
        ctx.font = "12px Arial";
        const idMetrics = ctx.measureText(idBadge);
        const idPadding = 4;

        ctx.fillStyle = isTracked ? "#00ff00" : "#4444ff";
        ctx.fillRect(x2 - idMetrics.width - idPadding * 2, y1, idMetrics.width + idPadding * 2, 18);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(idBadge, x2 - idMetrics.width - idPadding, y1 + 14);
      }

      // Draw center point and crosshair for tracked object
      const centerX = (x1 + x2) / 2;
      const centerY = (y1 + y2) / 2;

      if (isTracked) {
        // Crosshair for tracked target
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 2;
        const crossSize = 15;
        ctx.beginPath();
        ctx.moveTo(centerX - crossSize, centerY);
        ctx.lineTo(centerX + crossSize, centerY);
        ctx.moveTo(centerX, centerY - crossSize);
        ctx.lineTo(centerX, centerY + crossSize);
        ctx.stroke();
      } else if (!overlay) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Draw label background
      const label = `${class_name} ${(confidence * 100).toFixed(0)}%`;
      ctx.font = overlay ? "16px Arial" : "18px Arial";
      const textMetrics = ctx.measureText(label);
      const textHeight = overlay ? 20 : 24;
      const padding = 6;

      ctx.fillStyle = isTracked ? "#00ff00" : color;
      ctx.fillRect(x1, y1 - textHeight - padding, textMetrics.width + padding * 2, textHeight + padding);

      // Draw label text
      ctx.fillStyle = isTracked ? "#000000" : "#000000";
      ctx.fillText(label, x1 + padding, y1 - padding);
    });
  };

  // Draw detections-only view (clean background with bounding boxes)
  const drawDetectionsOnly = (ctx: CanvasRenderingContext2D, detections: DetectionFrame, canvasWidth: number, canvasHeight: number) => {
    // Clear canvas with dark background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw grid for reference (optional)
    ctx.strokeStyle = "#333333";
    ctx.lineWidth = 1;
    const gridSize = 50;
    for (let x = 0; x < canvasWidth; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasHeight);
      ctx.stroke();
    }
    for (let y = 0; y < canvasHeight; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasWidth, y);
      ctx.stroke();
    }

    // Draw center crosshair
    ctx.strokeStyle = "#666666";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(canvasWidth / 2 - 20, canvasHeight / 2);
    ctx.lineTo(canvasWidth / 2 + 20, canvasHeight / 2);
    ctx.moveTo(canvasWidth / 2, canvasHeight / 2 - 20);
    ctx.lineTo(canvasWidth / 2, canvasHeight / 2 + 20);
    ctx.stroke();

    // Draw detections
    drawDetections(ctx, detections, canvasWidth, canvasHeight, false);
  };

  // Handle video frames from Socket.IO
  useEffect(() => {
    if (!socket || !streamEnabled) return;

    const handleVideoFrame = (frame: JPEGVideoFrame) => {
      setStats((prev) => ({
        ...prev,
        video_frames_received: prev.video_frames_received + 1,
      }));

      if (!canvasRef.current || !videoEnabled) return;

      try {
        // Convert number array to Uint8Array
        const jpegData = new Uint8Array(frame.data);
        bytesReceivedRef.current += jpegData.length;

        // Create blob from JPEG data
        const blob = new Blob([jpegData], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);

        // Load and render JPEG to canvas
        const img = imgRef.current;
        img.onload = () => {
          if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
              // Set canvas size to match frame
              if (canvasRef.current.width !== frame.width ||
                  canvasRef.current.height !== frame.height) {
                canvasRef.current.width = frame.width;
                canvasRef.current.height = frame.height;
              }

              // Render based on view mode
              if (viewMode === "detections_only") {
                // Detections-only view: show only bounding boxes on dark background
                const detectionsToShow = trackedDetections || latestDetections;
                if (detectionsToShow && detectionsToShow.detections.length > 0) {
                  drawDetectionsOnly(ctx, detectionsToShow, frame.width, frame.height);
                } else {
                  // No detections - show empty grid
                  ctx.fillStyle = "#1a1a1a";
                  ctx.fillRect(0, 0, frame.width, frame.height);
                  ctx.fillStyle = "#666666";
                  ctx.font = "20px Arial";
                  ctx.textAlign = "center";
                  ctx.fillText("No objects detected", frame.width / 2, frame.height / 2);
                  ctx.textAlign = "left";
                }
              } else {
                // Camera view or camera + detections view
                ctx.drawImage(img, 0, 0, frame.width, frame.height);

                // Draw detections overlay if view mode includes detections
                // Prefer tracked detections (with IDs) over raw detections
                if (viewMode === "camera_with_detections") {
                  const detectionsToShow = trackedDetections || latestDetections;
                  if (detectionsToShow) {
                    drawDetections(ctx, detectionsToShow, frame.width, frame.height, true);
                  }
                }
              }
            }
          }

          // Clean up blob URL
          URL.revokeObjectURL(url);

          // Update FPS counter
          frameCountRef.current++;
          const now = Date.now();
          if (now - lastFpsUpdateRef.current >= 1000) {
            const elapsed = (now - lastFpsUpdateRef.current) / 1000;
            const fps = frameCountRef.current / elapsed;
            const bitrate = (bytesReceivedRef.current * 8) / elapsed / 1000; // kbps

            setStats(prev => ({
              ...prev,
              video_fps: fps,
              video_bitrate_kbps: bitrate
            }));

            frameCountRef.current = 0;
            bytesReceivedRef.current = 0;
            lastFpsUpdateRef.current = now;
          }
        };

        img.onerror = () => {
          console.error("❌ Failed to load JPEG image");
          URL.revokeObjectURL(url);
        };

        img.src = url;
      } catch (error) {
        console.error("❌ Error processing video frame:", error);
      }
    };

    socket.on("video_frame", handleVideoFrame);

    return () => {
      socket.off("video_frame", handleVideoFrame);
    };
  }, [socket, streamEnabled, videoEnabled, viewMode, latestDetections, trackedDetections, trackingTelemetry]);

  // Initialize Audio Context
  useEffect(() => {
    if (!streamEnabled || !audioEnabled) return;

    // Create AudioContext on first use (must be after user interaction)
    if (!audioContextRef.current) {
      try {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();

        // Create gain node for volume control
        gainNodeRef.current = audioContextRef.current.createGain();
        gainNodeRef.current.gain.value = 1.0;

        // Create low-pass filter to reduce high-frequency noise
        // This helps with resampling artifacts and microphone noise
        lowPassFilterRef.current = audioContextRef.current.createBiquadFilter();
        lowPassFilterRef.current.type = 'lowpass';
        lowPassFilterRef.current.frequency.value = 8000; // Cut off above 8kHz
        lowPassFilterRef.current.Q.value = 0.7; // Gentle slope

        // Connect audio chain: source -> gain -> filter -> destination
        gainNodeRef.current.connect(lowPassFilterRef.current);
        lowPassFilterRef.current.connect(audioContextRef.current.destination);

        console.log("AudioContext initialized:", {
          sampleRate: audioContextRef.current.sampleRate,
          state: audioContextRef.current.state,
          lowPassFilter: {
            frequency: lowPassFilterRef.current.frequency.value,
            type: lowPassFilterRef.current.type
          }
        });
      } catch (error) {
        console.error("❌ Failed to create AudioContext:", error);
      }
    }

    // Resume audio context if suspended (browser autoplay policy)
    if (audioContextRef.current?.state === "suspended") {
      audioContextRef.current.resume().then(() => {
        console.log("AudioContext resumed");
      });
    }

    return () => {
      // Don't close AudioContext on cleanup - keep it for next enable
    };
  }, [streamEnabled, audioEnabled]);

  // Handle audio frames from Socket.IO
  useEffect(() => {
    if (!socket || !streamEnabled || !audioEnabled) return;

    const handleAudioFrame = async (frame: AudioFrame) => {
      setStats((prev) => ({
        ...prev,
        audio_frames_received: prev.audio_frames_received + 1,
      }));

      if (!audioContextRef.current) {
        console.warn("AudioContext not initialized");
        return;
      }

      try {
        const audioContext = audioContextRef.current;
        const pcmData = new Uint8Array(frame.data);

        // Log detailed frame info for debugging
        if (stats.audio_frames_received < 5) {
          console.log("Audio frame details:", {
            frame_id: frame.frame_id,
            timestamp: frame.timestamp,
            sample_rate: frame.sample_rate,
            channels: frame.channels,
            format: frame.format,
            data_bytes: pcmData.length,
            first_10_bytes: Array.from(pcmData.slice(0, 10))
          });
        }

        // Calculate number of samples (S16LE = 2 bytes per sample)
        const totalSamples = pcmData.length / 2;
        const samplesPerChannel = Math.floor(totalSamples / frame.channels);

        if (samplesPerChannel <= 0) {
          console.warn("Invalid audio frame: no samples");
          return;
        }

        const durationMs = (samplesPerChannel / frame.sample_rate) * 1000;
        if (stats.audio_frames_received < 5) {
          console.log(`Calculated: ${samplesPerChannel} samples/channel, ${durationMs.toFixed(1)}ms duration`);
        }

        // Create AudioBuffer at the source sample rate
        // The browser will handle resampling to the AudioContext rate
        const audioBuffer = audioContext.createBuffer(
          frame.channels,
          samplesPerChannel,
          frame.sample_rate
        );

        // Convert S16LE PCM to Float32 for each channel
        if (frame.channels === 1) {
          // Mono audio - simpler processing
          const channelData = audioBuffer.getChannelData(0);
          for (let i = 0; i < samplesPerChannel; i++) {
            const offset = i * 2;
            const byte0 = pcmData[offset] ?? 0;
            const byte1 = pcmData[offset + 1] ?? 0;

            // Combine bytes to 16-bit little-endian
            const sample = byte0 | (byte1 << 8);

            // Convert unsigned to signed
            const signedSample = sample > 32767 ? sample - 65536 : sample;

            // Normalize to [-1.0, 1.0]
            channelData[i] = signedSample / 32768.0;
          }
        } else {
          // Stereo/multi-channel - interleaved data
          for (let channel = 0; channel < frame.channels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);

            for (let i = 0; i < samplesPerChannel; i++) {
              // Interleaved: [L0, R0, L1, R1, ...]
              const sampleIndex = i * frame.channels + channel;
              const offset = sampleIndex * 2;

              const byte0 = pcmData[offset] ?? 0;
              const byte1 = pcmData[offset + 1] ?? 0;

              const sample = byte0 | (byte1 << 8);
              const signedSample = sample > 32767 ? sample - 65536 : sample;
              channelData[i] = signedSample / 32768.0;
            }
          }
        }

        // Queue audio buffer for playback (with max queue size limit)
        if (audioQueueRef.current.length < maxBufferQueueSize.current) {
          audioQueueRef.current.push(audioBuffer);
        } else {
          // Drop oldest buffer if queue is full to prevent excessive latency
          audioQueueRef.current.shift();
          audioQueueRef.current.push(audioBuffer);
          console.warn("Audio queue full, dropping oldest buffer");
        }

        // Update buffer stats
        const bufferDuration = audioQueueRef.current.reduce((sum, buf) => sum + buf.duration, 0);
        setStats(prev => ({
          ...prev,
          audio_buffer_ms: bufferDuration * 1000
        }));

        // Start playback only if we have enough buffers to prevent underruns
        if (!isPlayingRef.current && audioQueueRef.current.length >= audioBufferThreshold.current) {
          console.log(`🔊 Starting audio playback with ${audioQueueRef.current.length} buffers (${bufferDuration.toFixed(3)}s)`);
          isPlayingRef.current = true;
          // Initialize next play time with a small delay to build buffer
          nextPlayTimeRef.current = audioContext.currentTime + 0.1;
          scheduleNextAudioBuffer();
        }

      } catch (error) {
        console.error("Error processing audio frame:", error, frame);
      }
    };

    // Schedule and play audio buffers from queue
    const scheduleNextAudioBuffer = () => {
      if (!audioContextRef.current || !gainNodeRef.current) {
        isPlayingRef.current = false;
        return;
      }

      const audioContext = audioContextRef.current;

      // Check if we have buffers to play
      if (audioQueueRef.current.length === 0) {
        console.warn("Audio buffer underrun - queue empty, stopping playback");
        isPlayingRef.current = false;
        // Reset timing for next playback start
        nextPlayTimeRef.current = 0;
        return;
      }

      const audioBuffer = audioQueueRef.current.shift();

      if (!audioBuffer) {
        isPlayingRef.current = false;
        return;
      }

      // Create buffer source
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;

      // Connect through gain node (which connects to filter -> destination)
      source.connect(gainNodeRef.current);

      // Schedule playback with seamless timing
      const currentTime = audioContext.currentTime;

      // Sync timing: if we're behind, catch up gradually
      if (nextPlayTimeRef.current < currentTime) {
        nextPlayTimeRef.current = currentTime;
      }

      const playTime = nextPlayTimeRef.current;

      // Detect large gaps in playback (>100ms)
      const gap = playTime - currentTime;
      if (gap > 0.1) {
        console.warn(`Audio timing drift: ${(gap * 1000).toFixed(1)}ms ahead`);
        // Adjust to prevent excessive latency buildup
        nextPlayTimeRef.current = currentTime + 0.05;
      }

      source.start(playTime);
      nextPlayTimeRef.current = playTime + audioBuffer.duration;

      // Schedule next buffer slightly before this one ends to ensure continuity
      const schedulingTime = (audioBuffer.duration * 1000) - 10; // 10ms before end
      setTimeout(() => {
        if (isPlayingRef.current) {
          scheduleNextAudioBuffer();
        }
      }, Math.max(schedulingTime, 0));

      // Log playback info periodically
      if (Math.random() < 0.05) {
        console.log(`Audio: ${audioBuffer.duration.toFixed(3)}s buffer, queue: ${audioQueueRef.current.length}, latency: ${(gap * 1000).toFixed(1)}ms`);
      }
    };

    socket.on("audio_frame", handleAudioFrame);

    return () => {
      socket.off("audio_frame", handleAudioFrame);

      // Clear audio queue on cleanup
      audioQueueRef.current = [];
      isPlayingRef.current = false;
    };
  }, [socket, streamEnabled, audioEnabled]);

  // Handle detection frames from Socket.IO
  useEffect(() => {
    if (!socket || !streamEnabled) return;

    const handleDetections = (detectionFrame: DetectionFrame) => {
      setLatestDetections(detectionFrame);

      // Update detection stats
      setStats((prev) => ({
        ...prev,
        detections_received: prev.detections_received + 1,
        total_objects_detected: detectionFrame.detections.length,
      }));

      // Update detection FPS
      detectionCountRef.current++;
      const now = Date.now();
      if (now - lastDetectionFpsUpdateRef.current >= 1000) {
        const elapsed = (now - lastDetectionFpsUpdateRef.current) / 1000;
        const fps = detectionCountRef.current / elapsed;

        setStats(prev => ({
          ...prev,
          detection_fps: fps,
        }));

        detectionCountRef.current = 0;
        lastDetectionFpsUpdateRef.current = now;
      }
    };

    const handleTrackedDetections = (detectionFrame: DetectionFrame) => {
      setTrackedDetections(detectionFrame);
    };

    const handleTrackingTelemetry = (telemetry: TrackingTelemetry) => {
      setTrackingTelemetry(telemetry);
    };

    socket.on("detections", handleDetections);
    socket.on("tracked_detections", handleTrackedDetections);
    socket.on("tracking_telemetry", handleTrackingTelemetry);

    return () => {
      socket.off("detections", handleDetections);
      socket.off("tracked_detections", handleTrackedDetections);
      socket.off("tracking_telemetry", handleTrackingTelemetry);
    };
  }, [socket, streamEnabled]);

  // Stream control
  const toggleStream = () => {
    if (!socket) return;

    const newState = !streamEnabled;
    setStreamEnabled(newState);

    console.log(newState ? "Stream started" : "Stream stopped");
  };

  const toggleVideo = () => {
    const newState = !videoEnabled;
    setVideoEnabled(newState);
  };

  const toggleAudio = () => {
    if (!socket) return;

    const newState = !audioEnabled;
    setAudioEnabled(newState);

    socket.emit("audio_control", {
      command: newState ? "start" : "stop"
    });

    if (!newState) {
      // Clear audio queue when disabling
      audioQueueRef.current = [];
      isPlayingRef.current = false;
    }

    console.log(newState ? "Audio enabled" : "Audio disabled");
  };

  const toggleCamera = () => {
    if (!socket) return;

    const newState = !cameraEnabled;
    setCameraEnabled(newState);

    socket.emit("camera_control", {
      command: newState ? "start" : "stop"
    });

    console.log(newState ? "Camera enabled" : "Camera disabled");
  };

  const cycleViewMode = () => {
    const modes: ViewMode[] = ["camera", "camera_with_detections", "detections_only"];
    const currentIndex = modes.indexOf(viewMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    const newMode = modes[nextIndex]!; // Safe because nextIndex is always valid
    setViewMode(newMode);

    const modeNames: Record<ViewMode, string> = {
      camera: "Camera Only",
      camera_with_detections: "Camera + Detections",
      detections_only: "Detections Only"
    };
    console.log(`View mode: ${modeNames[newMode]}`);
  };

  const toggleFullscreen = () => {
    if (!canvasRef.current) return;

    if (!isFullscreen) {
      canvasRef.current.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
    setIsFullscreen(!isFullscreen);
  };

  // Tracking control functions
  const sendTrackingCommand = (command: WebTrackingCommand) => {
    if (!socket) return;
    socket.emit("tracking_command", command);
  };

  const toggleTracking = () => {
    const isEnabled = trackingTelemetry?.state !== "Disabled";
    sendTrackingCommand({
      command_type: isEnabled ? "disable" : "enable",
    });
    console.log(isEnabled ? "Tracking disabled" : "Tracking enabled");
  };

  const selectTrackingTarget = (trackingId: number) => {
    sendTrackingCommand({
      command_type: "select_target",
      tracking_id: trackingId,
    });
    console.log(`Selected tracking target ID: ${trackingId}`);
  };

  const clearTrackingTarget = () => {
    sendTrackingCommand({
      command_type: "clear_target",
    });
    console.log("Cleared tracking target");
  };

  // Handle canvas click for target selection
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !trackedDetections) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width);
    const y = ((event.clientY - rect.top) / rect.height);

    // Find clicked detection
    for (const detection of trackedDetections.detections) {
      const { bbox, tracking_id } = detection;
      if (x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2) {
        if (tracking_id !== undefined) {
          selectTrackingTarget(tracking_id);
          return;
        }
      }
    }
  };

  return (
      <div className="relative w-full h-full bg-black rounded-lg overflow-hidden">
        {/* Canvas for rendering JPEG frames */}
        <canvas
            ref={canvasRef}
            className="w-full h-full object-contain cursor-crosshair"
            style={{ imageRendering: 'auto' }}
            onClick={handleCanvasClick}
        />

        {/* Controls overlay with toggle */}
        <div className="absolute top-4 right-4 flex flex-row gap-2">
          {/* Control buttons */}

            <div className="flex flex-col gap-2">
              {/* Close button - only show if onClose is provided */}
              {onClose && (
                <>
                  <button
                      onClick={onClose}
                      className="p-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg backdrop-blur-md transition border border-red-400/30"
                      title="Close Camera View"
                  >
                    <X className="w-5 h-5 text-red-400" />
                  </button>
                  <div className="h-px bg-white/20 my-1" />
                </>
              )}

              {/* Toggle button for controls */}
              <button
                  onClick={() => setShowControls(!showControls)}
                  className="p-2 bg-black/40 hover:bg-black/60 border border-white/20 rounded-lg backdrop-blur-sm transition shadow-lg"
                  title={showControls ? "Hide controls" : "Show controls"}
              >
                {showControls ? (
                    <ChevronRight className="w-5 h-5 text-gray-300" />
                ) : (
                    <ChevronLeft className="w-5 h-5 text-blue-400" />
                )}
              </button>

              {showControls && (
              <>
                <button
                    onClick={toggleStream}
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-md transition"
                    title={streamEnabled ? "Stop Stream" : "Start Stream"}
                >
                  {streamEnabled ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
                </button>

                <button
                    onClick={toggleCamera}
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-md transition"
                    title={cameraEnabled ? "Turn Camera Off" : "Turn Camera On"}
                    disabled={!isConnected}
                >
                  <Power className={`w-5 h-5 ${!cameraEnabled ? "text-red-400" : "text-green-400"}`} />
                </button>

                <button
                    onClick={toggleVideo}
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-md transition"
                    title={videoEnabled ? "Disable Video" : "Enable Video"}
                    disabled={!streamEnabled}
                >
                  <Camera className={`w-5 h-5 ${!videoEnabled ? "text-red-400" : ""}`} />
                </button>

                <button
                    onClick={toggleAudio}
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-md transition"
                    title={audioEnabled ? "Turn Audio Off" : "Turn Audio On"}
                    disabled={!isConnected}
                >
                  {audioEnabled ? <Volume2 className="w-5 h-5 text-green-400" /> : <VolumeX className="w-5 h-5 text-red-400" />}
                </button>

                <button
                    onClick={cycleViewMode}
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-md transition group relative"
                    title="Cycle View Mode"
                >
                  <Layers className={`w-5 h-5 ${
                      viewMode === "camera" ? "text-blue-400" :
                          viewMode === "camera_with_detections" ? "text-green-400" :
                              "text-purple-400"
                  }`} />
                  <span className="absolute right-full mr-2 px-2 py-1 bg-black/80 rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition pointer-events-none">
                  {viewMode === "camera" && "Camera Only"}
                    {viewMode === "camera_with_detections" && "Camera + Detections"}
                    {viewMode === "detections_only" && "Detections Only"}
                </span>
                </button>

                {/* Tracking controls divider */}
                <div className="h-px bg-white/20 my-1" />

                <button
                    onClick={toggleTracking}
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-md transition"
                    title={trackingTelemetry?.state !== "Disabled" ? "Disable Tracking" : "Enable Tracking"}
                    disabled={!isConnected}
                >
                  <Crosshair className={`w-5 h-5 ${
                      trackingTelemetry?.state === "Tracking" ? "text-green-400" :
                          trackingTelemetry?.state === "Enabled" ? "text-yellow-400" :
                              trackingTelemetry?.state === "TargetLost" ? "text-red-400" :
                                  "text-gray-400"
                  }`} />
                </button>

                {trackingTelemetry?.target && (
                    <button
                        onClick={clearTrackingTarget}
                        className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-md transition"
                        title="Clear Tracking Target"
                    >
                      <XCircle className="w-5 h-5 text-red-400" />
                    </button>
                )}

                <button
                    onClick={toggleFullscreen}
                    className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-md transition"
                    title="Toggle Fullscreen"
                >
                  {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
                </button>
              </>
              )}
            </div>
        </div>

        {/* View mode indicator badge */}
        {streamEnabled && (
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
              <div className={`px-4 py-2 rounded-full backdrop-blur-md text-xs font-semibold flex items-center gap-2 ${
                viewMode === "camera" ? "bg-blue-500/20 text-blue-300" :
                viewMode === "camera_with_detections" ? "bg-green-500/20 text-green-300" :
                "bg-purple-500/20 text-purple-300"
              }`}>
                <Layers className="w-4 h-4" />
                {viewMode === "camera" && "Camera Only"}
                {viewMode === "camera_with_detections" && "Camera + Detections"}
                {viewMode === "detections_only" && "Detections Only"}
              </div>
            </div>
        )}

        {/* Stats overlay */}
        {streamEnabled && (
            <div className="absolute bottom-4 left-4 flex items-end">
              {/* Stats panel */}
              {showStats && (
                <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-lg p-2 text-xs text-white shadow-lg">
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                    {/* Video stats */}
                    <div className="flex items-center gap-1.5">
                      <Camera className="w-3 h-3 text-blue-400" />
                      <span className="text-gray-400">Video:</span>
                    </div>
                    <span className="font-mono text-blue-300">{stats.video_fps.toFixed(1)} fps</span>

                    <span className="text-gray-400 col-start-1">Bitrate:</span>
                    <span className="font-mono text-blue-300">{stats.video_bitrate_kbps.toFixed(0)} kbps</span>

                    {/* Audio stats */}
                    <div className="flex items-center gap-1.5 col-start-1">
                      <Volume2 className="w-3 h-3 text-green-400" />
                      <span className="text-gray-400">Audio:</span>
                    </div>
                    <span className="font-mono text-green-300">{stats.audio_frames_received} frames</span>

                    <span className="text-gray-400 col-start-1">Buffer:</span>
                    <span className="font-mono text-green-300">{stats.audio_buffer_ms.toFixed(0)} ms</span>

                    {/* Detection stats - only show when detections are active */}
                    {viewMode !== "camera" && (
                      <>
                        <div className="flex items-center gap-1.5 col-start-1">
                          <Target className="w-3 h-3 text-purple-400" />
                          <span className="text-gray-400">Detect:</span>
                        </div>
                        <span className="font-mono text-purple-300">{stats.detection_fps.toFixed(1)} fps</span>

                        <span className="text-gray-400 col-start-1">Objects:</span>
                        <span className="font-mono text-purple-300">{stats.total_objects_detected}</span>
                      </>
                    )}

                    {/* Connection status */}
                    <div className="col-span-2 flex items-center gap-2 pt-1 border-t border-white/10 mt-1">
                      <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
                      <span className={isConnected ? "text-green-400" : "text-red-400"}>
                        {isConnected ? "Connected" : "Disconnected"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Toggle button */}
              <button
                onClick={() => setShowStats(!showStats)}
                className="p-2 bg-black/40 hover:bg-black/60 border border-white/20 rounded-lg backdrop-blur-sm transition shadow-lg"
                title={showStats ? "Hide stats" : "Show stats"}
              >
                {showStats ? (
                  <ChevronDown className="w-4 h-4 text-gray-300" />
                ) : (
                  <Activity className="w-4 h-4 text-blue-400" />
                )}
              </button>
            </div>
        )}

        {/* Connection warning */}
        {!isConnected && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/80">
              <div className="text-white text-center">
                <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">Not Connected</p>
                <p className="text-sm text-gray-400 mt-2">Waiting for connection...</p>
              </div>
            </div>
        )}

        {/* Tracking status panel */}
        {trackingTelemetry && trackingTelemetry.state !== "Disabled" && (
            <div className="absolute bottom-4 right-14 flex items-start">
              {/* Tracking panel */}
              {/* Toggle button */}
              <button
                  onClick={() => setShowTracking(!showTracking)}
                  className="p-2 bg-black/40 hover:bg-black/60 border border-white/20 rounded-lg backdrop-blur-sm transition shadow-lg"
                  title={showTracking ? "Hide tracking status" : "Show tracking status"}
              >
                {showTracking ? (
                    <ChevronDown className="w-4 h-4 text-gray-300" />
                ) : (
                    <Crosshair className="w-4 h-4 text-yellow-400" />
                )}
              </button>

              {showTracking && (
                <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-lg p-2 text-xs text-white max-w-xs shadow-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Crosshair className={`w-4 h-4 ${
                      trackingTelemetry.state === "Tracking" ? "text-green-400" :
                      trackingTelemetry.state === "Enabled" ? "text-yellow-400" :
                      "text-red-400"
                    }`} />
                    <span className="font-semibold">Tracking Status</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-gray-300">State:</span>
                      <span className={`font-medium ${
                        trackingTelemetry.state === "Tracking" ? "text-green-400" :
                        trackingTelemetry.state === "Enabled" ? "text-yellow-400" :
                        "text-red-400"
                      }`}>{trackingTelemetry.state}</span>
                    </div>
                    {trackingTelemetry.target && (
                      <>
                        <div className="h-px bg-white/20 my-1" />
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-gray-300">ID:</span>
                          <span className="font-mono">{trackingTelemetry.target.tracking_id}</span>
                        </div>
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-gray-300">Class:</span>
                          <span style={{ color: getClassColor(trackingTelemetry.target.class_name) }}>
                            {trackingTelemetry.target.class_name}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-gray-300">Confidence:</span>
                          <span>{(trackingTelemetry.target.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-gray-300">Lost Frames:</span>
                          <span className={trackingTelemetry.target.lost_frames > 10 ? "text-red-400" : ""}>
                            {trackingTelemetry.target.lost_frames}
                          </span>
                        </div>
                      </>
                    )}
                    <div className="mt-2 text-gray-400 italic text-xs">
                      Click on detected objects to track
                    </div>
                  </div>
                </div>
              )}
            </div>
        )}

        {/* Detection list panel */}
        {viewMode !== "camera" && (trackedDetections || latestDetections) && (
            <div className="absolute top-4 left-4 flex items-start">
              {/* Detection panel */}
              {showDetections && (
                <div className="bg-black/40 backdrop-blur-sm border border-white/20 rounded-lg p-2 text-xs text-white max-w-xs shadow-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="w-4 h-4 text-green-400" />
                    <span className="font-semibold">Detected Objects ({(trackedDetections || latestDetections)?.detections.length || 0})</span>
                  </div>
                  <div className="space-y-1 max-h-48 overflow-y-auto [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-white/20 [&::-webkit-scrollbar-thumb]:rounded">
                    {(trackedDetections || latestDetections)?.detections.map((detection, index) => {
                      const isTracked = trackingTelemetry?.target?.tracking_id === detection.tracking_id;
                      return (
                        <div
                          key={index}
                          className={`flex items-center justify-between gap-2 py-1 px-2 rounded cursor-pointer transition ${
                            isTracked ? "bg-green-500/30" : "bg-white/10 hover:bg-white/20"
                          }`}
                          style={{ borderLeft: `3px solid ${isTracked ? "#00ff00" : getClassColor(detection.class_name)}` }}
                          onClick={() => detection.tracking_id && selectTrackingTarget(detection.tracking_id)}
                        >
                          <div className="flex items-center gap-2">
                            {detection.tracking_id !== undefined && (
                              <span className="font-mono text-xs bg-white/20 px-1 rounded">
                                {detection.tracking_id}
                              </span>
                            )}
                            <span className="font-medium">{detection.class_name}</span>
                          </div>
                          <span className="text-gray-300">{(detection.confidence * 100).toFixed(0)}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Toggle button */}
              <button
                onClick={() => setShowDetections(!showDetections)}
                className="p-2 bg-black/40 hover:bg-black/60 border border-white/20 rounded-lg backdrop-blur-sm transition shadow-lg"
                title={showDetections ? "Hide detected objects" : "Show detected objects"}
              >
                {showDetections ? (
                  <ChevronUp className="w-4 h-4 text-gray-300" />
                ) : (
                  <Target className="w-4 h-4 text-green-400" />
                )}
              </button>
            </div>
        )}
      </div>
  );
};
