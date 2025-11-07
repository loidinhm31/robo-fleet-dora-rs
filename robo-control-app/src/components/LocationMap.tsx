import React, { useEffect, useRef, useState } from "react";
import { RoverTelemetry } from "../types/robo.ts";
import { RotateCcw, ZoomIn, ZoomOut } from "lucide-react";

interface RobotLocationMapProps {
  telemetry: RoverTelemetry | null;
  width?: string;
  height?: string;
}

interface PathPoint {
  x: number;
  y: number;
  timestamp: number;
}

export const RobotLocationMap: React.FC<RobotLocationMapProps> = ({
                                                                    telemetry,
                                                                    width = "100%",
                                                                    height = "600px",
                                                                  }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pathHistory, setPathHistory] = useState<PathPoint[]>([]);
  const [scale, setScale] = useState(50); // pixels per meter
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showGrid, setShowGrid] = useState(true);
  const [showPath, setShowPath] = useState(true);
  const [showNavSensors, setShowNavSensors] = useState(true);

  const MAX_PATH_LENGTH = 500;
  const ROBOT_SIZE = 0.3; // meters

  // Add new position to path history
  useEffect(() => {
    if (telemetry?.position) {
      const [x, y] = telemetry.position;
      const newPoint: PathPoint = {
        x,
        y,
        timestamp: telemetry.timestamp,
      };

      setPathHistory((prev) => {
        const updated = [...prev, newPoint];
        return updated.slice(-MAX_PATH_LENGTH);
      });
    }
  }, [telemetry?.position, telemetry?.timestamp]);

  // Drawing functions
  const drawGrid = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    if (!showGrid) return;

    const centerX = canvas.width / 2 + offset.x;
    const centerY = canvas.height / 2 + offset.y;
    const gridSpacing = 1; // 1 meter

    ctx.strokeStyle = "rgba(99, 102, 241, 0.2)";
    ctx.lineWidth = 1;

    // Vertical lines
    for (let i = -20; i <= 20; i++) {
      const x = centerX + i * gridSpacing * scale;
      if (x >= 0 && x <= canvas.width) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();

        // Label every 5 meters
        if (i % 5 === 0) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
          ctx.font = "10px monospace";
          ctx.fillText(`${i}m`, x + 2, centerY + 12);
        }
      }
    }

    // Horizontal lines
    for (let i = -20; i <= 20; i++) {
      const y = centerY + i * gridSpacing * scale;
      if (y >= 0 && y <= canvas.height) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();

        // Label every 5 meters
        if (i % 5 === 0) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
          ctx.font = "10px monospace";
          ctx.fillText(`${-i}m`, centerX + 2, y - 2);
        }
      }
    }

    // Draw axes
    ctx.strokeStyle = "rgba(99, 102, 241, 0.5)";
    ctx.lineWidth = 2;

    // X-axis
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(canvas.width, centerY);
    ctx.stroke();

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, canvas.height);
    ctx.stroke();

    // Origin marker
    ctx.fillStyle = "rgba(99, 102, 241, 0.8)";
    ctx.beginPath();
    ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
    ctx.fill();
  };

  const drawPath = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    if (!showPath || pathHistory.length < 2) return;

    const centerX = canvas.width / 2 + offset.x;
    const centerY = canvas.height / 2 + offset.y;

    ctx.strokeStyle = "rgba(6, 182, 212, 0.6)";
    ctx.lineWidth = 2;
    ctx.beginPath();

    pathHistory.forEach((point, index) => {
      const x = centerX + point.x * scale;
      const y = centerY - point.y * scale; // Invert Y for screen coordinates

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw path points
    ctx.fillStyle = "rgba(6, 182, 212, 0.4)";
    pathHistory.forEach((point, index) => {
      if (index % 10 === 0) {
        // Draw every 10th point
        const x = centerX + point.x * scale;
        const y = centerY - point.y * scale;
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  };

  const drawNavigationSensors = (
    ctx: CanvasRenderingContext2D,
    _canvas: HTMLCanvasElement,
    robotX: number,
    robotY: number,
    yaw: number
  ) => {
    if (!showNavSensors || !telemetry?.nav_angles || !telemetry?.nav_dists) return;

    const { nav_angles, nav_dists } = telemetry;

    ctx.strokeStyle = "rgba(251, 146, 60, 0.6)";
    ctx.lineWidth = 1;

    nav_angles.forEach((angle, index) => {
      const distance = nav_dists[index];
      if (distance === undefined || distance > 5) return; // Skip far objects

      // Convert angle to world coordinates (relative to robot yaw)
      const worldAngle = yaw + angle;
      const endX = robotX + distance * Math.cos(worldAngle) * scale;
      const endY = robotY - distance * Math.sin(worldAngle) * scale;

      // Draw sensor line
      ctx.beginPath();
      ctx.moveTo(robotX, robotY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Draw obstacle point
      ctx.fillStyle = "rgba(251, 146, 60, 0.8)";
      ctx.beginPath();
      ctx.arc(endX, endY, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  const drawRobot = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    if (!telemetry) return;

    const centerX = canvas.width / 2 + offset.x;
    const centerY = canvas.height / 2 + offset.y;

    const [x, y] = telemetry.position;
    const yaw = telemetry.yaw;

    const robotX = centerX + x * scale;
    const robotY = centerY - y * scale; // Invert Y

    // Draw navigation sensors
    drawNavigationSensors(ctx, canvas, robotX, robotY, yaw);

    // Draw robot body (triangle pointing in direction of yaw)
    const size = ROBOT_SIZE * scale;

    ctx.save();
    ctx.translate(robotX, robotY);
    ctx.rotate(-yaw); // Negative because canvas Y is inverted

    // Robot body - triangle
    ctx.fillStyle = telemetry.near_sample
      ? "rgba(34, 197, 94, 0.8)" // Green when near sample
      : "rgba(6, 182, 212, 0.9)"; // Cyan normally
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(size, 0); // Front
    ctx.lineTo(-size * 0.5, size * 0.8); // Back left
    ctx.lineTo(-size * 0.5, -size * 0.8); // Back right
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Direction indicator
    ctx.strokeStyle = "rgba(251, 191, 36, 0.9)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(size * 1.3, 0);
    ctx.stroke();

    ctx.restore();

    // Draw velocity vector
    if (telemetry.velocity > 0.01) {
      ctx.strokeStyle = "rgba(251, 191, 36, 0.6)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(robotX, robotY);
      const velocityLength = telemetry.velocity * scale * 2;
      ctx.lineTo(
        robotX + velocityLength * Math.cos(yaw),
        robotY - velocityLength * Math.sin(yaw)
      );
      ctx.stroke();

      // Arrowhead
      ctx.fillStyle = "rgba(251, 191, 36, 0.8)";
      const arrowX = robotX + velocityLength * Math.cos(yaw);
      const arrowY = robotY - velocityLength * Math.sin(yaw);
      ctx.save();
      ctx.translate(arrowX, arrowY);
      ctx.rotate(-yaw);
      ctx.beginPath();
      ctx.moveTo(8, 0);
      ctx.lineTo(-4, 6);
      ctx.lineTo(-4, -6);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  };

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = "rgba(15, 23, 42, 0.95)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw components
    drawGrid(ctx, canvas);
    drawPath(ctx, canvas);
    drawRobot(ctx, canvas);
  };

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const animate = () => {
      draw();
      requestAnimationFrame(animate);
    };

    const animationId = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(animationId);
  }, [telemetry, pathHistory, scale, offset, showGrid, showPath, showNavSensors]);

  // Mouse interaction handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setScale((prev) => Math.max(10, Math.min(200, prev * delta)));
  };

  const zoomIn = () => setScale((prev) => Math.min(200, prev * 1.2));
  const zoomOut = () => setScale((prev) => Math.max(10, prev / 1.2));
  const resetView = () => {
    setScale(50);
    setOffset({ x: 0, y: 0 });
  };
  const clearPath = () => setPathHistory([]);

  return (
    <div className="relative" style={{ width, height }}>
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-2xl cursor-move"
        style={{ background: "linear-gradient(to bottom, #0f172a, #1e293b)" }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Control Panel */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button
          onClick={zoomIn}
          className="p-2 glass-card-light rounded-lg hover:bg-white/20 transition-all"
          title="Zoom In"
        >
          <ZoomIn className="w-5 h-5 text-white" />
        </button>
        <button
          onClick={zoomOut}
          className="p-2 glass-card-light rounded-lg hover:bg-white/20 transition-all"
          title="Zoom Out"
        >
          <ZoomOut className="w-5 h-5 text-white" />
        </button>
        <button
          onClick={resetView}
          className="p-2 glass-card-light rounded-lg hover:bg-white/20 transition-all"
          title="Reset View"
        >
          <RotateCcw className="w-5 h-5 text-white" />
        </button>
      </div>

      {/* Toggle Controls */}
      <div className="absolute bottom-4 left-4 glass-card-light rounded-lg p-3 space-y-2">
        <label className="flex items-center gap-2 text-xs text-white cursor-pointer">
          <input
            type="checkbox"
            checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)}
            className="rounded"
          />
          Grid
        </label>
        <label className="flex items-center gap-2 text-xs text-white cursor-pointer">
          <input
            type="checkbox"
            checked={showPath}
            onChange={(e) => setShowPath(e.target.checked)}
            className="rounded"
          />
          Path
        </label>
        <label className="flex items-center gap-2 text-xs text-white cursor-pointer">
          <input
            type="checkbox"
            checked={showNavSensors}
            onChange={(e) => setShowNavSensors(e.target.checked)}
            className="rounded"
          />
          Sensors
        </label>
        <button
          onClick={clearPath}
          className="w-full text-xs text-red-300 hover:text-red-200 transition-colors"
        >
          Clear Path
        </button>
      </div>

      {/* Info Display */}
      {telemetry && (
        <div className="absolute top-4 left-4 glass-card-light rounded-lg p-3 text-xs font-mono space-y-1">
          <div className="text-white/90">
            <span className="text-white/60">Position:</span>{" "}
            ({telemetry.position[0].toFixed(2)}, {telemetry.position[1].toFixed(2)})m
          </div>
          <div className="text-white/90">
            <span className="text-white/60">Heading:</span>{" "}
            {((telemetry.yaw * 180) / Math.PI).toFixed(1)}°
          </div>
          <div className="text-white/90">
            <span className="text-white/60">Velocity:</span>{" "}
            {telemetry.velocity.toFixed(2)} m/s
          </div>
          <div className="text-white/90">
            <span className="text-white/60">Scale:</span>{" "}
            {scale.toFixed(0)} px/m
          </div>
          {telemetry.near_sample && (
            <div className="text-green-400 font-bold animate-pulse">
              ⚠️ NEAR SAMPLE
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="absolute bottom-4 right-4 glass-card-light rounded-lg p-2 text-[10px] text-white/60">
        Drag to pan • Scroll to zoom
      </div>
    </div>
  );
};