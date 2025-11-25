"""
Configuration management for PyBullet simulation.

Loads environment variables and provides robot parameters including:
- Robot geometry (wheel radius, chassis radius)
- Motor limits (torque, velocity)
- Physics settings
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class WheelConfig:
    """Configuration for mecanum wheels."""
    radius: float  # meters
    max_torque: float  # Newton-meters
    max_velocity: float  # rad/s

    @classmethod
    def from_env(cls) -> 'WheelConfig':
        """Load wheel configuration from environment variables."""
        # ST3215/STS3215 specs:
        # - Torque: 30 kg·cm = 2.94 N·m
        # - Speed: 0.0017 s/60° = 61.76 RPM = 6.47 rad/s
        return cls(
            radius=float(os.getenv('WHEEL_RADIUS', '0.05')),  # 50mm default
            max_torque=float(os.getenv('MAX_WHEEL_TORQUE', '2.94')),  # 30 kg·cm
            max_velocity=float(os.getenv('MAX_WHEEL_VELOCITY', '6.47'))  # ~62 RPM
        )


@dataclass
class ChassisConfig:
    """Configuration for robot chassis."""
    radius: float  # meters - distance from center to wheel
    mass: float  # kg

    @classmethod
    def from_env(cls) -> 'ChassisConfig':
        """Load chassis configuration from environment variables."""
        return cls(
            radius=float(os.getenv('CHASSIS_RADIUS', '0.15')),  # 150mm default
            mass=float(os.getenv('CHASSIS_MASS', '2.0'))  # 2kg default
        )


@dataclass
class ArmConfig:
    """Configuration for 6-DOF robotic arm."""
    joint_count: int = 6
    max_torque: float = 2.94  # N·m (same servo as wheels)
    max_velocity: float = 6.47  # rad/s

    # Joint limits (radians) - based on ST3215 typical range
    joint_limits: list = None

    def __post_init__(self):
        if self.joint_limits is None:
            # Default: ±180° for most joints, ±90° for wrist
            self.joint_limits = [
                (-3.14, 3.14),  # shoulder_pan
                (-3.14, 3.14),  # shoulder_lift
                (-3.14, 3.14),  # elbow_flex
                (-1.57, 1.57),  # wrist_flex
                (-3.14, 3.14),  # wrist_roll
                (0.0, 1.57),    # gripper (0 = open, 90° = closed)
            ]

    @classmethod
    def from_env(cls) -> 'ArmConfig':
        """Load arm configuration from environment variables."""
        return cls(
            max_torque=float(os.getenv('MAX_ARM_TORQUE', '2.94')),
            max_velocity=float(os.getenv('MAX_ARM_VELOCITY', '6.47'))
        )


@dataclass
class PhysicsConfig:
    """Physics simulation parameters."""
    time_step: float = 1.0 / 240.0  # PyBullet default
    gravity: float = -9.81  # m/s²
    friction_coefficient: float = 0.7  # Rubber on wood/concrete
    restitution: float = 0.1  # Low bounce
    lateral_friction: float = 0.9  # Mecanum wheel side friction
    spinning_friction: float = 0.01  # Rolling resistance

    @classmethod
    def from_env(cls) -> 'PhysicsConfig':
        """Load physics configuration from environment variables."""
        return cls(
            time_step=float(os.getenv('PHYSICS_TIME_STEP', '0.004166')),  # 240 Hz
            gravity=float(os.getenv('GRAVITY', '-9.81')),
            friction_coefficient=float(os.getenv('FRICTION_COEFFICIENT', '0.7')),
            restitution=float(os.getenv('RESTITUTION', '0.1')),
            lateral_friction=float(os.getenv('LATERAL_FRICTION', '0.9')),
            spinning_friction=float(os.getenv('SPINNING_FRICTION', '0.01'))
        )


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    urdf_path: str
    wheel: WheelConfig
    chassis: ChassisConfig
    arm: ArmConfig
    physics: PhysicsConfig
    gui_enabled: bool = True
    real_time_simulation: bool = False

    @classmethod
    def from_env(cls) -> 'SimulationConfig':
        """Load complete configuration from environment variables."""
        urdf_path = os.getenv('URDF_PATH')
        if not urdf_path:
            raise ValueError("URDF_PATH environment variable must be set")

        return cls(
            urdf_path=urdf_path,
            wheel=WheelConfig.from_env(),
            chassis=ChassisConfig.from_env(),
            arm=ArmConfig.from_env(),
            physics=PhysicsConfig.from_env(),
            gui_enabled=os.getenv('GUI_ENABLED', 'true').lower() in ('true', '1', 'yes'),
            real_time_simulation=os.getenv('REAL_TIME', 'false').lower() in ('true', '1', 'yes')
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        import os.path

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        if self.wheel.radius <= 0:
            raise ValueError(f"Invalid wheel radius: {self.wheel.radius}")

        if self.chassis.radius <= 0:
            raise ValueError(f"Invalid chassis radius: {self.chassis.radius}")

        if self.wheel.max_torque <= 0:
            raise ValueError(f"Invalid max torque: {self.wheel.max_torque}")

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = [
            "=== PyBullet Simulation Configuration ===",
            f"URDF: {self.urdf_path}",
            "",
            "Wheel:",
            f"  Radius: {self.wheel.radius * 1000:.1f} mm",
            f"  Max Torque: {self.wheel.max_torque:.2f} N·m ({self.wheel.max_torque / 0.0981:.1f} kg·cm)",
            f"  Max Velocity: {self.wheel.max_velocity:.2f} rad/s ({self.wheel.max_velocity * 60 / (2 * 3.14159):.1f} RPM)",
            "",
            "Chassis:",
            f"  Radius: {self.chassis.radius * 1000:.1f} mm",
            f"  Mass: {self.chassis.mass:.2f} kg",
            "",
            "Arm:",
            f"  Joints: {self.arm.joint_count}",
            f"  Max Torque: {self.arm.max_torque:.2f} N·m",
            f"  Max Velocity: {self.arm.max_velocity:.2f} rad/s",
            "",
            "Physics:",
            f"  Time Step: {self.physics.time_step * 1000:.2f} ms ({1.0 / self.physics.time_step:.1f} Hz)",
            f"  Gravity: {self.physics.gravity:.2f} m/s²",
            f"  Friction: {self.physics.friction_coefficient:.2f}",
            f"  Lateral Friction: {self.physics.lateral_friction:.2f}",
            "",
            f"GUI: {'Enabled' if self.gui_enabled else 'Disabled'}",
            f"Real-time: {'Enabled' if self.real_time_simulation else 'Disabled'}",
            "========================================"
        ]
        return "\n".join(lines)


def get_config() -> SimulationConfig:
    """
    Get and validate simulation configuration.

    Returns:
        SimulationConfig: Validated configuration

    Raises:
        ValueError: If required environment variables are missing
        FileNotFoundError: If URDF file doesn't exist
    """
    config = SimulationConfig.from_env()
    config.validate()
    return config


if __name__ == "__main__":
    # Test configuration loading
    import sys

    try:
        config = get_config()
        print(config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        print("\nRequired environment variables:", file=sys.stderr)
        print("  URDF_PATH - Path to LeKiwi.urdf file", file=sys.stderr)
        print("\nOptional environment variables:", file=sys.stderr)
        print("  WHEEL_RADIUS - Wheel radius in meters (default: 0.05)", file=sys.stderr)
        print("  MAX_WHEEL_TORQUE - Max torque in N·m (default: 2.94)", file=sys.stderr)
        print("  MAX_WHEEL_VELOCITY - Max velocity in rad/s (default: 6.47)", file=sys.stderr)
        print("  CHASSIS_RADIUS - Chassis radius in meters (default: 0.15)", file=sys.stderr)
        print("  GUI_ENABLED - Enable PyBullet GUI (default: true)", file=sys.stderr)
        sys.exit(1)
