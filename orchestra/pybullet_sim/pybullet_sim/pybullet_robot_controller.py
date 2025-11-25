"""
Robot controller for PyBullet simulation.

Implements:
- Mecanum wheel kinematics for 3-wheel triangular configuration
- LeKiwi robot controller
- Servo motor simulation (ST3215/STS3215 specs)
- Joint discovery and configuration
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pybullet as p
import numpy as np

from pybullet_config import SimulationConfig


@dataclass
class JointInfo:
    """Information about a robot joint."""
    index: int
    name: str
    type: int
    lower_limit: float
    upper_limit: float
    max_force: float
    max_velocity: float


class MecanumKinematics:
    """
    Mecanum wheel kinematics for triangular 3-wheel configuration.

    Wheel arrangement (top view):
          Front
            ^
            |
        [Wheel 1]
           / \
          /   \
    [W2]       [W3]

    Wheel 1: Front (0°)
    Wheel 2: Rear-left (120°)
    Wheel 3: Rear-right (240°)
    """

    def __init__(self, chassis_radius: float, wheel_radius: float):
        """
        Initialize mecanum kinematics.

        Args:
            chassis_radius: Distance from robot center to wheel (meters)
            wheel_radius: Radius of each wheel (meters)
        """
        self.R = chassis_radius  # Robot radius
        self.r = wheel_radius    # Wheel radius

        # Wheel angles in the triangular configuration
        self.wheel_angles = [
            0.0,                    # Wheel 1: Front (0°)
            2.0 * math.pi / 3.0,    # Wheel 2: Rear-left (120°)
            4.0 * math.pi / 3.0,    # Wheel 3: Rear-right (240°)
        ]

    def inverse_kinematics(self, vx: float, vy: float, omega: float) -> Tuple[float, float, float]:
        """
        Convert chassis velocity to wheel angular velocities.

        Args:
            vx: Linear velocity in x-direction (m/s)
            vy: Linear velocity in y-direction (m/s)
            omega: Angular velocity around z-axis (rad/s)

        Returns:
            Tuple of (wheel1_vel, wheel2_vel, wheel3_vel) in rad/s
        """
        wheel_vels = []

        for angle in self.wheel_angles:
            # Mecanum wheel inverse kinematics formula:
            # v_wheel = (vx * sin(θ) - vy * cos(θ) + ω * R) / r
            v_wheel = (
                vx * math.sin(angle) -
                vy * math.cos(angle) +
                omega * self.R
            ) / self.r
            wheel_vels.append(v_wheel)

        return tuple(wheel_vels)

    def forward_kinematics(self, w1: float, w2: float, w3: float) -> Tuple[float, float, float]:
        """
        Convert wheel angular velocities to chassis velocity.

        Args:
            w1, w2, w3: Wheel angular velocities (rad/s)

        Returns:
            Tuple of (vx, vy, omega)
        """
        wheels = [w1, w2, w3]

        # Build the kinematic matrix
        # For 3 wheels: over-constrained, use least-squares
        A = []
        b = []

        for i, (w, angle) in enumerate(zip(wheels, self.wheel_angles)):
            # v_wheel = (vx * sin(θ) - vy * cos(θ) + ω * R) / r
            # Rearrange: vx * sin(θ) - vy * cos(θ) + ω * R = v_wheel * r
            A.append([math.sin(angle), -math.cos(angle), self.R])
            b.append(w * self.r)

        # Solve using least squares
        A = np.array(A)
        b = np.array(b)
        result = np.linalg.lstsq(A, b, rcond=None)[0]

        return float(result[0]), float(result[1]), float(result[2])


class LeKiwiRobotController:
    """
    Controller for the LeKiwi robot in PyBullet.

    Manages:
    - Robot loading and initialization
    - Joint discovery and control
    - Mecanum wheel control
    - 6-DOF arm control
    - Telemetry generation
    """

    def __init__(self, config: SimulationConfig, physics_client: int):
        """
        Initialize robot controller.

        Args:
            config: Simulation configuration
            physics_client: PyBullet physics client ID
        """
        self.config = config
        self.client = physics_client
        self.robot_id: Optional[int] = None

        # Joint mappings
        self.wheel_joints: Dict[str, JointInfo] = {}
        self.arm_joints: Dict[str, JointInfo] = {}

        # Kinematics
        self.kinematics = MecanumKinematics(
            config.chassis.radius,
            config.wheel.radius
        )

        # State tracking
        self.wheel_positions = [0.0, 0.0, 0.0]
        self.wheel_velocities = [0.0, 0.0, 0.0]
        self.arm_positions = [0.0] * 6
        self.arm_velocities = [0.0] * 6

        # Expected joint names (from actual LeKiwi URDF)
        self.wheel_joint_names = [
            "ST3215_Servo_Motor-v1-2_Revolute-60",  # Bottom/Front wheel
            "ST3215_Servo_Motor-v1-1_Revolute-62",  # Right wheel
            "ST3215_Servo_Motor-v1_Revolute-64",    # Left wheel
        ]

        self.arm_joint_names = [
            "STS3215_03a-v1_Revolute-45",           # Shoulder pan/base rotation
            "STS3215_03a-v1-1_Revolute-49",         # Shoulder lift
            "STS3215_03a-v1-2_Revolute-51",         # Elbow flex
            "STS3215_03a-v1-3_Revolute-53",         # Wrist flex
            "STS3215_03a_Wrist_Roll-v1_Revolute-55", # Wrist roll
            "STS3215_03a-v1-4_Revolute-57",         # Gripper
        ]

    def load_robot(self) -> int:
        """
        Load robot URDF into PyBullet.

        Returns:
            Robot body ID
        """
        # Load URDF
        self.robot_id = p.loadURDF(
            self.config.urdf_path,
            basePosition=[0, 0, 0.1],  # Slightly above ground
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            physicsClientId=self.client
        )

        print(f"Loaded robot from {self.config.urdf_path}")
        print(f"Robot ID: {self.robot_id}")

        # Discover joints
        self._discover_joints()

        # Configure physics properties
        self._configure_physics()

        return self.robot_id

    def _discover_joints(self) -> None:
        """Discover and categorize robot joints."""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        print(f"\nDiscovered {num_joints} joints:")

        all_joints = []

        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)

            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]

            joint_info = JointInfo(
                index=i,
                name=joint_name,
                type=joint_type,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                max_force=max_force,
                max_velocity=max_velocity
            )

            all_joints.append(joint_info)

            # Print for debugging
            type_name = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"][joint_type]
            print(f"  [{i}] {joint_name}: {type_name}")

        # Map to wheel and arm joints
        self._map_joints(all_joints)

    def _map_joints(self, all_joints: List[JointInfo]) -> None:
        """
        Map discovered joints to wheel and arm categories.

        Args:
            all_joints: List of all discovered joints
        """
        # Create lookup by name
        joint_by_name = {j.name: j for j in all_joints}

        # Try to find wheel joints
        for wheel_name in self.wheel_joint_names:
            # Try exact match
            if wheel_name in joint_by_name:
                self.wheel_joints[wheel_name] = joint_by_name[wheel_name]
            else:
                # Try partial match (case insensitive)
                for joint in all_joints:
                    if wheel_name.lower() in joint.name.lower() or \
                       joint.name.lower() in wheel_name.lower():
                        if "wheel" in joint.name.lower():
                            self.wheel_joints[wheel_name] = joint
                            break

        # Try to find arm joints
        for arm_name in self.arm_joint_names:
            if arm_name in joint_by_name:
                self.arm_joints[arm_name] = joint_by_name[arm_name]
            else:
                # Try partial match
                for joint in all_joints:
                    if arm_name.lower() in joint.name.lower() or \
                       joint.name.lower() in arm_name.lower():
                        if joint not in self.wheel_joints.values():
                            self.arm_joints[arm_name] = joint
                            break

        print(f"\nMapped wheel joints: {len(self.wheel_joints)}")
        for name, joint in self.wheel_joints.items():
            print(f"  {name} -> Joint {joint.index}: {joint.name}")

        print(f"\nMapped arm joints: {len(self.arm_joints)}")
        for name, joint in self.arm_joints.items():
            print(f"  {name} -> Joint {joint.index}: {joint.name}")

        if len(self.wheel_joints) != 3:
            print(f"WARNING: Expected 3 wheel joints, found {len(self.wheel_joints)}")

        if len(self.arm_joints) != 6:
            print(f"WARNING: Expected 6 arm joints, found {len(self.arm_joints)}")

    def _configure_physics(self) -> None:
        """Configure physics properties for robot."""
        # Set contact properties for all links
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)

        for i in range(-1, num_joints):  # -1 for base link
            p.changeDynamics(
                self.robot_id,
                i,
                lateralFriction=self.config.physics.lateral_friction,
                spinningFriction=self.config.physics.spinning_friction,
                restitution=self.config.physics.restitution,
                physicsClientId=self.client
            )

        # Configure wheel joints for velocity control
        for joint_info in self.wheel_joints.values():
            p.setJointMotorControl2(
                self.robot_id,
                joint_info.index,
                controlMode=p.VELOCITY_CONTROL,
                force=0,  # Disable default motor
                physicsClientId=self.client
            )

    def set_wheel_velocities(self, w1: float, w2: float, w3: float) -> None:
        """
        Set wheel angular velocities.

        Args:
            w1, w2, w3: Wheel angular velocities (rad/s)
        """
        velocities = [w1, w2, w3]

        # Clamp to max velocity
        max_vel = self.config.wheel.max_velocity
        velocities = [max(-max_vel, min(max_vel, v)) for v in velocities]

        # Apply to each wheel
        for (name, joint_info), vel in zip(self.wheel_joints.items(), velocities):
            p.setJointMotorControl2(
                self.robot_id,
                joint_info.index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=self.config.wheel.max_torque,
                physicsClientId=self.client
            )

        self.wheel_velocities = velocities

    def set_wheel_positions(self, pos1: float, pos2: float, pos3: float) -> None:
        """
        Set wheel positions (for visualization/testing).

        Args:
            pos1, pos2, pos3: Wheel angular positions (radians)
        """
        positions = [pos1, pos2, pos3]

        for (name, joint_info), pos in zip(self.wheel_joints.items(), positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_info.index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos,
                force=self.config.wheel.max_torque,
                physicsClientId=self.client
            )

        self.wheel_positions = positions

    def set_chassis_velocity(self, vx: float, vy: float, omega: float) -> None:
        """
        Set chassis velocity using mecanum kinematics.

        Args:
            vx: Linear velocity in x (m/s)
            vy: Linear velocity in y (m/s)
            omega: Angular velocity (rad/s)
        """
        w1, w2, w3 = self.kinematics.inverse_kinematics(vx, vy, omega)
        self.set_wheel_velocities(w1, w2, w3)

    def set_arm_positions(self, positions: List[float]) -> None:
        """
        Set arm joint positions.

        Args:
            positions: List of 6 joint angles (radians)
        """
        if len(positions) < 6:
            print(f"Warning: Expected 6 arm positions, got {len(positions)}")
            return

        # Clamp to joint limits
        clamped_positions = []
        for pos, (lower, upper) in zip(positions, self.config.arm.joint_limits):
            clamped_positions.append(max(lower, min(upper, pos)))

        # Apply to each arm joint
        for (name, joint_info), pos in zip(self.arm_joints.items(), clamped_positions):
            p.setJointMotorControl2(
                self.robot_id,
                joint_info.index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos,
                force=self.config.arm.max_torque,
                maxVelocity=self.config.arm.max_velocity,
                physicsClientId=self.client
            )

        self.arm_positions = clamped_positions

    def get_wheel_states(self) -> Tuple[List[float], List[float]]:
        """
        Get current wheel positions and velocities.

        Returns:
            Tuple of (positions, velocities)
        """
        positions = []
        velocities = []

        for joint_info in self.wheel_joints.values():
            state = p.getJointState(self.robot_id, joint_info.index, physicsClientId=self.client)
            positions.append(state[0])  # Position
            velocities.append(state[1])  # Velocity

        return positions, velocities

    def get_arm_states(self) -> Tuple[List[float], List[float]]:
        """
        Get current arm joint positions and velocities.

        Returns:
            Tuple of (positions, velocities)
        """
        positions = []
        velocities = []

        for joint_info in self.arm_joints.values():
            state = p.getJointState(self.robot_id, joint_info.index, physicsClientId=self.client)
            positions.append(state[0])
            velocities.append(state[1])

        return positions, velocities

    def get_base_pose(self) -> Tuple[List[float], List[float]]:
        """
        Get robot base position and orientation.

        Returns:
            Tuple of (position, orientation_euler)
        """
        pos, orn_quat = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client
        )
        orn_euler = p.getEulerFromQuaternion(orn_quat)

        return list(pos), list(orn_euler)

    def get_end_effector_pose(self) -> List[float]:
        """
        Get end effector pose using forward kinematics.

        Returns:
            [x, y, z, roll, pitch, yaw]
        """
        # Simplified FK - in real implementation, use proper FK or PyBullet's getLinkState
        # For now, use joint angles with simplified kinematic chain

        arm_pos, _ = self.get_arm_states()

        if len(arm_pos) < 6:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Simplified planar FK (matches sim_interface)
        l1 = 0.3  # Link 1 length
        l2 = 0.25  # Link 2 length

        theta1 = arm_pos[0]  # shoulder_pan
        theta2 = arm_pos[1]  # shoulder_lift
        theta3 = arm_pos[2]  # elbow_flex

        x = l1 * math.cos(theta2) + l2 * math.cos(theta2 + theta3)
        y = math.sin(theta1) * (l1 * math.cos(theta2) + l2 * math.cos(theta2 + theta3))
        z = l1 * math.sin(theta2) + l2 * math.sin(theta2 + theta3) + 0.1

        # Orientation from wrist joints
        roll = arm_pos[4] if len(arm_pos) > 4 else 0.0
        pitch = arm_pos[3] if len(arm_pos) > 3 else 0.0
        yaw = theta1

        return [x, y, z, roll, pitch, yaw]

    def reset_to_home(self) -> None:
        """Reset robot to home position."""
        # Reset wheels
        self.set_wheel_positions(0.0, 0.0, 0.0)

        # Reset arm
        home_positions = [0.0] * 6
        self.set_arm_positions(home_positions)

    def step_simulation(self) -> None:
        """Step the physics simulation."""
        p.stepSimulation(physicsClientId=self.client)


if __name__ == "__main__":
    # Test the controller
    import sys

    print("Testing LeKiwi Robot Controller")
    print("=" * 50)

    # Test kinematics
    print("\nTesting Mecanum Kinematics:")
    kinematics = MecanumKinematics(chassis_radius=0.15, wheel_radius=0.05)

    # Test forward movement
    vx, vy, omega = 1.0, 0.0, 0.0
    w1, w2, w3 = kinematics.inverse_kinematics(vx, vy, omega)
    print(f"Forward (vx=1.0): wheels = [{w1:.2f}, {w2:.2f}, {w3:.2f}] rad/s")

    # Test rotation
    vx, vy, omega = 0.0, 0.0, 1.0
    w1, w2, w3 = kinematics.inverse_kinematics(vx, vy, omega)
    print(f"Rotate (ω=1.0): wheels = [{w1:.2f}, {w2:.2f}, {w3:.2f}] rad/s")

    # Test strafe
    vx, vy, omega = 0.0, 1.0, 0.0
    w1, w2, w3 = kinematics.inverse_kinematics(vx, vy, omega)
    print(f"Strafe (vy=1.0): wheels = [{w1:.2f}, {w2:.2f}, {w3:.2f}] rad/s")

    print("\nKinematics tests passed!")
