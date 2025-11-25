#!/home/loidinh/ws/robo-fleet-dora-rs/orchestra/pybullet_sim/venv/bin/python3
"""
PyBullet Simulation Node for dora-rs dataflow.

Integrates with dora-rs to:
- Process RoverCommand and ArmCommand
- Generate telemetry
- Event-driven architecture
- Full physics simulation with PyBullet
"""

import json
import sys
import time
import traceback
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data

# Import configuration and controller
from pybullet_config import get_config, SimulationConfig
from pybullet_robot_controller import LeKiwiRobotController


class PyBulletSimulationNode:
    """
    Main simulation node for PyBullet physics simulation.

    Manages the simulation lifecycle and dora-rs integration.
    """

    def __init__(self):
        """Initialize the simulation node."""
        print("PyBullet Simulation Node - Initializing", file=sys.stderr)

        # Load configuration
        self.config: Optional[SimulationConfig] = None
        self.physics_client: Optional[int] = None
        self.robot_controller: Optional[LeKiwiRobotController] = None

        # State tracking
        self.last_arm_command = None
        self.last_rover_command = None
        self.simulation_time = 0.0

    def setup(self) -> None:
        """Set up PyBullet simulation environment."""
        try:
            # Load configuration
            self.config = get_config()
            print(self.config, file=sys.stderr)

            # Connect to PyBullet
            if self.config.gui_enabled:
                self.physics_client = p.connect(p.GUI)
                print("PyBullet GUI enabled", file=sys.stderr)
            else:
                self.physics_client = p.connect(p.DIRECT)
                print("PyBullet headless mode", file=sys.stderr)

            # Configure camera (for GUI)
            if self.config.gui_enabled:
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.5,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 0],
                    physicsClientId=self.physics_client
                )

            # Set additional search path for PyBullet data
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Set gravity
            p.setGravity(0, 0, self.config.physics.gravity, physicsClientId=self.physics_client)

            # Set time step
            p.setTimeStep(self.config.physics.time_step, physicsClientId=self.physics_client)

            # Load ground plane
            ground_id = p.loadURDF(
                "plane.urdf",
                basePosition=[0, 0, 0],
                physicsClientId=self.physics_client
            )

            # Set ground friction
            p.changeDynamics(
                ground_id,
                -1,
                lateralFriction=self.config.physics.friction_coefficient,
                physicsClientId=self.physics_client
            )

            print("Ground plane loaded", file=sys.stderr)

            # Create robot controller
            self.robot_controller = LeKiwiRobotController(self.config, self.physics_client)

            # Load robot
            robot_id = self.robot_controller.load_robot()
            print(f"Robot loaded with ID: {robot_id}", file=sys.stderr)

            # Reset to home
            self.robot_controller.reset_to_home()

            # Step a few times to stabilize
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.physics_client)

            print("Simulation setup complete", file=sys.stderr)

        except Exception as e:
            print(f"Error during setup: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise

    def process_arm_command(self, command_json: dict) -> dict:
        """
        Process arm command and generate telemetry.

        Args:
            command_json: Arm command JSON (ArmCommandWithMetadata)

        Returns:
            Arm telemetry JSON
        """
        try:
            # Extract command
            command = command_json.get('command')
            if not command:
                return self.generate_arm_telemetry()

            command_type = list(command.keys())[0] if command else None

            if command_type == 'JointPosition':
                joint_angles = command['JointPosition'].get('joint_angles', [])
                if len(joint_angles) >= 6:
                    self.robot_controller.set_arm_positions(joint_angles)
                    print(f"Set arm positions: {joint_angles[:3]}...", file=sys.stderr)

            elif command_type == 'Home':
                self.robot_controller.set_arm_positions([0.0] * 6)
                print("Arm moving to home", file=sys.stderr)

            elif command_type in ['Stop', 'EmergencyStop']:
                # Keep current position
                print(f"Arm {command_type}", file=sys.stderr)

            # Step simulation
            for _ in range(10):  # Multiple steps for smooth motion
                self.robot_controller.step_simulation()
                if self.config.real_time_simulation:
                    time.sleep(self.config.physics.time_step)

            return self.generate_arm_telemetry()

        except Exception as e:
            print(f"Error processing arm command: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return self.generate_arm_telemetry()

    def process_rover_command(self, command_json: dict) -> dict:
        """
        Process rover command and generate telemetry.

        Args:
            command_json: Rover command JSON (RoverCommandWithMetadata)

        Returns:
            Rover telemetry JSON
        """
        try:
            # Extract command
            command = command_json.get('command')
            if not command:
                return self.generate_rover_telemetry()

            command_type = list(command.keys())[0] if command else None

            if command_type == 'JointPositions':
                data = command['JointPositions']
                wheel1 = data.get('wheel1', 0.0)
                wheel2 = data.get('wheel2', 0.0)
                wheel3 = data.get('wheel3', 0.0)

                self.robot_controller.set_wheel_positions(wheel1, wheel2, wheel3)
                print(f"Set wheel positions: [{wheel1:.2f}, {wheel2:.2f}, {wheel3:.2f}]", file=sys.stderr)

            elif command_type == 'Velocity':
                # Note: This is processed by rover-controller which converts to JointPositions
                # But we can handle it here for direct velocity control
                data = command['Velocity']
                vx = data.get('linear_x', 0.0)
                vy = data.get('linear_y', 0.0)
                omega = data.get('angular_z', 0.0)

                self.robot_controller.set_chassis_velocity(vx, vy, omega)
                print(f"Set chassis velocity: vx={vx:.2f}, vy={vy:.2f}, ω={omega:.2f}", file=sys.stderr)

            elif command_type == 'Stop':
                self.robot_controller.set_wheel_velocities(0.0, 0.0, 0.0)
                print("Rover stopped", file=sys.stderr)

            # Step simulation
            for _ in range(10):
                self.robot_controller.step_simulation()
                if self.config.real_time_simulation:
                    time.sleep(self.config.physics.time_step)

            return self.generate_rover_telemetry()

        except Exception as e:
            print(f"Error processing rover command: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return self.generate_rover_telemetry()

    def generate_arm_telemetry(self) -> dict:
        """
        Generate arm telemetry from simulation state.

        Returns:
            ArmTelemetry JSON
        """
        end_effector_pose = self.robot_controller.get_end_effector_pose()
        arm_positions, arm_velocities = self.robot_controller.get_arm_states()

        telemetry = {
            "entity_id": None,
            "end_effector_pose": end_effector_pose,
            "is_moving": any(abs(v) > 0.01 for v in arm_velocities),
            "timestamp": int(time.time() * 1000),
            "joint_angles": arm_positions if len(arm_positions) == 6 else [0.0] * 6,
            "joint_velocities": arm_velocities if len(arm_velocities) == 6 else [0.0] * 6,
            "source": "pybullet_simulation"
        }

        return telemetry

    def generate_rover_telemetry(self) -> dict:
        """
        Generate rover telemetry from simulation state.

        Returns:
            RoverTelemetry JSON
        """
        wheel_positions, wheel_velocities = self.robot_controller.get_wheel_states()
        base_pos, base_orn = self.robot_controller.get_base_pose()

        # Pad to 3 wheels if needed
        while len(wheel_positions) < 3:
            wheel_positions.append(0.0)
        while len(wheel_velocities) < 3:
            wheel_velocities.append(0.0)

        telemetry = {
            "entity_id": None,
            "timestamp": int(time.time() * 1000),
            "wheel_positions": wheel_positions[:3],
            "wheel_velocities": wheel_velocities[:3],
            "base_position": base_pos,
            "base_orientation": base_orn,
            "source": "pybullet_simulation"
        }

        return telemetry

    def cleanup(self) -> None:
        """Clean up simulation resources."""
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            print("PyBullet disconnected", file=sys.stderr)


def main():
    """Main entry point for dora-rs node."""
    print("=" * 60, file=sys.stderr)
    print("PyBullet Simulation Node for LeKiwi Robot", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    node = PyBulletSimulationNode()

    try:
        # Setup simulation
        node.setup()

        # Try to import dora
        try:
            from dora import Node
            use_dora = True
            print("Using dora-rs integration", file=sys.stderr)
        except ImportError:
            print("Warning: dora module not found, running in standalone mode", file=sys.stderr)
            print("Install with: pip install dora-rs", file=sys.stderr)
            use_dora = False

        if use_dora:
            # Dora-rs event loop
            dora_node = Node()

            print("Ready to receive commands", file=sys.stderr)

            for event in dora_node:
                event_type = event["type"]

                if event_type == "INPUT":
                    input_id = event["id"]
                    value = event["value"]

                    # Decode input data
                    try:
                        # Value should be bytes containing JSON
                        if isinstance(value, bytes):
                            data_json = json.loads(value.decode('utf-8'))
                        else:
                            # Try PyArrow array format
                            data_bytes = bytes(value[0])
                            data_json = json.loads(data_bytes.decode('utf-8'))

                        if input_id == "arm_command":
                            telemetry = node.process_arm_command(data_json)
                            telemetry_bytes = json.dumps(telemetry).encode('utf-8')
                            dora_node.send_output("arm_telemetry", telemetry_bytes)

                        elif input_id == "rover_command":
                            telemetry = node.process_rover_command(data_json)
                            telemetry_bytes = json.dumps(telemetry).encode('utf-8')
                            dora_node.send_output("rover_telemetry", telemetry_bytes)

                    except Exception as e:
                        print(f"Error processing input {input_id}: {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)

                elif event_type == "STOP":
                    print("Received stop event", file=sys.stderr)
                    break

        else:
            # Standalone mode - just keep simulation running
            print("\nStandalone mode - press Ctrl+C to exit", file=sys.stderr)
            print("Testing basic movements...\n", file=sys.stderr)

            # Test movements
            print("1. Moving arm to home position", file=sys.stderr)
            node.robot_controller.set_arm_positions([0.0] * 6)
            for _ in range(100):
                node.robot_controller.step_simulation()
                time.sleep(0.01)

            print("2. Moving arm to test position", file=sys.stderr)
            node.robot_controller.set_arm_positions([0.5, 0.3, -0.3, 0.0, 0.0, 0.5])
            for _ in range(100):
                node.robot_controller.step_simulation()
                time.sleep(0.01)

            print("3. Forward velocity", file=sys.stderr)
            node.robot_controller.set_chassis_velocity(0.5, 0.0, 0.0)
            for _ in range(200):
                node.robot_controller.step_simulation()
                time.sleep(0.01)

            print("4. Rotation", file=sys.stderr)
            node.robot_controller.set_chassis_velocity(0.0, 0.0, 1.0)
            for _ in range(200):
                node.robot_controller.step_simulation()
                time.sleep(0.01)

            print("5. Stop", file=sys.stderr)
            node.robot_controller.set_wheel_velocities(0.0, 0.0, 0.0)
            for _ in range(100):
                node.robot_controller.step_simulation()
                time.sleep(0.01)

            print("\nTest complete! Keeping simulation open...", file=sys.stderr)

            # Keep running
            if node.config.gui_enabled:
                try:
                    while True:
                        node.robot_controller.step_simulation()
                        time.sleep(0.01)
                except KeyboardInterrupt:
                    print("\nStopping...", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)

    except Exception as e:
        print(f"Error in main loop: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    finally:
        node.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
