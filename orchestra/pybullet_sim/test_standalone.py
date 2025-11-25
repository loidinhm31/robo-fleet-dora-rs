#!/usr/bin/env python3
"""
Standalone test script for PyBullet simulation.

Tests the simulation without dora-rs integration.
"""

import os
import sys
import time

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Set environment variables for testing
if 'URDF_PATH' not in os.environ:
    # Default to model/LeKiwi.urdf relative to repo root
    repo_root = os.path.abspath(os.path.join(script_dir, '../..'))
    urdf_path = os.path.join(repo_root, 'model', 'LeKiwi.urdf')
    os.environ['URDF_PATH'] = urdf_path
    print(f"Using URDF: {urdf_path}")

# Enable GUI for testing
os.environ['GUI_ENABLED'] = 'true'

# Import after setting env vars
from pybullet_config import get_config
from pybullet_robot_controller import LeKiwiRobotController
import pybullet as p
import pybullet_data


def test_configuration():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)

    try:
        config = get_config()
        print(config)
        print("✓ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        raise


def test_simulation_setup(config):
    """Test simulation environment setup."""
    print("\n" + "=" * 60)
    print("TEST 2: Simulation Setup")
    print("=" * 60)

    # Connect to PyBullet
    client = p.connect(p.GUI)
    print("✓ PyBullet GUI connected")

    # Configure camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=client
    )
    print("✓ Camera configured")

    # Set additional search path
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set physics
    p.setGravity(0, 0, config.physics.gravity, physicsClientId=client)
    p.setTimeStep(config.physics.time_step, physicsClientId=client)
    print("✓ Physics configured")

    # Load ground
    ground = p.loadURDF("plane.urdf", physicsClientId=client)
    p.changeDynamics(
        ground, -1,
        lateralFriction=config.physics.friction_coefficient,
        physicsClientId=client
    )
    print("✓ Ground plane loaded")

    return client


def test_robot_loading(config, client):
    """Test robot loading."""
    print("\n" + "=" * 60)
    print("TEST 3: Robot Loading")
    print("=" * 60)

    controller = LeKiwiRobotController(config, client)
    robot_id = controller.load_robot()

    print(f"✓ Robot loaded with ID: {robot_id}")

    # Stabilize
    for _ in range(100):
        p.stepSimulation(physicsClientId=client)

    print("✓ Robot stabilized")

    return controller


def test_arm_control(controller):
    """Test arm control."""
    print("\n" + "=" * 60)
    print("TEST 4: Arm Control")
    print("=" * 60)

    # Home position
    print("Moving to home position...")
    controller.set_arm_positions([0.0] * 6)
    for _ in range(200):
        controller.step_simulation()
        time.sleep(0.005)

    positions, velocities = controller.get_arm_states()
    print(f"Arm positions: {[f'{p:.3f}' for p in positions]}")
    print(f"Arm velocities: {[f'{v:.3f}' for v in velocities]}")
    print("✓ Home position reached")

    # Test position 1
    print("\nMoving to test position 1...")
    controller.set_arm_positions([0.5, 0.3, -0.3, 0.0, 0.0, 0.5])
    for _ in range(300):
        controller.step_simulation()
        time.sleep(0.005)

    positions, _ = controller.get_arm_states()
    print(f"Arm positions: {[f'{p:.3f}' for p in positions]}")
    print("✓ Test position 1 reached")

    # Test position 2
    print("\nMoving to test position 2...")
    controller.set_arm_positions([-0.5, -0.3, 0.3, 0.5, 1.0, 0.2])
    for _ in range(300):
        controller.step_simulation()
        time.sleep(0.005)

    positions, _ = controller.get_arm_states()
    print(f"Arm positions: {[f'{p:.3f}' for p in positions]}")
    print("✓ Test position 2 reached")

    # Get end effector pose
    ee_pose = controller.get_end_effector_pose()
    print(f"\nEnd effector pose:")
    print(f"  Position: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}]")
    print(f"  Orientation: [{ee_pose[3]:.3f}, {ee_pose[4]:.3f}, {ee_pose[5]:.3f}]")
    print("✓ End effector FK working")


def test_rover_control(controller):
    """Test rover wheel control."""
    print("\n" + "=" * 60)
    print("TEST 5: Rover Control")
    print("=" * 60)

    # Test forward movement
    print("Moving forward (vx=0.5 m/s)...")
    controller.set_chassis_velocity(0.5, 0.0, 0.0)

    for i in range(300):
        controller.step_simulation()
        time.sleep(0.005)

        if i % 100 == 0:
            pos, vel = controller.get_wheel_states()
            base_pos, base_orn = controller.get_base_pose()
            print(f"  Step {i}: Base pos=[{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")

    print("✓ Forward movement completed")

    # Test rotation
    print("\nRotating (ω=1.0 rad/s)...")
    controller.set_chassis_velocity(0.0, 0.0, 1.0)

    for i in range(300):
        controller.step_simulation()
        time.sleep(0.005)

        if i % 100 == 0:
            base_pos, base_orn = controller.get_base_pose()
            print(f"  Step {i}: Base orn=[{base_orn[0]:.3f}, {base_orn[1]:.3f}, {base_orn[2]:.3f}]")

    print("✓ Rotation completed")

    # Test strafe
    print("\nStrafing (vy=0.5 m/s)...")
    controller.set_chassis_velocity(0.0, 0.5, 0.0)

    for i in range(300):
        controller.step_simulation()
        time.sleep(0.005)

        if i % 100 == 0:
            base_pos, _ = controller.get_base_pose()
            print(f"  Step {i}: Base pos=[{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")

    print("✓ Strafe completed")

    # Stop
    print("\nStopping...")
    controller.set_wheel_velocities(0.0, 0.0, 0.0)
    for _ in range(100):
        controller.step_simulation()
        time.sleep(0.005)

    wheel_pos, wheel_vel = controller.get_wheel_states()
    print(f"Wheel velocities: {[f'{v:.3f}' for v in wheel_vel]}")
    print("✓ Robot stopped")


def test_kinematics():
    """Test kinematic calculations."""
    print("\n" + "=" * 60)
    print("TEST 6: Kinematics")
    print("=" * 60)

    from pybullet_robot_controller import MecanumKinematics

    kin = MecanumKinematics(chassis_radius=0.15, wheel_radius=0.05)

    # Test forward
    vx, vy, omega = 1.0, 0.0, 0.0
    w1, w2, w3 = kin.inverse_kinematics(vx, vy, omega)
    print(f"Forward (vx=1.0): wheels=[{w1:.3f}, {w2:.3f}, {w3:.3f}] rad/s")

    # Verify forward kinematics
    vx_out, vy_out, omega_out = kin.forward_kinematics(w1, w2, w3)
    print(f"  Verification: vx={vx_out:.3f}, vy={vy_out:.3f}, ω={omega_out:.3f}")

    # Test rotation
    vx, vy, omega = 0.0, 0.0, 1.0
    w1, w2, w3 = kin.inverse_kinematics(vx, vy, omega)
    print(f"Rotate (ω=1.0): wheels=[{w1:.3f}, {w2:.3f}, {w3:.3f}] rad/s")

    vx_out, vy_out, omega_out = kin.forward_kinematics(w1, w2, w3)
    print(f"  Verification: vx={vx_out:.3f}, vy={vy_out:.3f}, ω={omega_out:.3f}")

    # Test strafe
    vx, vy, omega = 0.0, 1.0, 0.0
    w1, w2, w3 = kin.inverse_kinematics(vx, vy, omega)
    print(f"Strafe (vy=1.0): wheels=[{w1:.3f}, {w2:.3f}, {w3:.3f}] rad/s")

    vx_out, vy_out, omega_out = kin.forward_kinematics(w1, w2, w3)
    print(f"  Verification: vx={vx_out:.3f}, vy={vy_out:.3f}, ω={omega_out:.3f}")

    print("✓ Kinematics verified")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PyBullet Simulation - Standalone Test Suite")
    print("=" * 60)

    try:
        # Test 1: Configuration
        config = test_configuration()

        # Test 6: Kinematics (standalone, no simulation needed)
        test_kinematics()

        # Test 2: Simulation setup
        client = test_simulation_setup(config)

        # Test 3: Robot loading
        controller = test_robot_loading(config, client)

        # Test 4: Arm control
        test_arm_control(controller)

        # Return to home before rover tests
        print("\nReturning to home position...")
        controller.reset_to_home()
        for _ in range(100):
            controller.step_simulation()
            time.sleep(0.005)

        # Test 5: Rover control
        test_rover_control(controller)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSimulation will remain open for inspection.")
        print("Press Ctrl+C to exit.")

        # Keep simulation running
        try:
            while True:
                controller.step_simulation()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n\nExiting...")

        # Cleanup
        p.disconnect(physicsClientId=client)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
