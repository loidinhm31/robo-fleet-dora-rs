#!/usr/bin/env python3
"""Quick test to verify joint mapping."""

import os
os.environ['URDF_PATH'] = '/home/loidinh/ws/robo-fleet-dora-rs/model/LeKiwi.urdf'
os.environ['GUI_ENABLED'] = 'false'

import pybullet as p
from pybullet_config import get_config
from pybullet_robot_controller import LeKiwiRobotController

print("Loading configuration...")
config = get_config()

print("Connecting to PyBullet...")
client = p.connect(p.DIRECT)

print("Creating controller...")
controller = LeKiwiRobotController(config, client)

print("Loading robot...")
robot_id = controller.load_robot()

print(f"\n✓ Robot loaded: ID={robot_id}")
print(f"✓ Wheel joints mapped: {len(controller.wheel_joints)}/3")
print(f"✓ Arm joints mapped: {len(controller.arm_joints)}/6")

if len(controller.wheel_joints) == 3 and len(controller.arm_joints) == 6:
    print("\n SUCCESS! All joints mapped correctly!")
else:
    print("\n  WARNING: Joint mapping incomplete")

p.disconnect(physicsClientId=client)
