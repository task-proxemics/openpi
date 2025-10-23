"""Dual SO-101 robotic arms constants and utility functions."""

import numpy as np
from typing import Dict, List, Tuple

# Dual SO-101 Hardware Specifications  
S0101_DUAL_DOF = 12  # 6 DOF per arm
S0101_LEFT_JOINTS = [
    "left_shoulder_pan",
    "left_shoulder_lift", 
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper"
]

S0101_RIGHT_JOINTS = [
    "right_shoulder_pan", 
    "right_shoulder_lift",
    "right_elbow_flex", 
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper"
]

S0101_DUAL_JOINTS = S0101_LEFT_JOINTS + S0101_RIGHT_JOINTS

# Joint limits in degrees (approximate)
S0101_DUAL_JOINT_LIMITS = {
    # Left arm
    "left_shoulder_pan": (-180.0, 180.0),
    "left_shoulder_lift": (-90.0, 90.0),
    "left_elbow_flex": (-135.0, 135.0),
    "left_wrist_flex": (-90.0, 90.0),
    "left_wrist_roll": (-180.0, 180.0),
    "left_gripper": (0.0, 100.0),  # Percentage opening
    # Right arm  
    "right_shoulder_pan": (-180.0, 180.0),
    "right_shoulder_lift": (-90.0, 90.0),
    "right_elbow_flex": (-135.0, 135.0),
    "right_wrist_flex": (-90.0, 90.0),
    "right_wrist_roll": (-180.0, 180.0),
    "right_gripper": (0.0, 100.0),  # Percentage opening
}

# Default poses for dual arms
S0101_DUAL_NEUTRAL_POSE = [0.0] * 12  # All joints at 0
S0101_DUAL_HOME_POSE = [
    # Left arm home position  
    0.0, -30.0, 45.0, -15.0, 0.0, 0.0,
    # Right arm home position
    0.0, -30.0, 45.0, -15.0, 0.0, 0.0
]
S0101_DUAL_READY_POSE = [
    # Left arm ready position
    -45.0, -45.0, 90.0, -45.0, 0.0, 50.0,
    # Right arm ready position  
    45.0, -45.0, 90.0, -45.0, 0.0, 50.0
]

# Camera specifications for dual setup
DUAL_CAMERA_SPECS = {
    "left_wrist": {
        "resolution": (640, 480),
        "fps": 30,
        "position": "left_end_effector", 
        "purpose": "left_arm_manipulation"
    },
    "right_wrist": {
        "resolution": (640, 480),
        "fps": 30,
        "position": "right_end_effector",
        "purpose": "right_arm_manipulation"
    },
    "realsense_rgb": {
        "resolution": (640, 480),
        "fps": 30,
        "depth_range": None,  # RGB only
        "position": "fixed_overhead",
        "purpose": "workspace_overview"
    }
}

# Safety limits for dual arms
DUAL_SAFETY_LIMITS = {
    "max_joint_velocity": 50.0,  # degrees/second
    "max_joint_acceleration": 100.0,  # degrees/second^2
    "emergency_stop_distance": 0.1,  # meters
    "collision_threshold": 0.05,  # meters between arms
    "workspace_boundaries": {
        "x": (-1.0, 1.0),  # meters
        "y": (-1.0, 1.0),  # meters  
        "z": (0.0, 1.5),   # meters
    }
}

# Arm coordination modes
class ArmCoordinationMode:
    INDEPENDENT = "independent"    # Arms work independently
    COORDINATED = "coordinated"    # Arms coordinate movements  
    MIRRORED = "mirrored"         # Right arm mirrors left arm
    HANDOFF = "handoff"           # Object passing between arms

def degrees_to_radians(degrees: np.ndarray) -> np.ndarray:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)

def radians_to_degrees(radians: np.ndarray) -> np.ndarray:
    """Convert radians to degrees.""" 
    return np.rad2deg(radians)

def split_dual_actions(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split dual arm actions into left and right arm actions."""
    if actions.shape[-1] != 12:
        raise ValueError(f"Expected 12 actions for dual arms, got {actions.shape[-1]}")
    return actions[..., :6], actions[..., 6:]

def combine_dual_actions(left_actions: np.ndarray, right_actions: np.ndarray) -> np.ndarray:
    """Combine left and right arm actions into dual arm actions."""
    if left_actions.shape[-1] != 6 or right_actions.shape[-1] != 6:
        raise ValueError("Expected 6 actions per arm")
    return np.concatenate([left_actions, right_actions], axis=-1)

def check_arm_collision(left_state: np.ndarray, right_state: np.ndarray, 
                       threshold: float = 0.1) -> bool:
    """
    Simple collision check between arms based on joint positions.
    
    Args:
        left_state: Left arm joint positions (6,)
        right_state: Right arm joint positions (6,)
        threshold: Distance threshold for collision detection
        
    Returns:
        True if potential collision detected
    """
    # Simplified collision check - in practice would use forward kinematics
    # Check if arms are too close based on shoulder pan angles
    left_pan = left_state[0]  # left_shoulder_pan
    right_pan = right_state[0]  # right_shoulder_pan
    
    # If both arms are pointing towards center, potential collision
    if left_pan > -30 and right_pan < 30 and abs(left_pan - right_pan) < 60:
        return True
    
    return False
