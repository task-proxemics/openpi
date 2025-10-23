"""Dual SO-101 robotic arms + Omni base constants and utility functions."""

import numpy as np
from typing import Dict, List, Tuple

# Mobile Dual SO-101 Hardware Specifications  
S0101_DUAL_OMNI_DOF = 15  # 6 DOF per arm + 3 DOF omni base
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

S0101_OMNI_BASE = [
    "base_x.vel",
    "base_y.vel", 
    "base_angular.vel"
]

S0101_DUAL_OMNI_JOINTS = S0101_LEFT_JOINTS + S0101_RIGHT_JOINTS + S0101_OMNI_BASE

# Joint limits for arms (degrees) and base (m/s, rad/s)
S0101_DUAL_OMNI_JOINT_LIMITS = {
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
    # Omni base (velocities)
    "base_x.vel": (-1.0, 1.0),  # m/s
    "base_y.vel": (-1.0, 1.0),  # m/s
    "base_angular.vel": (-2.0, 2.0),  # rad/s
}

# Default poses for mobile dual arms
S0101_DUAL_OMNI_NEUTRAL_POSE = [0.0] * 15  # All joints/velocities at 0
S0101_DUAL_OMNI_HOME_POSE = [
    # Left arm home position  
    0.0, -30.0, 45.0, -15.0, 0.0, 0.0,
    # Right arm home position
    0.0, -30.0, 45.0, -15.0, 0.0, 0.0,
    # Base stationary
    0.0, 0.0, 0.0
]
S0101_DUAL_OMNI_READY_POSE = [
    # Left arm ready position
    -45.0, -45.0, 90.0, -45.0, 0.0, 50.0,
    # Right arm ready position  
    45.0, -45.0, 90.0, -45.0, 0.0, 50.0,
    # Base stationary
    0.0, 0.0, 0.0
]

# Camera specifications for mobile dual setup
MOBILE_DUAL_CAMERA_SPECS = {
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
        "position": "fixed_on_base",
        "purpose": "navigation_and_workspace_overview"
    }
}

# Omni base specifications
OMNI_BASE_SPECS = {
    "wheel_count": 3,
    "wheel_radius": 0.05,  # meters
    "base_radius": 0.25,   # meters
    "max_linear_velocity": 1.0,  # m/s
    "max_angular_velocity": 2.0,  # rad/s
    "weight_capacity": 50.0,  # kg (including arms)
}

# Safety limits for mobile dual arms
MOBILE_DUAL_SAFETY_LIMITS = {
    "max_joint_velocity": 50.0,  # degrees/second
    "max_joint_acceleration": 100.0,  # degrees/second^2
    "max_base_velocity": 0.5,  # m/s (conservative)
    "max_base_acceleration": 0.2,  # m/s^2
    "emergency_stop_distance": 0.2,  # meters
    "collision_threshold": 0.05,  # meters between arms
    "navigation_clearance": 0.3,  # meters around base
    "workspace_boundaries": {
        "x": (-5.0, 5.0),  # meters (expanded due to mobility)
        "y": (-5.0, 5.0),  # meters  
        "z": (0.0, 2.0),   # meters
    }
}

# Mobile manipulation modes
class MobileManipulationMode:
    STATIC_MANIPULATION = "static_manipulation"      # Base stops, arms manipulate
    MOBILE_MANIPULATION = "mobile_manipulation"      # Base moves while arms work
    NAVIGATE_THEN_MANIPULATE = "navigate_then_manipulate"  # Sequential
    COORDINATED_MOBILE = "coordinated_mobile"        # Simultaneous base + arms

# Navigation modes  
class NavigationMode:
    POINT_TO_POINT = "point_to_point"    # Navigate to specific position
    CONTINUOUS = "continuous"             # Continuous movement during task
    ORBIT = "orbit"                      # Circle around workspace
    APPROACH = "approach"                # Approach objects for manipulation

def degrees_to_radians(degrees: np.ndarray) -> np.ndarray:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)

def radians_to_degrees(radians: np.ndarray) -> np.ndarray:
    """Convert radians to degrees.""" 
    return np.rad2deg(radians)

def split_mobile_dual_actions(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split mobile dual arm actions into left arm, right arm, and base actions."""
    if actions.shape[-1] != 15:
        raise ValueError(f"Expected 15 actions for mobile dual arms, got {actions.shape[-1]}")
    return actions[..., :6], actions[..., 6:12], actions[..., 12:]

def combine_mobile_dual_actions(left_actions: np.ndarray, right_actions: np.ndarray, 
                               base_actions: np.ndarray) -> np.ndarray:
    """Combine left arm, right arm, and base actions into mobile dual arm actions."""
    if left_actions.shape[-1] != 6 or right_actions.shape[-1] != 6 or base_actions.shape[-1] != 3:
        raise ValueError("Expected 6 actions per arm and 3 for base")
    return np.concatenate([left_actions, right_actions, base_actions], axis=-1)

def check_mobile_arm_collision(left_state: np.ndarray, right_state: np.ndarray, 
                              base_state: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Enhanced collision check for mobile dual arms including base movement.
    
    Args:
        left_state: Left arm joint positions (6,)
        right_state: Right arm joint positions (6,)
        base_state: Base velocities (3,) [x_vel, y_vel, ang_vel]
        threshold: Distance threshold for collision detection
        
    Returns:
        True if potential collision detected
    """
    # Check arm-arm collision (same as dual arm setup)
    left_pan = left_state[0]  # left_shoulder_pan
    right_pan = right_state[0]  # right_shoulder_pan
    
    # If both arms are pointing towards center, potential collision
    if left_pan > -30 and right_pan < 30 and abs(left_pan - right_pan) < 60:
        return True
    
    # Check if base is moving too fast while arms are extended
    base_speed = np.linalg.norm(base_state[:2])  # x, y velocity magnitude
    arm_extension = max(abs(left_state[2]), abs(right_state[2]))  # elbow extension
    
    # Reduce max base speed when arms are extended
    if base_speed > 0.3 and arm_extension > 45:  # Arms extended and base moving fast
        return True
    
    return False

def calculate_workspace_position(base_position: np.ndarray, arm_config: np.ndarray) -> np.ndarray:
    """
    Calculate end-effector position in global coordinates for mobile base.
    
    Args:
        base_position: Base position [x, y, theta] in global frame
        arm_config: Arm joint positions (6,)
        
    Returns:
        End-effector position in global coordinates
    """
    # Simplified forward kinematics for mobile base + arm
    # In practice, would use full kinematic chain
    
    base_x, base_y, base_theta = base_position
    
    # Simple approximation: arm reach based on joint configuration
    arm_reach = 0.5  # meters (approximate)
    arm_angle = arm_config[0] + base_theta  # shoulder_pan + base orientation
    
    # End-effector position relative to base
    ee_x = base_x + arm_reach * np.cos(np.radians(arm_angle))
    ee_y = base_y + arm_reach * np.sin(np.radians(arm_angle))
    ee_z = 0.5  # Approximate height
    
    return np.array([ee_x, ee_y, ee_z])
