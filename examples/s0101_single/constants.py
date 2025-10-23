"""Single SO-101 robotic arm constants and utility functions."""

import numpy as np
from typing import Dict, List, Tuple

# SO-101 Hardware Specifications
SO101_DOF = 6
SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift", 
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

# Joint limits in degrees (approximate)
SO101_JOINT_LIMITS = {
    "shoulder_pan": (-180.0, 180.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-135.0, 135.0),
    "wrist_flex": (-90.0, 90.0),
    "wrist_roll": (-180.0, 180.0),
    "gripper": (0.0, 100.0),  # Percentage opening
}

# Default poses
SO101_NEUTRAL_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SO101_HOME_POSE = [0.0, -30.0, 45.0, -15.0, 0.0, 0.0]
SO101_READY_POSE = [0.0, -45.0, 90.0, -45.0, 0.0, 50.0]

# Camera specifications
CAMERA_SPECS = {
    "wrist": {
        "resolution": (640, 480),
        "fps": 30,
        "position": "end_effector",
        "purpose": "close_manipulation"
    },
    "external": {
        "resolution": (640, 480), 
        "fps": 30,
        "position": "fixed_overhead",
        "purpose": "scene_overview"
    },
    "realsense_d455": {
        "resolution": (640, 480),
        "fps": 30,
        "depth_range": (0.3, 5.0),  # meters
        "baseline": 95,  # mm
        "position": "flexible",
        "purpose": "depth_perception"
    }
}

# Safety limits
SAFETY_LIMITS = {
    "max_joint_velocity": 50.0,  # degrees/second
    "max_joint_acceleration": 100.0,  # degrees/second^2
    "emergency_stop_distance": 0.1,  # meters
    "collision_threshold": 0.05,  # meters
}


def degrees_to_radians(degrees: np.ndarray) -> np.ndarray:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)


def radians_to_degrees(radians: np.ndarray) -> np.ndarray:
    """Convert radians to degrees."""
    return np.rad2deg(radians)


def validate_joint_positions(positions: np.ndarray) -> bool:
    """
    Validate that joint positions are within safe limits.
    
    Args:
        positions: Joint positions in degrees
        
    Returns:
        True if all positions are within limits
    """
    if len(positions) != SO101_DOF:
        return False
    
    for i, (joint_name, pos) in enumerate(zip(SO101_JOINTS, positions)):
        min_limit, max_limit = SO101_JOINT_LIMITS[joint_name]
        if not (min_limit <= pos <= max_limit):
            return False
    
    return True


def clamp_joint_positions(positions: np.ndarray) -> np.ndarray:
    """
    Clamp joint positions to safe limits.
    
    Args:
        positions: Joint positions in degrees
        
    Returns:
        Clamped joint positions
    """
    clamped = positions.copy()
    
    for i, joint_name in enumerate(SO101_JOINTS):
        min_limit, max_limit = SO101_JOINT_LIMITS[joint_name]
        clamped[i] = np.clip(clamped[i], min_limit, max_limit)
    
    return clamped


def create_joint_dict(positions: np.ndarray) -> Dict[str, float]:
    """
    Create joint dictionary from position array.
    
    Args:
        positions: Joint positions array (6 DOF)
        
    Returns:
        Dictionary mapping joint names to positions
    """
    return {
        f"{joint}.pos": float(pos) 
        for joint, pos in zip(SO101_JOINTS, positions)
    }


def extract_joint_positions(joint_dict: Dict[str, float]) -> np.ndarray:
    """
    Extract joint positions from dictionary.
    
    Args:
        joint_dict: Dictionary with joint positions
        
    Returns:
        Joint positions array (6 DOF)
    """
    positions = []
    for joint in SO101_JOINTS:
        key = f"{joint}.pos"
        positions.append(joint_dict.get(key, 0.0))
    
    return np.array(positions, dtype=np.float32)


def interpolate_trajectory(
    start_pos: np.ndarray, 
    end_pos: np.ndarray, 
    num_steps: int = 50
) -> np.ndarray:
    """
    Generate smooth trajectory between two positions.
    
    Args:
        start_pos: Starting joint positions
        end_pos: Ending joint positions
        num_steps: Number of interpolation steps
        
    Returns:
        Trajectory array (num_steps, 6)
    """
    trajectory = np.linspace(start_pos, end_pos, num_steps)
    
    # Validate each step
    for i, pos in enumerate(trajectory):
        trajectory[i] = clamp_joint_positions(pos)
    
    return trajectory


def compute_joint_velocities(
    positions: np.ndarray, 
    dt: float = 0.1
) -> np.ndarray:
    """
    Compute joint velocities from position trajectory.
    
    Args:
        positions: Position trajectory (N, 6)
        dt: Time step
        
    Returns:
        Velocity trajectory (N-1, 6)
    """
    if len(positions) < 2:
        return np.zeros((0, SO101_DOF))
    
    velocities = np.diff(positions, axis=0) / dt
    return velocities


def check_collision_risk(
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    obstacle_positions: List[np.ndarray] = None
) -> bool:
    """
    Check if trajectory has collision risk.
    
    Args:
        current_pos: Current joint positions
        target_pos: Target joint positions
        obstacle_positions: List of obstacle positions
        
    Returns:
        True if collision risk detected
    """
    # Simple collision check based on joint limits
    trajectory = interpolate_trajectory(current_pos, target_pos, 10)
    
    for pos in trajectory:
        if not validate_joint_positions(pos):
            return True
    
    # TODO: Add more sophisticated collision detection
    # with obstacle_positions if provided
    
    return False


def get_camera_transform_matrix(camera_name: str) -> np.ndarray:
    """
    Get camera transformation matrix relative to robot base.
    
    Args:
        camera_name: Name of the camera
        
    Returns:
        4x4 transformation matrix
    """
    # Placeholder transformation matrices
    # These should be calibrated for actual hardware setup
    
    transforms = {
        "wrist": np.array([
            [1, 0, 0, 0.0],
            [0, 1, 0, 0.0], 
            [0, 0, 1, 0.15],  # 15cm above end effector
            [0, 0, 0, 1]
        ]),
        "external": np.array([
            [1, 0, 0, 0.5],   # 50cm in front
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.8],   # 80cm above base
            [0, 0, 0, 1]
        ]),
        "realsense_d455": np.array([
            [1, 0, 0, 0.3],   # 30cm in front
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.6],   # 60cm above base
            [0, 0, 0, 1]
        ])
    }
    
    return transforms.get(camera_name, np.eye(4))


# Predefined useful poses
PREDEFINED_POSES = {
    "neutral": SO101_NEUTRAL_POSE,
    "home": SO101_HOME_POSE,
    "ready": SO101_READY_POSE,
    "observe": [0.0, -60.0, 60.0, 0.0, 0.0, 0.0],
    "grasp_ready": [0.0, -45.0, 90.0, -45.0, 0.0, 80.0],
    "place_ready": [0.0, -30.0, 60.0, -30.0, 0.0, 0.0],
}
