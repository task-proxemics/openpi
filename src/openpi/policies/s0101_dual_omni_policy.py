"""Dual SO-101 robotic arms + Omni base policy transforms for OpenPI with RGB cameras only."""

import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    if isinstance(image, np.ndarray) and image.dtype == np.float32 and image.ndim == 3 and image.shape[0] == 3:
        # Convert from (C,H,W) float32 to (H,W,C) uint8
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class S0101DualOmniInputs(transforms.DataTransformFn):
    """
    Input transforms for dual SO-101 robotic arms + Omni base with RGB cameras only.
    
    Two SO-101 arms with 6 degrees of freedom each + Omni base with 3 DOF:
    Left arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    Right arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    Omni base: x_velocity, y_velocity, angular_velocity
    Total: 15 DOF
    
    Sensors:
    - 1x Left wrist RGB camera  
    - 1x Right wrist RGB camera
    - 1x Intel RealSense RGB camera (depth disabled)
    """
    
    # Determines which model will be used
    model_type: _model.ModelType = _model.ModelType.PI05
    
    # Expected camera names for dual SO-101 + Omni setup
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist_camera", "right_wrist_camera", "realsense_rgb")
    
    def __call__(self, data: dict) -> dict:
        # Parse dual SO-101 + Omni joint positions (15 DOF total)
        if "observation/joint_positions" in data:
            state = np.asarray(data["observation/joint_positions"])
        elif "state" in data:
            state = np.asarray(data["state"])
        else:
            # Default joint order: left arm (6) + right arm (6) + omni base (3)
            left_joint_keys = ["left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos", 
                              "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos"]
            right_joint_keys = ["right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos", 
                               "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos"]
            omni_keys = ["base_x.vel", "base_y.vel", "base_angular.vel"]
            all_joint_keys = left_joint_keys + right_joint_keys + omni_keys
            state = np.array([data.get(key, 0.0) for key in all_joint_keys])
        
        # Ensure we have 15 DOF
        if len(state) != 15:
            raise ValueError(f"Expected 15 joint positions for dual SO-101 + Omni, got {len(state)}")
        
        # Parse left wrist RGB camera
        left_wrist_image = None
        left_wrist_keys = [
            "observation/left_wrist_camera",
            "left_wrist_camera",
            "left_wrist",
            "observation/images/left_wrist"
        ]
        
        for key in left_wrist_keys:
            if key in data:
                left_wrist_image = _parse_image(data[key])
                break
                
        if left_wrist_image is None:
            left_wrist_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Parse right wrist RGB camera
        right_wrist_image = None
        right_wrist_keys = [
            "observation/right_wrist_camera",
            "right_wrist_camera",
            "right_wrist",
            "observation/images/right_wrist"
        ]
        
        for key in right_wrist_keys:
            if key in data:
                right_wrist_image = _parse_image(data[key])
                break
                
        if right_wrist_image is None:
            right_wrist_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Parse Intel RealSense RGB (depth disabled)
        realsense_rgb = None
        realsense_keys = [
            "observation/realsense_rgb",
            "realsense_rgb", 
            "realsense",
            "observation/d455_color",
            "d455_color",
            "observation/color",
            "color",
            "observation/images/realsense"
        ]
        
        for key in realsense_keys:
            if key in data:
                realsense_rgb = _parse_image(data[key])
                break
        
        if realsense_rgb is None:
            realsense_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Create inputs dict matching π₀.₅ expected format (RGB only)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                images = {
                    "base_0_rgb": realsense_rgb,        # RealSense RGB only 
                    "left_wrist_0_rgb": left_wrist_image,   # Left wrist camera
                    "right_wrist_0_rgb": right_wrist_image, # Right wrist camera
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                }
            case _model.ModelType.PI0_FAST:
                images = {
                    "base_0_rgb": realsense_rgb,        # RealSense RGB
                    "wrist_0_rgb": left_wrist_image,    # Left wrist camera
                    "wrist_1_rgb": right_wrist_image,   # Right wrist camera  
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                    "wrist_1_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }
        
        # Add actions if available (for training)
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        
        # Add prompt if available
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class S0101DualOmniOutputs(transforms.DataTransformFn):
    """Output transforms for dual SO-101 robotic arms + Omni base."""
    
    def __call__(self, actions: np.ndarray) -> dict:
        """
        Convert model actions to dual SO-101 + Omni base commands.
        
        Args:
            actions: Model actions (15,) array [left_arm(6) + right_arm(6) + omni_base(3)]
            
        Returns:
            Dictionary with joint commands for dual SO-101 + Omni base
        """
        # Ensure we have 15 actions for dual SO-101 + Omni
        if actions.shape[-1] != 15:
            raise ValueError(f"Expected 15 actions for dual SO-101 + Omni, got {actions.shape[-1]}")
        
        # Split actions into components
        left_actions = actions[..., :6]
        right_actions = actions[..., 6:12]
        omni_actions = actions[..., 12:]
        
        # Map actions to joint names
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                      "wrist_flex", "wrist_roll", "gripper"]
        
        joint_commands = {}
        
        # Left arm commands
        for i, joint_name in enumerate(joint_names):
            joint_commands[f"left_{joint_name}.pos"] = left_actions[..., i]
        
        # Right arm commands
        for i, joint_name in enumerate(joint_names):
            joint_commands[f"right_{joint_name}.pos"] = right_actions[..., i]
        
        # Omni base commands (velocities)
        omni_command_names = ["base_x.vel", "base_y.vel", "base_angular.vel"]
        for i, cmd_name in enumerate(omni_command_names):
            joint_commands[cmd_name] = omni_actions[..., i]
        
        return {
            "joint_commands": joint_commands,
            "actions": actions,
            "left_arm_actions": left_actions,
            "right_arm_actions": right_actions,
            "omni_base_actions": omni_actions,
        }
