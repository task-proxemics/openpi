"""Single SO-101 robotic arm policy transforms for OpenPI with RGB cameras only."""

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
class S0101SingleInputs(transforms.DataTransformFn):
    """
    Input transforms for single SO-101 robotic arm with RGB cameras only.
    
    The SO-101 has 6 degrees of freedom:
    - shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    
    Sensors:
    - 1x Wrist RGB camera (mapped to right_wrist_0_rgb for right arm tasks)
    - 1x Intel RealSense RGB camera (depth disabled)
    
    Note: For right arm tasks, the wrist camera is mapped to right_wrist_0_rgb
    and left_wrist_0_rgb is padded with zeros.
    """
    
    # Determines which model will be used
    model_type: _model.ModelType = _model.ModelType.PI05
    
    # Expected camera names for single SO-101 setup
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("wrist_camera", "realsense_rgb")
    
    def __call__(self, data: dict) -> dict:
        # Parse SO-101 joint positions (6 DOF)
        if "observation/joint_positions" in data:
            state = np.asarray(data["observation/joint_positions"])
        elif "state" in data:
            state = np.asarray(data["state"])
        else:
            # Default joint order for SO-101
            joint_keys = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", 
                         "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
            state = np.array([data.get(key, 0.0) for key in joint_keys])
        
        # Parse wrist RGB camera
        wrist_image = None
        wrist_keys = [
            "observation/wrist_camera",
            "wrist_camera",
            "wrist1",
            "observation/images/wrist1",
            "wrist2",
            "observation/images/wrist2"
        ]
        
        for key in wrist_keys:
            if key in data:
                wrist_image = _parse_image(data[key])
                break
                
        if wrist_image is None:
            wrist_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
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
                    "base_0_rgb": realsense_rgb,      # RealSense RGB only 
                    "left_wrist_0_rgb": np.zeros_like(realsense_rgb),  # Padded with zeros (no left arm)
                    "right_wrist_0_rgb": wrist_image,  # Right arm wrist camera
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.False_,  # Mask out the padded left camera
                    "right_wrist_0_rgb": np.True_,  # Use the right arm camera
                }
            case _model.ModelType.PI0_FAST:
                images = {
                    "base_0_rgb": realsense_rgb,  # RealSense RGB
                    "wrist_0_rgb": wrist_image,   # Wrist camera
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
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
class S0101SingleOutputs(transforms.DataTransformFn):
    """Output transforms for single SO-101 robotic arm."""
    
    def __call__(self, data: dict) -> dict:
        """
        Convert model actions to SO-101 joint commands.
        
        Args:
            data: Dictionary containing "actions" key with model actions
            
        Returns:
            Dictionary with joint commands for SO-101
        """
        actions = np.asarray(data["actions"])
        
        # Extract 6D S0101 actions from 32D model output
        s0101_actions = actions[:, :6]  # First 6 dimensions
        
        # Map actions to joint names
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                      "wrist_flex", "wrist_roll", "gripper"]
        
        joint_commands = {}
        for i, joint_name in enumerate(joint_names):
            joint_commands[f"{joint_name}.pos"] = s0101_actions[..., i]
        
        return {
            "joint_commands": joint_commands,
            "actions": s0101_actions,
        }