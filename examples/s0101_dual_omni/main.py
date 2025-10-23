"""Main testing script for mobile dual SO-101 arms with OpenPI π₀.₅."""

import argparse
import logging
from typing import Any, Dict

import numpy as np

import openpi.training.config as _config
import openpi.models.policy_config as _policy_config
from examples.s0101_dual_omni.constants import (
    S0101_DUAL_OMNI_JOINTS, 
    S0101_DUAL_OMNI_READY_POSE,
    split_mobile_dual_actions
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mobile_dual_s0101_observation(
    left_joint_positions: np.ndarray,
    right_joint_positions: np.ndarray,
    base_velocities: np.ndarray,
    left_wrist_image: np.ndarray,
    right_wrist_image: np.ndarray, 
    realsense_image: np.ndarray,
    prompt: str = "navigate to workspace and coordinate both arms for manipulation"
) -> Dict[str, Any]:
    """Create a dummy observation for mobile dual SO-101 arms."""
    # Combine all state: left arm + right arm + base velocities
    state = np.concatenate([left_joint_positions, right_joint_positions, base_velocities])
    
    return {
        "state": state,
        "left_wrist_camera": left_wrist_image,
        "right_wrist_camera": right_wrist_image,
        "realsense_rgb": realsense_image,
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Mobile Dual SO-101 + π₀.₅ Integration Example")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Path to π₀.₅ checkpoint directory"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_s0101_dual_omni",
        help="Configuration name to use"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="navigate to the table and pick objects with coordinated arm movements",
        help="Task instruction for the mobile robot"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of inference steps to run"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Mobile Dual SO-101 + π₀.₅ integration example")
    
    # Load configuration and create policy
    logger.info(f"Loading config: {args.config_name}")
    config = _config.get_config(args.config_name)
    
    logger.info(f"Creating policy from checkpoint: {args.checkpoint_dir}")
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    
    logger.info("Policy loaded successfully!")
    logger.info(f"Policy metadata: {policy.metadata}")
    
    # Create dummy observations for testing
    logger.info("Running mobile manipulation inference test...")
    
    for step in range(args.num_steps):
        logger.info(f"Step {step + 1}/{args.num_steps}")
        
        # Create dummy joint positions (start from ready pose)
        left_joints = np.array(S0101_DUAL_OMNI_READY_POSE[:6]) + np.random.normal(0, 5, 6)
        right_joints = np.array(S0101_DUAL_OMNI_READY_POSE[6:12]) + np.random.normal(0, 5, 6)
        
        # Create dummy base velocities (small random movements)
        base_velocities = np.random.normal(0, 0.1, 3)  # [x_vel, y_vel, ang_vel]
        
        # Create dummy RGB images (224x224x3)
        left_wrist_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        right_wrist_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        realsense_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Create observation
        observation = create_mobile_dual_s0101_observation(
            left_joints, right_joints, base_velocities,
            left_wrist_image, right_wrist_image, realsense_image,
            args.prompt
        )
        
        # Run inference
        try:
            actions = policy.act(observation)
            logger.info(f"Generated actions shape: {actions.shape}")
            
            # Split and display actions
            left_actions, right_actions, base_actions = split_mobile_dual_actions(actions)
            logger.info(f"Left arm actions: {left_actions}")
            logger.info(f"Right arm actions: {right_actions}")
            logger.info(f"Base actions: {base_actions}")
            
            # Map to joint names
            logger.info("=== Left Arm Commands ===")
            for i, joint_name in enumerate(S0101_DUAL_OMNI_JOINTS[:6]):
                logger.info(f"  {joint_name}: {left_actions[i]:.3f}°")
                
            logger.info("=== Right Arm Commands ===")
            for i, joint_name in enumerate(S0101_DUAL_OMNI_JOINTS[6:12]):
                logger.info(f"  {joint_name}: {right_actions[i]:.3f}°")
                
            logger.info("=== Omni Base Commands ===")
            for i, joint_name in enumerate(S0101_DUAL_OMNI_JOINTS[12:]):
                unit = "m/s" if "vel" in joint_name and "angular" not in joint_name else "rad/s"
                logger.info(f"  {joint_name}: {base_actions[i]:.3f} {unit}")
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            break
        
        logger.info("-" * 60)
    
    logger.info("Mobile Dual SO-101 testing completed successfully!")


if __name__ == "__main__":
    main()
