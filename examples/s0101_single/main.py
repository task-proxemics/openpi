"""
SO-101 robotic arm integration example with OpenPI π₀.₅ model.

This script demonstrates how to use the π₀.₅ model with the SO-101 robotic arm.

Example usage:
    python so101_example.py --checkpoint_dir "path/to/pi05_checkpoint"
"""

import argparse
import logging
import time
from typing import Dict, Any

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.policies import so101_policy
from openpi.training import config as _config

logger = logging.getLogger(__name__)


def create_so101_observation(
    joint_positions: np.ndarray,
    wrist_image: np.ndarray,
    external_image: np.ndarray,
    prompt: str = "pick up the object and place it in the container"
) -> Dict[str, Any]:
    """Create a properly formatted observation for SO-101."""
    return {
        "state": joint_positions.astype(np.float32),
        "wrist_camera": wrist_image,
        "external_camera": external_image,
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="SO-101 + π₀.₅ Integration Example")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Path to π₀.₅ checkpoint directory"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_so101",
        help="Configuration name to use"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="pick up the object and place it in the container",
        help="Task instruction for the robot"
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
    logger.info("Starting SO-101 + π₀.₅ integration example")
    
    # Load configuration and create policy
    logger.info(f"Loading config: {args.config_name}")
    config = _config.get_config(args.config_name)
    
    logger.info(f"Creating policy from checkpoint: {args.checkpoint_dir}")
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    
    logger.info("Policy loaded successfully!")
    logger.info(f"Policy metadata: {policy.metadata}")
    
    # Create dummy observations for testing
    logger.info("Running inference test...")
    
    for step in range(args.num_steps):
        # Create dummy joint positions (6 DOF for SO-101)
        joint_positions = np.random.uniform(-1.0, 1.0, 6)
        
        # Create dummy images (224x224x3 RGB)
        wrist_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        external_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Create observation
        observation = create_so101_observation(
            joint_positions=joint_positions,
            wrist_image=wrist_image,
            external_image=external_image,
            prompt=args.prompt
        )
        
        # Run inference
        start_time = time.time()
        result = policy.infer(observation)
        inference_time = time.time() - start_time
        
        # Extract actions
        actions = result["actions"]
        
        logger.info(f"Step {step + 1}/{args.num_steps}:")
        logger.info(f"  Input joint positions: {joint_positions}")
        logger.info(f"  Output actions: {actions}")
        logger.info(f"  Inference time: {inference_time:.3f}s")
        logger.info(f"  Policy timing: {result.get('policy_timing', {})}")
        
        # Validate output shape
        expected_shape = (config.model.action_horizon, 6)  # 6 DOF for SO-101
        if actions.shape != expected_shape:
            logger.warning(f"Unexpected action shape: {actions.shape}, expected: {expected_shape}")
        
        time.sleep(0.1)  # Small delay between steps
    
    logger.info("Integration test completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Connect to real SO-101 hardware using LeRobot")
    logger.info("2. Replace dummy observations with real camera feeds and joint positions")
    logger.info("3. Send predicted actions to SO-101 motors via Feetech protocol")


if __name__ == "__main__":
    main()
