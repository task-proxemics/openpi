"""
SO-101 Hardware Integration Bridge with Intel RealSense D455

This module provides the bridge between OpenPI π₀.₅ model and SO-101 hardware
using the LeRobot framework with Intel RealSense D455 depth camera support.

Requirements:
- LeRobot installed with Feetech support: pip install -e ".[feetech]"
- Intel RealSense SDK 2.0: pip install pyrealsense2
- SO-101 hardware properly assembled and calibrated
- Cameras connected and configured (2x RGB + 1x RealSense D455)
"""

import logging
import time
from typing import Dict, Any, Optional

import cv2
import numpy as np

try:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    SO101Follower = None
    SO101FollowerConfig = None
    OpenCVCamera = None
    OpenCVCameraConfig = None

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)


class RealSenseD455:
    """Intel RealSense D455 camera interface."""
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        enable_color: bool = True,
    ):
        """
        Initialize RealSense D455 camera.
        
        Args:
            width: Image width
            height: Image height
            fps: Frame rate
            enable_depth: Enable depth stream
            enable_color: Enable color stream
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not installed. Install with: pip install pyrealsense2")
        
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if enable_color:
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Create alignment object (align depth to color)
        if enable_depth and enable_color:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None
        
        self.is_connected = False
    
    def connect(self):
        """Start the RealSense pipeline."""
        try:
            self.pipeline.start(self.config)
            self.is_connected = True
            logger.info("RealSense D455 connected successfully")
            
            # Wait for auto-exposure to stabilize
            for _ in range(30):
                self.pipeline.wait_for_frames()
                
        except Exception as e:
            logger.error(f"Failed to connect RealSense D455: {e}")
            raise
    
    def capture(self) -> Dict[str, np.ndarray]:
        """
        Capture RGB and depth frames.
        
        Returns:
            Dictionary with 'color' and 'depth' arrays
        """
        if not self.is_connected:
            raise RuntimeError("RealSense not connected")
        
        frames = self.pipeline.wait_for_frames()
        
        result = {}
        
        if self.align:
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            if self.enable_color:
                color_frame = aligned_frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    # Convert BGR to RGB
                    result['color'] = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            if self.enable_depth:
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    # Apply spatial filter to reduce noise
                    spatial = rs.spatial_filter()
                    depth_frame = spatial.process(depth_frame)
                    
                    depth_image = np.asanyarray(depth_frame.get_data())
                    result['depth'] = depth_image
        else:
            # No alignment
            if self.enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    result['color'] = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    result['depth'] = depth_image
        
        return result
    
    def disconnect(self):
        """Stop the RealSense pipeline."""
        if self.is_connected:
            self.pipeline.stop()
            self.is_connected = False
            logger.info("RealSense D455 disconnected")


class SO101PolicyBridge:
    """Bridge between OpenPI π₀.₅ policy and SO-101 hardware with RealSense D455."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        config_name: str = "pi05_so101",
        robot_port: str = "/dev/ttyACM0",
        wrist_camera_index: int = 0,
        external_camera_index: int = 1,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        use_realsense: bool = True,
        realsense_width: int = 640,
        realsense_height: int = 480,
        realsense_fps: int = 30,
    ):
        """
        Initialize the SO-101 policy bridge with RealSense D455.
        
        Args:
            checkpoint_dir: Path to π₀.₅ checkpoint
            config_name: OpenPI configuration name
            robot_port: Serial port for SO-101 communication
            wrist_camera_index: Camera index for wrist camera
            external_camera_index: Camera index for external camera
            camera_width: Regular camera image width
            camera_height: Regular camera image height
            camera_fps: Regular camera frame rate
            use_realsense: Whether to use RealSense D455
            realsense_width: RealSense image width
            realsense_height: RealSense image height
            realsense_fps: RealSense frame rate
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "LeRobot is not installed. Please install with: pip install -e '.[feetech]'"
            )
        
        self.checkpoint_dir = checkpoint_dir
        self.config_name = config_name
        self.use_realsense = use_realsense
        
        # Initialize OpenPI policy
        logger.info(f"Loading OpenPI policy: {config_name}")
        self.openpi_config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(
            self.openpi_config, checkpoint_dir
        )
        logger.info("OpenPI policy loaded successfully")
        
        # Initialize SO-101 robot
        logger.info(f"Initializing SO-101 robot on port: {robot_port}")
        self.robot_config = SO101FollowerConfig(
            port=robot_port,
            cameras={
                "wrist": OpenCVCameraConfig(
                    index_or_path=wrist_camera_index,
                    width=camera_width,
                    height=camera_height,
                    fps=camera_fps,
                ),
                "external": OpenCVCameraConfig(
                    index_or_path=external_camera_index,
                    width=camera_width,
                    height=camera_height,
                    fps=camera_fps,
                ),
            }
        )
        self.robot = SO101Follower(self.robot_config)
        
        # Initialize RealSense D455 if requested
        self.realsense = None
        if use_realsense:
            logger.info("Initializing Intel RealSense D455...")
            self.realsense = RealSenseD455(
                width=realsense_width,
                height=realsense_height,
                fps=realsense_fps,
                enable_depth=True,
                enable_color=True,
            )
            self.realsense.connect()
        
        # Connect and calibrate robot
        logger.info("Connecting to SO-101 robot...")
        self.robot.connect(calibrate=True)
        logger.info("SO-101 robot connected and calibrated")
        
        # Reset to neutral pose
        self.reset_robot()
    
    def reset_robot(self):
        """Reset robot to neutral pose."""
        reset_pose = self.policy.metadata.get("reset_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        logger.info(f"Resetting robot to pose: {reset_pose}")
        
        # Create action dict for SO-101
        action_dict = {
            "shoulder_pan.pos": reset_pose[0],
            "shoulder_lift.pos": reset_pose[1],
            "elbow_flex.pos": reset_pose[2],
            "wrist_flex.pos": reset_pose[3],
            "wrist_roll.pos": reset_pose[4],
            "gripper.pos": reset_pose[5],
        }
        
        self.robot.send_action(action_dict)
        time.sleep(2.0)  # Wait for robot to reach position
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from SO-101 sensors including RealSense D455."""
        # Get joint positions
        joint_state = self.robot.capture_observation()
        joint_positions = np.array([
            joint_state["shoulder_pan.pos"],
            joint_state["shoulder_lift.pos"],
            joint_state["elbow_flex.pos"],
            joint_state["wrist_flex.pos"],
            joint_state["wrist_roll.pos"],
            joint_state["gripper.pos"],
        ], dtype=np.float32)
        
        # Get regular camera images
        wrist_image = joint_state["wrist"]  # Shape: (H, W, 3)
        external_image = joint_state["external"]  # Shape: (H, W, 3)
        
        # Resize images to expected size (224x224)
        wrist_image = cv2.resize(wrist_image, (224, 224))
        external_image = cv2.resize(external_image, (224, 224))
        
        observation = {
            "state": joint_positions,
            "wrist_camera": wrist_image,
            "external_camera": external_image,
        }
        
        # Get RealSense D455 data if available
        if self.realsense and self.realsense.is_connected:
            try:
                realsense_data = self.realsense.capture()
                
                if 'color' in realsense_data:
                    realsense_rgb = cv2.resize(realsense_data['color'], (224, 224))
                    observation["realsense_rgb"] = realsense_rgb
                
                if 'depth' in realsense_data:
                    realsense_depth = cv2.resize(realsense_data['depth'], (224, 224))
                    # Convert depth from uint16 mm to float32 meters
                    realsense_depth = realsense_depth.astype(np.float32) / 1000.0
                    observation["realsense_depth"] = realsense_depth
                    
            except Exception as e:
                logger.warning(f"Failed to capture RealSense data: {e}")
                # Provide dummy data if RealSense fails
                observation["realsense_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)
                observation["realsense_depth"] = np.zeros((224, 224), dtype=np.float32)
        else:
            # Provide dummy RealSense data if not available
            observation["realsense_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)
            observation["realsense_depth"] = np.zeros((224, 224), dtype=np.float32)
        
        return observation
    
    def predict_action(self, prompt: str) -> np.ndarray:
        """
        Predict action using π₀.₅ model.
        
        Args:
            prompt: Task instruction
            
        Returns:
            Predicted actions for SO-101 (shape: [action_horizon, 6])
        """
        # Get current observation
        observation = self.get_observation()
        observation["prompt"] = prompt
        
        # Run inference
        start_time = time.time()
        result = self.policy.infer(observation)
        inference_time = time.time() - start_time
        
        actions = result["actions"]
        
        logger.info(f"Inference completed in {inference_time:.3f}s")
        logger.info(f"Predicted actions shape: {actions.shape}")
        
        return actions
    
    def execute_action_sequence(
        self, 
        actions: np.ndarray, 
        execution_hz: float = 10.0,
        safety_check: bool = True
    ):
        """
        Execute a sequence of actions on SO-101.
        
        Args:
            actions: Action sequence (shape: [action_horizon, 6])
            execution_hz: Execution frequency in Hz
            safety_check: Whether to perform safety checks
        """
        if actions.ndim != 2 or actions.shape[1] != 6:
            raise ValueError(f"Expected actions shape [N, 6], got {actions.shape}")
        
        dt = 1.0 / execution_hz
        
        logger.info(f"Executing {len(actions)} actions at {execution_hz} Hz")
        
        for i, action in enumerate(actions):
            if safety_check:
                # Basic safety checks
                if np.any(np.abs(action) > 2.0):  # Reasonable joint limit check
                    logger.warning(f"Action {i} has large values: {action}")
                    logger.warning("Skipping potentially unsafe action")
                    continue
            
            # Convert to SO-101 action format
            action_dict = {
                "shoulder_pan.pos": float(action[0]),
                "shoulder_lift.pos": float(action[1]),
                "elbow_flex.pos": float(action[2]),
                "wrist_flex.pos": float(action[3]),
                "wrist_roll.pos": float(action[4]),
                "gripper.pos": float(action[5]),
            }
            
            # Send action to robot
            self.robot.send_action(action_dict)
            
            logger.debug(f"Executed action {i+1}/{len(actions)}: {action}")
            
            # Wait for next execution step
            time.sleep(dt)
    
    def run_task(self, prompt: str, num_iterations: int = 1):
        """
        Run a complete task using π₀.₅ model.
        
        Args:
            prompt: Task instruction
            num_iterations: Number of times to run the task
        """
        logger.info(f"Starting task: '{prompt}'")
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Predict actions
            actions = self.predict_action(prompt)
            
            # Execute actions
            self.execute_action_sequence(actions)
            
            logger.info(f"Completed iteration {iteration + 1}")
            
            if iteration < num_iterations - 1:
                time.sleep(1.0)  # Brief pause between iterations
        
        logger.info("Task completed successfully")
    
    def disconnect(self):
        """Disconnect from SO-101 robot and RealSense D455."""
        logger.info("Disconnecting from SO-101 robot")
        self.robot.disconnect()
        
        if self.realsense:
            self.realsense.disconnect()


def main():
    """Example usage of SO-101 policy bridge with RealSense D455."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SO-101 + π₀.₅ Hardware Integration with RealSense D455")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to π₀.₅ checkpoint")
    parser.add_argument("--robot_port", default="/dev/ttyACM0", help="SO-101 serial port")
    parser.add_argument("--wrist_camera", type=int, default=0, help="Wrist camera index")
    parser.add_argument("--external_camera", type=int, default=1, help="External camera index")
    parser.add_argument("--use_realsense", action="store_true", default=True, help="Use RealSense D455")
    parser.add_argument("--no_realsense", action="store_false", dest="use_realsense", help="Disable RealSense D455")
    parser.add_argument("--prompt", default="pick up the object", help="Task instruction")
    parser.add_argument("--iterations", type=int, default=1, help="Number of task iterations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize bridge
        bridge = SO101PolicyBridge(
            checkpoint_dir=args.checkpoint_dir,
            robot_port=args.robot_port,
            wrist_camera_index=args.wrist_camera,
            external_camera_index=args.external_camera,
            use_realsense=args.use_realsense,
        )
        
        # Run task
        bridge.run_task(args.prompt, args.iterations)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        if 'bridge' in locals():
            bridge.disconnect()


if __name__ == "__main__":
    main()
