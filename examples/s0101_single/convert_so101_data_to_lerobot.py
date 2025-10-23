"""
Convert SO-101 data to LeRobot format for training.

This script helps convert raw SO-101 demonstration data to the LeRobot dataset format
for training OpenPI models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import SO101_JOINTS, SO101_DOF


logger = logging.getLogger(__name__)


def load_so101_episode(episode_path: Path) -> Dict[str, Any]:
    """
    Load a single SO-101 episode from directory.
    
    Expected structure:
    episode_path/
    ├── metadata.json
    ├── joint_positions.csv
    ├── actions.csv
    ├── images/
    │   ├── wrist/
    │   ├── external/
    │   └── realsense/
    └── depth/
        └── realsense/
    
    Args:
        episode_path: Path to episode directory
        
    Returns:
        Episode data dictionary
    """
    episode_path = Path(episode_path)
    
    # Load metadata
    metadata_file = episode_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Load joint positions and actions
    joint_positions = pd.read_csv(episode_path / "joint_positions.csv")
    actions = pd.read_csv(episode_path / "actions.csv")
    
    # Load images
    images = {}
    image_dir = episode_path / "images"
    
    for camera in ["wrist", "external", "realsense"]:
        camera_dir = image_dir / camera
        if camera_dir.exists():
            image_files = sorted(camera_dir.glob("*.jpg")) + sorted(camera_dir.glob("*.png"))
            images[camera] = [str(f) for f in image_files]
    
    # Load depth data if available
    depth_dir = episode_path / "depth" / "realsense"
    depth_files = []
    if depth_dir.exists():
        depth_files = sorted(depth_dir.glob("*.npy"))
        depth_files = [str(f) for f in depth_files]
    
    return {
        "metadata": metadata,
        "joint_positions": joint_positions,
        "actions": actions,
        "images": images,
        "depth_files": depth_files,
        "episode_length": len(joint_positions)
    }


def convert_to_lerobot_format(
    episode_data: Dict[str, Any],
    episode_id: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Convert SO-101 episode to LeRobot format.
    
    Args:
        episode_data: Raw episode data
        episode_id: Episode identifier
        output_dir: Output directory for converted data
        
    Returns:
        LeRobot format episode data
    """
    episode_length = episode_data["episode_length"]
    
    # Create output directories
    episode_dir = output_dir / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert observations
    observations = {}
    
    # Joint positions (state)
    joint_data = episode_data["joint_positions"]
    observations["state"] = joint_data[SO101_JOINTS].values.astype(np.float32)
    
    # Process images
    for camera, image_files in episode_data["images"].items():
        if not image_files:
            continue
            
        camera_key = f"observation/{camera}_image"
        processed_images = []
        
        for img_file in image_files:
            img = cv2.imread(img_file)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize to standard size
                img = cv2.resize(img, (224, 224))
                processed_images.append(img)
        
        if processed_images:
            observations[camera_key] = np.stack(processed_images)
    
    # Process depth data
    if episode_data["depth_files"]:
        depth_data = []
        for depth_file in episode_data["depth_files"]:
            depth = np.load(depth_file)
            # Resize and normalize
            depth = cv2.resize(depth, (224, 224))
            depth = depth.astype(np.float32) / 1000.0  # Convert mm to meters
            depth_data.append(depth)
        
        if depth_data:
            observations["observation/realsense_depth"] = np.stack(depth_data)
    
    # Convert actions
    action_data = episode_data["actions"]
    actions = action_data[SO101_JOINTS].values.astype(np.float32)
    
    # Create LeRobot episode structure
    lerobot_episode = {
        "episode_index": episode_id,
        "length": episode_length,
        "timestamp": episode_data["metadata"].get("timestamp", 0),
        "task": episode_data["metadata"].get("task", "unknown"),
        "observations": observations,
        "actions": actions,
        "rewards": np.zeros(episode_length, dtype=np.float32),  # Placeholder
        "dones": np.zeros(episode_length, dtype=bool),
    }
    
    # Mark last step as done
    lerobot_episode["dones"][-1] = True
    
    return lerobot_episode


def save_lerobot_episode(
    episode_data: Dict[str, Any],
    output_file: Path
) -> None:
    """
    Save episode in LeRobot format.
    
    Args:
        episode_data: Episode data in LeRobot format
        output_file: Output file path
    """
    # Save as compressed numpy archive
    np.savez_compressed(
        output_file,
        **episode_data
    )
    
    logger.info(f"Saved episode to {output_file}")


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str = "so101_demonstrations"
) -> None:
    """
    Convert entire SO-101 dataset to LeRobot format.
    
    Args:
        input_dir: Directory containing raw SO-101 episodes
        output_dir: Output directory for LeRobot dataset
        dataset_name: Name of the dataset
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all episode directories
    episode_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode")]
    episode_dirs.sort()
    
    logger.info(f"Found {len(episode_dirs)} episodes to convert")
    
    # Convert each episode
    dataset_info = {
        "dataset_name": dataset_name,
        "robot_type": "so101",
        "total_episodes": len(episode_dirs),
        "episodes": []
    }
    
    for i, episode_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        try:
            # Load raw episode
            episode_data = load_so101_episode(episode_dir)
            
            # Convert to LeRobot format
            lerobot_episode = convert_to_lerobot_format(
                episode_data, i, output_dir
            )
            
            # Save episode
            episode_file = output_dir / f"episode_{i:06d}.npz"
            save_lerobot_episode(lerobot_episode, episode_file)
            
            # Add to dataset info
            dataset_info["episodes"].append({
                "episode_id": i,
                "length": episode_data["episode_length"],
                "task": episode_data["metadata"].get("task", "unknown"),
                "file": str(episode_file.name)
            })
            
        except Exception as e:
            logger.error(f"Failed to convert episode {episode_dir}: {e}")
            continue
    
    # Save dataset metadata
    metadata_file = output_dir / "dataset_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Dataset conversion complete. Saved to {output_dir}")
    logger.info(f"Total episodes: {len(dataset_info['episodes'])}")


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(description="Convert SO-101 data to LeRobot format")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing raw SO-101 episodes"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path,
        required=True,
        help="Output directory for LeRobot dataset"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="so101_demonstrations",
        help="Name of the dataset"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Convert dataset
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )


if __name__ == "__main__":
    main()
