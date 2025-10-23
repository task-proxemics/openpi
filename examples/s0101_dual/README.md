# Dual S0101 Robotic Arms Integration with OpenPI π₀.₅

This directory contains examples for integrating dual SO-101 robotic arms with OpenPI's π₀.₅ model using RGB cameras only.

## Hardware Requirements

- **2x SO-101 Robotic Arms** (6 DOF each with Feetech STS3215 servos)
- **2x Wrist RGB Cameras** (1 mounted on each arm's end effector)
- **1x Intel RealSense Camera** (used as RGB only, depth disabled)
- **USB Connections** for cameras and robot communication

## Configuration Details

- **Total DOF**: 12 (6 per arm)
- **Action Space**: [left_arm(6) + right_arm(6)]
- **Cameras**: 3 RGB cameras total
- **No omni-directional base** (stationary dual arm setup)

## Software Dependencies

```bash
# Install LeRobot with Feetech support
pip install -e ".[feetech]"

# Install OpenCV for image processing
opencv-python

# Install other dependencies
pip install -r requirements.txt
```

## Files Overview

| **File**             | **Purpose**              | **Description**                          |
| -------------------- | ------------------------ | ---------------------------------------- |
| `main.py`            | **Testing & Simulation** | Test π₀.₅ model with dummy dual arm data |
| `hardware_bridge.py` | **Hardware Integration** | Complete hardware bridge for dual arms   |
| `constants.py`       | **Configuration**        | Dual arm specifications and constants    |
| `README.md`          | **Documentation**        | This file                                |

## Quick Start

### 1. Testing with Simulation Data

Test the π₀.₅ model with dummy dual SO-101 observations:

```bash
cd /path/to/openpi

python examples/s0101_dual/main.py \
    --checkpoint_dir "path/to/pi05_checkpoint" \
    --config_name "pi05_s0101_dual" \
    --prompt "coordinate both arms to pick up objects" \
    --num_steps 5
```

### 2. Hardware Integration

Run with real dual SO-101 hardware:

```bash
python examples/s0101_dual/hardware_bridge.py \
    --checkpoint_dir "path/to/pi05_checkpoint" \
    --left_robot_port "/dev/ttyACM0" \
    --right_robot_port "/dev/ttyACM1" \
    --left_wrist_camera 0 \
    --right_wrist_camera 1 \
    --realsense_camera 2 \
    --prompt "coordinate picking and placing with both arms" \
    --iterations 1
```

## Training

To train on your dual arm dataset:

```bash
# Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_s0101_dual_finetune

# Start training  
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_s0101_dual_finetune --exp-name=dual_arm_demo --overwrite
```

## Hardware Notes

- **Arm Coordination**: Both arms can work independently or in coordination
- **Workspace**: Ensure arms don't collide during operation
- **Camera Placement**: Position RealSense for optimal workspace view
- **Calibration**: Each arm requires individual calibration
- **Safety**: Implement collision detection between arms
