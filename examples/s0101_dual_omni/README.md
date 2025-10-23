# Dual S0101 Robotic Arms + Omni Base Integration with OpenPI π₀.₅

This directory contains examples for integrating dual SO-101 robotic arms mounted on an omni-directional base with OpenPI's π₀.₅ model using RGB cameras only.

## Hardware Requirements

- **2x SO-101 Robotic Arms** (6 DOF each with Feetech STS3215 servos)
- **1x Omni-directional Base** (3 DOF: x, y, rotation)
- **2x Wrist RGB Cameras** (1 mounted on each arm's end effector)
- **1x Intel RealSense Camera** (used as RGB only, depth disabled)
- **USB Connections** for cameras and robot communication

## Configuration Details

- **Total DOF**: 15 (6 per arm + 3 for omni base)
- **Action Space**: [left_arm(6) + right_arm(6) + omni_base(3)]
- **Cameras**: 3 RGB cameras total
- **Mobile base** for workspace navigation

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

| **File**             | **Purpose**              | **Description**                              |
| -------------------- | ------------------------ | -------------------------------------------- |
| `main.py`            | **Testing & Simulation** | Test π₀.₅ model with dummy mobile dual arms  |
| `hardware_bridge.py` | **Hardware Integration** | Complete hardware bridge for mobile platform |
| `constants.py`       | **Configuration**        | Mobile dual arm specifications and constants |
| `README.md`          | **Documentation**        | This file                                    |

## Quick Start

### 1. Testing with Simulation Data

Test the π₀.₅ model with dummy mobile dual SO-101 observations:

```bash
cd /path/to/openpi

python examples/s0101_dual_omni/main.py \
    --checkpoint_dir "path/to/pi05_checkpoint" \
    --config_name "pi05_s0101_dual_omni" \
    --prompt "navigate to table and pick objects with both arms" \
    --num_steps 5
```

### 2. Hardware Integration

Run with real mobile dual SO-101 hardware:

```bash
python examples/s0101_dual_omni/hardware_bridge.py \
    --checkpoint_dir "path/to/pi05_checkpoint" \
    --left_robot_port "/dev/ttyACM0" \
    --right_robot_port "/dev/ttyACM1" \
    --omni_base_port "/dev/ttyACM2" \
    --left_wrist_camera 0 \
    --right_wrist_camera 1 \
    --realsense_camera 2 \
    --prompt "navigate and manipulate objects with coordinated arms" \
    --iterations 1
```

## Training

To train on your mobile dual arm dataset:

```bash
# Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_s0101_dual_omni_finetune

# Start training  
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_s0101_dual_omni_finetune --exp-name=mobile_dual_arm_demo --overwrite
```

## Hardware Notes

- **Mobile Base**: Omni-directional wheels allow translation and rotation
- **Arm Coordination**: Both arms work while base can move simultaneously  
- **Navigation**: Base can navigate to different workspace locations
- **Workspace Extension**: Mobile base expands reachable workspace significantly
- **Camera Placement**: Position RealSense for optimal workspace view during movement
- **Calibration**: Each arm and base requires individual calibration
- **Safety**: Implement collision detection between arms and with environment

## Omni Base Control

The omni base uses 3 DOF velocity control:
- **X Velocity**: Forward/backward movement (m/s)
- **Y Velocity**: Left/right movement (m/s)  
- **Angular Velocity**: Rotation around vertical axis (rad/s)

## Coordination Modes

1. **Static Manipulation**: Base stationary, arms manipulate
2. **Mobile Manipulation**: Base moves while arms adjust posture
3. **Sequential**: Navigate then manipulate
4. **Coordinated**: Simultaneous base and arm movement
