# Single S0101 Robotic Arm Integration with OpenPI π₀.₅

This directory contains examples for integrating a single SO-101 robotic arm with OpenPI's π₀.₅ model using RGB cameras only.

## Hardware Requirements

- **1x SO-101 Robotic Arm** (6 DOF with Feetech STS3215 servos)
- **1x Wrist RGB Camera** (mounted on end effector)
- **1x Intel RealSense Camera** (used as RGB only, depth disabled)
- **USB Connections** for cameras and robot communication

## Software Dependencies

```bash
# Install LeRobot with Feetech support
pip install -e ".[feetech]"

# Install Intel RealSense SDK
pip install pyrealsense2

# Install OpenCV for image processing
pip install opencv-python
```

## Files Overview

| **File**             | **Purpose**              | **Description**                              |
| -------------------- | ------------------------ | -------------------------------------------- |
| `main.py`            | **Testing & Simulation** | Test π₀.₅ model with dummy SO-101 data       |
| `hardware_bridge.py` | **Hardware Integration** | Complete hardware bridge with RealSense D455 |
| `README.md`          | **Documentation**        | This file                                    |

## Quick Start

### 1. Testing with Simulation Data

Test the π₀.₅ model with dummy SO-101 observations:

```bash
cd /path/to/openpi

python examples/so101/main.py \
    --checkpoint_dir "path/to/pi05_checkpoint" \
    --config_name "pi05_so101" \
    --prompt "pick up the red block and place it in the blue container" \
    --num_steps 5
```

### 2. Hardware Integration

Run with real SO-101 hardware and RealSense D455:

```bash
python examples/so101/hardware_bridge.py \
    --checkpoint_dir "path/to/pi05_checkpoint" \
    --robot_port "/dev/ttyACM0" \
    --wrist_camera 0 \
    --external_camera 1 \
    --use_realsense \
    --prompt "pick up the nearest object and place it in the container" \
    --iterations 1
```

### 3. RGB-Only Mode (without RealSense)

If RealSense D455 is not available:

```bash
python examples/so101/hardware_bridge.py \
    --checkpoint_dir "path/to/pi05_checkpoint" \
    --no_realsense \
    --prompt "pick up the object"
```

## SO-101 Configuration

The SO-101 uses the following joint configuration:

| **Joint**       | **Index** | **Range** | **Description**    |
| --------------- | --------- | --------- | ------------------ |
| `shoulder_pan`  | 0         | ±180°     | Base rotation      |
| `shoulder_lift` | 1         | ±90°      | Shoulder elevation |
| `elbow_flex`    | 2         | ±135°     | Elbow flexion      |
| `wrist_flex`    | 3         | ±90°      | Wrist flexion      |
| `wrist_roll`    | 4         | ±180°     | Wrist rotation     |
| `gripper`       | 5         | 0-100%    | Gripper opening    |

## Sensor Configuration

### Camera Setup

1. **Wrist Camera** (index 0): Close-up manipulation view
2. **External Camera** (index 1): Scene overview
3. **RealSense D455**: RGB-D perception with depth

### RealSense D455 Features

- **Resolution**: 640x480 or 1280x720
- **Depth Range**: 0.3 - 5.0 meters
- **Frame Rate**: 30 FPS
- **Baseline**: 95mm for accurate depth
- **Auto-exposure**: Automatic lighting adaptation

## Advanced Usage

### Custom Prompts

The π₀.₅ model supports natural language instructions:

```bash
# Distance-based tasks (using depth)
--prompt "pick up the closest red object"
--prompt "place the object on the far table"

# Spatial reasoning
--prompt "avoid the obstacles and reach the target"
--prompt "stack the blocks in order of size"

# Complex manipulation
--prompt "open the container and place the item inside"
```

### Camera Configuration

```bash
# High resolution mode
python examples/so101/hardware_bridge.py \
    --checkpoint_dir "path/to/checkpoint" \
    --realsense_width 1280 \
    --realsense_height 720 \
    --realsense_fps 15

# Multiple iterations
python examples/so101/hardware_bridge.py \
    --checkpoint_dir "path/to/checkpoint" \
    --iterations 5 \
    --prompt "repeatedly pick and place objects"
```

## Troubleshooting

### Common Issues

1. **RealSense Connection Failed**
   ```bash
   # Check USB connection
   lsusb | grep Intel
   
   # Test RealSense
   realsense-viewer
   ```

2. **SO-101 Not Responding**
   ```bash
   # Check serial port
   ls /dev/ttyACM*
   
   # Test with LeRobot
   python -m lerobot.test_robot so101_follower
   ```

3. **Camera Index Issues**
   ```bash
   # List available cameras
   v4l2-ctl --list-devices
   
   # Test camera
   python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read()[0])"
   ```

### Performance Optimization

- **Lighting**: Ensure good lighting for RGB cameras
- **Mounting**: Secure RealSense mounting to avoid vibration
- **Calibration**: Regular calibration of SO-101 servos
- **Network**: Use local inference for low latency

## Integration with π₀.₅

The SO-101 integration uses OpenPI's π₀.₅ model with:

- **Flow Matching**: Advanced action generation algorithm
- **Multi-Modal Input**: RGB + Depth + Proprioception
- **Language Conditioning**: Natural language task instructions
- **Action Horizon**: 50-step action sequences

## Data Collection

To collect training data for your specific tasks:

```bash
# Use LeRobot's data collection tools
python -m lerobot.record \
    --robot so101_follower \
    --cameras wrist external realsense \
    --output-dir ./data/so101_tasks
```

## Safety Notes

⚠️ **Important Safety Guidelines**:

- Always test in simulation first
- Keep emergency stop accessible
- Check joint limits before execution
- Monitor robot behavior continuously
- Use safety barriers for initial testing

## Support

For issues and questions:

- **OpenPI Issues**: [GitHub Issues](https://github.com/physical-intelligence/openpi)
- **LeRobot SO-101**: [LeRobot Documentation](https://github.com/huggingface/lerobot)
- **RealSense Support**: [Intel RealSense Community](https://github.com/IntelRealSense/librealsense)
