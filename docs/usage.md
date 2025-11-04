# Usage Guide
Detailed instructions on how to use the HRNet Pose Detection application.

**Table of Contents**
- [The Basics](#the-basics)
    - [Video Stream](#video-stream)
    - [Image Processing](#image-processing)
    - [Testing](#testing)
    - [NPU Optimization](#npu-optimization)
    - [Command Line Options](#command-line-options)
- [Advanced Functionality](#advanced-functionality)
    - [Camera Setup](#camera-setup)
    - [Custom Model Path](#custom-model-path)
    - [API Usage](#api-usage)

## The Basics
### Video Stream
```bash
# Run real-time pose detection with NPU acceleration
python main.py

# Force MediaPipe mode
python main.py --mediapipe
```
### Image Processing
```bash
# Process single image
python main.py --image photo.jpg

# Save result
python main.py --image photo.jpg --output result.jpg
```

### Testing
```bash
# Run comprehensive tests
python main.py --test
```

### NPU Optimization
This one-time operation takes ~5-10 seconds and makes startup 50x faster.
```bash
# Generate context model (one-time, ~5-10 seconds)
python main.py --generate-context
```
This optimization is not required, but skipping this step will make the ONNX Runtime model load more slowly as it needs to convert the model from float16 on each startup.

### Command Line Options
Get a full list of command line options:
```bash
python main.py --help
```

## Advanced Functionality
### Camera Setup
```bash
# Check available cameras
python main.py --info

# Test with specific camera
python main.py --camera 1
```

### Custom Model Path
```bash
python main.py --model path/to/your/model.onnx
```

### API Usage
If you want to use this package programmatically, you can import and use the `PoseDetectionApp` class:
```python
from src.pose_detection import PoseDetectionApp

# Initialize with NPU (default)
app = PoseDetectionApp()

# Process image
annotated_image, keypoints = app.process_image("photo.jpg")

# Process camera feed
app.process_camera()

# Get detector info
info = app.get_info()
print(f"Using: {info['detector_type']}")

# Always release resources
app.release()
```
