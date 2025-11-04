# User Guide

Complete guide for using HRNet Pose Detection with NPU Acceleration.

## Quick Start

### Real-time Pose Detection
```bash
# Start real-time detection with default camera
python main.py

# Use specific camera
python main.py --camera 1

# Force MediaPipe mode
python main.py --mediapipe
```

### Image Processing
```bash
# Process single image
python main.py --image photo.jpg

# Save result to file
python main.py --image photo.jpg --output result.jpg
```

## Command Line Interface

### Basic Commands
```bash
# Real-time camera detection (default)
python main.py

# Get system information
python main.py --info

# Run functionality tests
python main.py --test

# Show help
python main.py --help
```

### Advanced Options
```bash
# Specify model path
python main.py --model path/to/model.onnx

# Generate context for faster startup
python main.py --generate-context

# Use specific camera device
python main.py --camera 0
```

## Detection Modes

### NPU Mode (Recommended)
- **Best for**: Snapdragon X Elite devices
- **Performance**: 60-100+ FPS
- **Startup**: <0.1s (with context), ~5s (first-time)
- **Features**: Hardware acceleration, context caching

```bash
# Default mode (auto-selects NPU if available)
python main.py
```

### MediaPipe Mode (Fallback)
- **Best for**: Cross-platform compatibility
- **Performance**: 20-60 FPS
- **Startup**: ~1-2s
- **Features**: 33-point landmarks, reliable detection

```bash
# Force MediaPipe mode
python main.py --mediapipe
```

## Performance Optimization

### NPU Context Generation
Generate a context model for 50x faster startup:

```bash
# One-time optimization (takes 5-10 seconds)
python main.py --generate-context
```

**Before**: 5-7 seconds loading time  
**After**: 0.1 seconds loading time

### Camera Settings
The application automatically optimizes camera settings:
- Resolution: 640x480 (default)
- FPS: 30 (target)
- Mirror effect: Enabled for selfie-style view

## Keypoint Detection

### NPU Mode Output
- Variable number of keypoints (model-dependent)
- Typically 17-33 body landmarks
- Coordinates relative to input image

### MediaPipe Mode Output
- 33 pose landmarks
- Standardized body pose model
- Includes visibility scores

## Real-time Controls

### During Camera Detection
- **'q' key**: Quit application
- **Window close**: Stop detection
- **Ctrl+C**: Emergency stop

### Display Information
- FPS counter (top-left)
- Keypoint count
- Detection method (NPU-ONNX or MediaPipe)

## Programmatic Usage

### Basic Application
```python
from src.pose_detection import PoseDetectionApp

# Initialize with default settings
app = PoseDetectionApp()

# Process single image
result_image, keypoint_count = app.process_image("photo.jpg")

# Get application info
info = app.get_info()
print(f"Using: {info['detector_type']}")

# Clean up
app.release()
```

### Advanced Configuration
```python
# Force MediaPipe mode
app = PoseDetectionApp(force_mediapipe=True)

# Custom model path
app = PoseDetectionApp(model_path="custom/model.onnx")

# Process with specific output
app.process_image("input.jpg", "output.jpg")
```

### Camera Processing
```python
# Start real-time detection
app.process_camera(camera_id=0)

# This blocks until user quits with 'q'
```

## File Organization

### Input Formats
- **Images**: JPG, PNG, BMP, TIFF
- **Video**: MP4, AVI, MOV (for future versions)

### Output Formats
- **Images**: Same format as input
- **Annotations**: Keypoints drawn as colored circles
- **Connections**: Lines between related keypoints (MediaPipe)

### Model Files
```
model/
├── model.onnx          # Original model (required)
└── model_ctx.onnx      # Generated context (optional, faster)
```

## Integration Examples

### Batch Processing
```python
import os
from src.pose_detection import PoseDetectionApp

app = PoseDetectionApp()

# Process all images in directory
input_dir = "input_images"
output_dir = "output_images"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        app.process_image(input_path, output_path)

app.release()
```

### Custom Analysis
```python
from src.pose_detection.detectors.onnx_detector import ONNXPoseDetector
import cv2

# Direct detector usage
detector = ONNXPoseDetector("model/model.onnx")

# Load and process image
image = cv2.imread("photo.jpg")
annotated_image, keypoints = detector.detect_pose(image)

# Analyze keypoints
print(f"Detected {len(keypoints)} keypoints")
for i, (x, y) in enumerate(keypoints):
    print(f"Keypoint {i}: ({x}, {y})")

detector.release()
```

## Best Practices

### For Best Performance
1. **Generate context model**: Use `--generate-context` for fastest startup
2. **Stable lighting**: Ensure good, consistent lighting
3. **Clear background**: Minimize background clutter
4. **Proper distance**: Stand 3-8 feet from camera
5. **Full body visible**: Ensure full pose is in frame

### For Development
1. **Test first**: Always run tests before deployment
2. **Handle exceptions**: Graceful error handling
3. **Release resources**: Always call `app.release()`
4. **Monitor performance**: Check FPS and adjust settings

### For Deployment
1. **Verify hardware**: Test NPU availability
2. **Fallback ready**: Ensure MediaPipe works
3. **Model availability**: Include model files
4. **Dependencies**: Install all requirements

## Troubleshooting

### Common Issues
- **Low FPS**: Check [Troubleshooting](troubleshooting.md)
- **NPU not working**: Verify Snapdragon X Elite device
- **Camera issues**: Check camera permissions and availability
- **Model errors**: Verify model file location and format

### Getting Help
1. Run `python main.py --info` for system status
2. Check [Troubleshooting Guide](troubleshooting.md)
3. Review logs for error messages
4. Test with `python main.py --test`

## Next Steps

- Explore [API Reference](api.md) for detailed programming interface
- Check [Developer Guide](developer_guide.md) for contributing
- See [Troubleshooting](troubleshooting.md) for problem solving