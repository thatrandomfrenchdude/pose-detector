# Setup Guide

Complete setup instructions for HRNet Pose Detection with NPU Acceleration.

## System Requirements
- **Python**: Version 3.8 or higher
- **Camera**: Built-in or external camera for real-time detection

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/thatrandomfrenchdude/pose-detector.git
cd pose-detector
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Or install manually:
pip install opencv-python numpy onnxruntime-qnn mediapipe
```

### 4. Download Model
1. Visit [Qualcomm AI Hub](https://aihub.qualcomm.com/)
2. Download the HRNet pose estimation model
3. Place the model file as `model/model.onnx`

### 5. Verify Installation
```bash
# Run basic tests
python main.py --test

# Check system info
python main.py --info
```

## Platform-Specific Setup

### Windows (Snapdragon X Elite)
```bash
# Ensure you have the latest Windows updates
# Install Visual C++ Redistributable if needed
pip install onnxruntime-qnn
```

### Windows (Other Processors)
```bash
# Use standard ONNX Runtime
pip install onnxruntime
# Or keep onnxruntime-qnn (will fallback to CPU)
```

### macOS/Linux
```bash
# MediaPipe typically works best on these platforms
pip install mediapipe opencv-python numpy
# ONNX Runtime for CPU
pip install onnxruntime
```

## Optimization Steps

### Generate NPU Context (Recommended)
```bash
# One-time optimization for 50x faster startup
python main.py --generate-context
```

This creates an optimized context model that loads much faster on subsequent runs.

### Camera Setup
```bash
# Check available cameras
python main.py --info

# Test with specific camera
python main.py --camera 1
```

## Verification

### Quick Test
```bash
python main.py --test
```

Expected output:
```
🧪 Running pose detection tests...
✅ Package imports: Available
✅ Model files: Found
✅ Application: Ready
🎉 TESTS PASSED - Application ready for use!
```

### Performance Test
```bash
# Run comprehensive test suite
python tests/test_suite.py
```

## Common Setup Issues

### NPU Not Detected
- Ensure you have a Snapdragon X Elite device
- Update Windows to latest version
- Install latest drivers from device manufacturer

### Model File Issues
- Verify model is exactly at `model/model.onnx`
- Check file size (should be 5-50MB)
- Ensure model is from Qualcomm AI Hub

### Camera Not Working
- Check camera permissions in system settings
- Try different camera index with `--camera 1`
- Ensure no other applications are using the camera

### Import Errors
```bash
# Reinstall dependencies
pip uninstall -y opencv-python mediapipe onnxruntime-qnn
pip install -r requirements.txt
```

## Next Steps

After successful setup:
1. Read the [User Guide](user_guide.md) for usage instructions
2. Check [API Reference](api.md) for programmatic usage
3. See [Troubleshooting](troubleshooting.md) for common issues

## Support

If you encounter issues:
1. Check [Troubleshooting](troubleshooting.md)
2. Run `python main.py --info` and share output
3. Create an issue with your system details