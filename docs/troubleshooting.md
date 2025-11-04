# Troubleshooting Guide

Solutions for common issues with HRNet Pose Detection.

## Quick Diagnostics

### Run System Check
```bash
# Get comprehensive system information
python main.py --info

# Run functionality tests
python main.py --test
```

## Common Issues

### 1. NPU Not Working

**Symptoms:**
- "NPU not available, using CPU" message
- Slow inference performance
- QNN provider failed errors

**Solutions:**
```bash
# Check QNN availability
python -c "import onnxruntime; print('QNN' in onnxruntime.get_available_providers())"

# Verify Snapdragon X Elite device
# Update Windows to latest version
# Install latest device drivers

# Test with MediaPipe fallback
python main.py --mediapipe
```

**Hardware Requirements:**
- Snapdragon X Elite processor with NPU
- Windows 11 with latest updates
- Proper QNN drivers installed

### 2. Model Loading Errors

**Symptoms:**
- "Model not found" errors
- "Unsupported model IR version" errors
- Model loading timeouts

**Solutions:**
```bash
# Check model file exists
ls -la model/model.onnx

# Verify model size (should be 5-50MB)
# Re-download model from Qualcomm AI Hub

# Try generating context
python main.py --generate-context

# Test with different model
python main.py --model path/to/other/model.onnx
```

**Model Requirements:**
- ONNX format from Qualcomm AI Hub
- HRNet or compatible architecture
- Placed exactly at `model/model.onnx`

### 3. Slow Startup Performance

**Symptoms:**
- 5-10 second startup time
- "First load will be slow" messages

**Solutions:**
```bash
# Generate context model (one-time)
python main.py --generate-context

# Verify context was created
ls -la model/model_ctx.onnx

# Subsequent runs should be <0.1s
```

**Expected Performance:**
- **Without context**: 5-10 seconds
- **With context**: <0.1 seconds

### 4. Camera Issues

**Symptoms:**
- "Cannot open camera" errors
- Black/frozen camera feed
- Camera not detected

**Solutions:**
```bash
# Check available cameras
python main.py --info

# Try different camera index
python main.py --camera 1
python main.py --camera 2

# Check camera permissions in system settings
# Close other applications using camera
```

**Camera Requirements:**
- USB or built-in camera
- Camera permissions enabled
- No other applications using camera

### 5. Import/Dependency Errors

**Symptoms:**
- "ModuleNotFoundError" messages
- Import errors for ONNX or MediaPipe

**Solutions:**
```bash
# Check current environment
pip list

# Reinstall dependencies
pip uninstall -y opencv-python mediapipe onnxruntime-qnn
pip install -r requirements.txt

# For specific issues:
pip install --upgrade opencv-python
pip install --upgrade mediapipe
pip install --upgrade onnxruntime-qnn
```

**Environment Requirements:**
- Python 3.8+
- Virtual environment recommended
- All dependencies from requirements.txt

### 6. Low Performance/FPS

**Symptoms:**
- <10 FPS on real-time detection
- Laggy camera feed
- High CPU usage

**Solutions:**
```bash
# Generate context model
python main.py --generate-context

# Use MediaPipe for comparison
python main.py --mediapipe

# Check system resources
# Close unnecessary applications
# Lower camera resolution if needed
```

**Performance Expectations:**
- **NPU mode**: 60-100+ FPS
- **MediaPipe**: 20-60 FPS
- **CPU fallback**: 5-20 FPS

### 7. Image Processing Errors

**Symptoms:**
- "Could not load image" errors
- Blank output images
- Format not supported

**Solutions:**
```bash
# Check image format (JPG, PNG supported)
# Verify file exists and is readable
# Try with different image

# Test with known good image
python main.py --image tests/sample.jpg
```

**Supported Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## Platform-Specific Issues

### Windows Issues

**QNN Provider Errors:**
```bash
# Install Visual C++ Redistributable
# Update Windows to build 22000+
# Check Windows Update for driver updates
```

**Permission Errors:**
```bash
# Run as Administrator if needed
# Check antivirus software blocking
# Verify Windows Defender exclusions
```

### Linux/macOS Issues

**ONNX Runtime on ARM:**
```bash
# Use CPU version for ARM Macs
pip install onnxruntime

# For Linux ARM, compile from source if needed
```

**Camera Access:**
```bash
# Linux: Check user permissions
sudo usermod -a -G video $USER

# macOS: Grant camera permissions in System Preferences
```

## Advanced Diagnostics

### Check ONNX Runtime Providers
```python
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
print("QNN available:", "QNNExecutionProvider" in ort.get_available_providers())
```

### Test Model Loading Manually
```python
from src.pose_detection.detectors.onnx_detector import ONNXPoseDetector

try:
    detector = ONNXPoseDetector("model/model.onnx")
    print("✅ Model loaded successfully")
    print("Info:", detector.get_model_info())
    detector.release()
except Exception as e:
    print("❌ Model loading failed:", e)
```

### Check Camera Manually
```python
import cv2

# Test camera access
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera {i}: Available")
        cap.release()
    else:
        print(f"❌ Camera {i}: Not available")
```

## Performance Optimization

### For NPU Mode
1. **Generate context**: `python main.py --generate-context`
2. **Update drivers**: Latest Snapdragon X Elite drivers
3. **Close applications**: Free up NPU resources
4. **Use recommended model**: HRNet from Qualcomm AI Hub

### For MediaPipe Mode
1. **Reduce complexity**: Use `model_complexity=0` for speed
2. **Adjust confidence**: Lower detection thresholds
3. **Skip frames**: Process every 2nd or 3rd frame
4. **Optimize resolution**: Use 320x240 for speed

### System-Level
1. **Power settings**: High performance mode
2. **Background apps**: Close unnecessary software
3. **System cooling**: Ensure adequate cooling
4. **Memory**: 8GB+ RAM recommended

## Logging and Debugging

### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
python main.py --info
```

### Save Debug Information
```bash
# Run with output redirection
python main.py --info > debug_info.txt 2>&1

# Share debug_info.txt when reporting issues
```

## Getting Help

### Before Reporting Issues
1. Run `python main.py --test` and share output
2. Run `python main.py --info` and share output
3. Check this troubleshooting guide
4. Try different detection modes

### Information to Include
- Operating system and version
- Hardware specifications (CPU, NPU)
- Python version and environment
- Complete error messages
- Output of `--info` command

### Community Resources
- Check existing issues in repository
- Review documentation thoroughly
- Test with provided examples
- Use MediaPipe fallback for comparison

## FAQ

**Q: Why is startup slow on first run?**
A: NPU needs to optimize the model. Generate context with `--generate-context` for faster subsequent runs.

**Q: Can I use this without NPU?**
A: Yes! MediaPipe fallback works on any system. Use `--mediapipe` flag.

**Q: What model should I use?**
A: Download HRNet pose model from Qualcomm AI Hub for best NPU performance.

**Q: Why don't I see all keypoints?**
A: Different models detect different numbers of keypoints. NPU models vary, MediaPipe uses 33 landmarks.

**Q: Can I process video files?**
A: Currently supports real-time camera and static images. Video file support planned for future versions.