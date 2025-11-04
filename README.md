# HRNet Pose Detection with NPU Acceleration

Real-time human pose estimation using **Qualcomm Snapdragon X Elite NPU** with MediaPipe fallback support.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run real-time NPU pose detection (default)
python main.py

# Process image
python main.py --image photo.jpg --output result.jpg

# Run tests
python main.py --test
```

## 🏗️ Installation

### Automated Setup (Recommended)

**Windows:**
```powershell
# Run automated setup script
.\scripts\setup.ps1

# Or with options
.\scripts\setup.ps1 -SkipTests -SkipVenv
```

**Linux/macOS:**
```bash
# Make script executable and run
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or with options
./scripts/setup.sh --skip-tests --skip-venv
```

### Manual Setup

1. **Clone and setup**:
   ```bash
   git clone https://github.com/thatrandomfrenchdude/pose-detector.git
   cd pose-detector
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model**:
   - Download HRNet model from [Qualcomm AI Hub](https://aihub.qualcomm.com/)
   - Place as `model/model.onnx`

4. **Verify installation**:
   ```bash
   # Quick validation
   python scripts/validate_install.py
   
   # Or run tests
   python main.py --test
   ```

## 🎮 Usage

### Real-time Camera Detection
```bash
# Default NPU mode
python main.py

# Specify camera
python main.py --camera 1

# Force MediaPipe mode
python main.py --mediapipe
```

### Image Processing
```bash
# Process single image
python main.py --image input.jpg

# Save result
python main.py --image input.jpg --output result.jpg
```

### NPU Optimization
```bash
# Generate context model (one-time, ~5-10 seconds)
python main.py --generate-context

# After generation, startup is 50x faster!
```

### Information and Testing
```bash
# Show detector info
python main.py --info

# Run comprehensive tests
python main.py --test

# Run test suite with benchmarks
python test_suite.py
```

## ⚙️ Configuration

The application automatically selects the best available detection method:

1. **NPU (QNN)** - Primary choice for Snapdragon X Elite
2. **MediaPipe** - Automatic fallback for compatibility

### Detection Priority:
- If ONNX model + QNN available → **NPU Mode**
- If NPU fails or unavailable → **MediaPipe Fallback**
- Graceful error handling with informative messages

## 🏎️ Performance

### NPU Mode (Snapdragon X Elite)
- **Startup**: <0.1s (with context), ~5s (first-time)
- **Inference**: ~10-30ms per frame
- **Throughput**: 30-100+ FPS (depending on resolution)

### MediaPipe Mode
- **Startup**: ~1-2s
- **Inference**: ~15-50ms per frame
- **Throughput**: 20-60 FPS

### Context Model Benefits
```bash
# Generate once for 50x faster startup
python main.py --generate-context

# Before: 5-7 seconds loading
# After:  0.1 seconds loading
```

## 🗂️ Project Structure

```
pose-detection/
├── src/                    # Source code
│   └── pose_detection/    # Main package
│       ├── core/          # Core application logic
│       ├── detectors/     # NPU and MediaPipe detectors
│       └── utils/         # Utility functions
├── tests/                 # Test suite
│   ├── test_suite.py     # Comprehensive tests
│   ├── test_runner.py    # Test runner
│   └── test_*.py         # Individual test modules
├── docs/                  # Documentation
│   ├── setup.md          # Setup guide
│   ├── user_guide.md     # User documentation
│   ├── api.md            # API reference
│   └── troubleshooting.md # Troubleshooting guide
├── scripts/               # Setup and utility scripts
│   ├── setup.ps1         # Windows setup
│   ├── setup.sh          # Linux/macOS setup
│   └── run_tests.*       # Test runners
├── model/                 # Model files
│   ├── model.onnx        # Your HRNet model
│   └── model_ctx.onnx    # Generated context model
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🧪 Testing

### Quick Tests
```bash
# Basic functionality test
python main.py --test

# Validate installation
python scripts/validate_install.py
```

### Comprehensive Testing
```bash
# Windows
.\scripts\run_tests.ps1

# Linux/macOS
./scripts/run_tests.sh

# Manual comprehensive tests
python tests/test_suite.py
```

### Test Options
```bash
# Quick tests only
.\scripts\run_tests.ps1 -Quick

# With coverage analysis
.\scripts\run_tests.ps1 -Coverage

# With code linting
.\scripts\run_tests.ps1 -Lint
```

**Test Coverage:**
- ✅ Package imports and dependencies
- ✅ Model file validation
- ✅ NPU detection functionality
- ✅ MediaPipe fallback
- ✅ Application initialization
- ✅ Camera availability
- ✅ Performance benchmarks
- ✅ Context generation

## ❗ Troubleshooting

### NPU Not Working
```bash
# Check QNN installation
python -c "import onnxruntime; print('QNN' in onnxruntime.get_available_providers())"

# Verify model file
python main.py --info

# Test with MediaPipe fallback
python main.py --mediapipe
```

### Slow Startup
```bash
# Generate context model for faster loading
python main.py --generate-context
```

### Camera Issues
```bash
# Check available cameras
python main.py --test

# Try different camera index
python main.py --camera 1
```

### Model Issues
- Ensure model file exists at `model/model.onnx`
- Verify model is from Qualcomm AI Hub (HRNet format)
- Check model file size (should be ~5-50MB)

## 🔧 Advanced Usage

### Custom Model Path
```bash
python main.py --model path/to/your/model.onnx
```

### Command Line Options
```bash
python main.py --help
```

### API Usage
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

## 🎯 Key Classes

### `PoseDetectionApp` (src/pose_detection/core/app.py)
Main application class that automatically selects best detector:
- `process_camera()` - Real-time detection
- `process_image()` - Single image processing
- `get_info()` - Detector information

### `ONNXPoseDetector` (src/pose_detection/detectors/onnx_detector.py)
NPU-accelerated detection:
- Automatic context model detection
- QNN provider optimization
- Keypoint extraction and visualization

### `MediaPipePoseDetector` (src/pose_detection/detectors/mediapipe_detector.py)
Fallback detection:
- Google MediaPipe framework
- 33-point pose landmarks
- Cross-platform compatibility

## 🌟 Best Practices

### For Best NPU Performance:
1. **Generate context model**: `python main.py --generate-context`
2. **Use recommended model**: Download from Qualcomm AI Hub
3. **Optimal input size**: 256x256 or 384x384 resolution
4. **Batch processing**: Process multiple images together when possible

### For Development:
1. **Test first**: Always run `python main.py --test`
2. **Use fallbacks**: MediaPipe ensures compatibility
3. **Monitor performance**: Built-in FPS counters
4. **Handle errors**: Graceful fallback on NPU issues

## 📊 Benchmarks

Run comprehensive benchmarks:
```bash
python test_suite.py
```

**Typical Results (Snapdragon X Elite):**
- NPU Mode: 60-100 FPS
- MediaPipe: 30-60 FPS
- Context Loading: 50x faster startup

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass: `python tests/test_suite.py` or `./scripts/run_tests.sh`
5. Update documentation in `docs/`
6. Submit pull request

### Development Setup
```bash
# Use automated setup
./scripts/setup.sh

# Install development dependencies
pip install pytest coverage flake8

# Run comprehensive tests
./scripts/run_tests.sh --coverage --lint
```

### Project Guidelines
- Follow the modular structure in `src/`
- Add comprehensive tests to `tests/`
- Update documentation in `docs/`
- Use the provided scripts for setup and testing

## 📜 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **Qualcomm**: NPU acceleration and QNN framework
- **Google**: MediaPipe pose detection
- **Microsoft**: ONNX Runtime framework
- **OpenCV**: Computer vision utilities

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Setup Guide](docs/setup.md)** - Detailed installation instructions
- **[User Guide](docs/user_guide.md)** - Complete usage documentation  
- **[API Reference](docs/api.md)** - Programming interface details
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

### Quick Links
- **First time setup**: Follow [Setup Guide](docs/setup.md)
- **Having issues**: Check [Troubleshooting](docs/troubleshooting.md)  
- **Programming**: See [API Reference](docs/api.md)
- **General usage**: Read [User Guide](docs/user_guide.md)

---

**🚀 Ready to detect poses with NPU acceleration? Start with `python main.py`!**

For detailed setup instructions, see the [Setup Guide](docs/setup.md).