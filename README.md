# HRNet Pose Detection with NPU Acceleration

Real-time human pose estimation using **Qualcomm Snapdragon X Elite NPU** with MediaPipe fallback support.

## 🚀 Quick Start
1. Clone the Repository and navigate into it:
   ```bash
   git clone https://github.com/thatrandomfrenchdude/pose-detector.git
   cd pose-detector
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv

   # Windows PowerShell
   venv\Scripts\Activate.ps1

   # macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   # Install base dependencies
   pip install opencv-python numpy mediapipe

   # Install ONNX Runtime according to your machine
   # For Windows on Snapdragon X Elite
   pip install onnxruntime-qnn

   # For other systems, use standard ONNX Runtime
   pip install onnxruntime
   ```
4. Download the HRNet model from [Qualcomm AI Hub](https://aihub.qualcomm.com/) and place it as `model/model.onnx`.
5. Validate the installation and/or run tests:
   ```bash
   python scripts/validate_install.py

   # Or run built-in tests
   python main.py --test
   ```
6. (Optional) Generate the context model for faster startup:
   ```bash
   python main.py --generate-context
   ```
7. Use the application:
   ```bash
   # Run real-time NPU pose detection (default)
   python main.py

   # Process image
   python main.py --image photo.jpg --output result.jpg

   # Run tests
   python main.py --test
   ```

## 🎮 Extended Usage

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
```

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

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Update documentation in `docs/`
6. Submit pull request

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Setup Guide](docs/setup.md)** - Detailed installation instructions

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
