# Setup Guide

Complete setup instructions for HRNet Pose Detection with NPU Acceleration.

## System Requirements
- **Python**: Version 3.8 or higher
- **Camera**: Built-in or external camera for real-time detection

## Setup Guide

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

   # For MacOs/Linux/x86 Windows, use standard ONNX Runtime
   pip install onnxruntime
   ```
4. Download the HRNet model from [Qualcomm AI Hub](https://aihub.qualcomm.com/) and place it as `model/model.onnx`.
5. Validate the installation and/or run tests:
   ```bash
   python scripts/validate_install.py

   # Or run built-in tests
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
6. (Optional) Generate the context model for faster startup:
   ```bash
   python main.py --generate-context
   ```
   This is a one-time operation that takes ~5-10 seconds and reduces future startup times by 50x.

## Common Setup Issues

### ONNXRuntime Collision
Onnxruntime and onnxruntime-qnn can conflict. If you encounter issues, uninstall both and reinstall only the required package:
```bash
# Reinstall dependencies
pip uninstall -y onnxruntime onnxruntime-qnn

# Windows on Snapdragon
pip install onnxruntime-qnn

# macOS/Linux/x86 Windows
pip install onnxruntime
```
### Model Not Found
Ensure the model file is correctly placed at `model/model.onnx`. If missing, download it from [Qualcomm AI Hub](https://aihub.qualcomm.com/).