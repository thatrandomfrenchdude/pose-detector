# HRNet Pose Detection with NPU Acceleration

Real-time human pose estimation using **Qualcomm Snapdragon X Elite NPU** with MediaPipe fallback support.

## 🚀 Quick Start
1. Follow the [Setup Guide](./docs/setup.md) to prepare your environment.
2. Use the application:
   ```bash
   # Process an image
   python main.py --image vitruvian-man.jpg --output result.jpg

   # Run with camera stream
   python main.py

   # Run camera stream with MediaPipe
   python main.py --mediapipe
   ```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Setup Guide](docs/setup.md)** - Detailed installation instructions
- **[Usage Guide](docs/usage.md)** - How to use the application and its features

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Update documentation in `docs/`
6. Submit pull request

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
