#!/usr/bin/env python3
"""
HRNet Pose Detection with NPU Acceleration
Entry point for pose detection application

This is a simplified entry point that imports all functionality from the src package.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_detection import PoseDetectionApp
from pose_detection.utils.model_utils import generate_npu_context
from pose_detection.utils.camera_utils import get_available_cameras

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_tests():
    """Run basic functionality tests."""
    print("🧪 Running pose detection tests...")
    
    # Import test runner
    try:
        from tests.test_runner import TestRunner
        runner = TestRunner()
        return runner.run_quick_tests()
    except ImportError:
        logger.warning("Test runner not found. Using basic tests.")
        return run_basic_tests()


def run_basic_tests():
    """Run basic functionality tests (fallback)."""
    print("\n1. Testing imports...")
    try:
        from pose_detection.detectors.onnx_detector import ONNX_AVAILABLE
        from pose_detection.detectors.mediapipe_detector import MEDIAPIPE_AVAILABLE
        
        if ONNX_AVAILABLE:
            print("✅ ONNX Runtime available")
        else:
            print("❌ ONNX Runtime not available")
        
        if MEDIAPIPE_AVAILABLE:
            print("✅ MediaPipe available")
        else:
            print("❌ MediaPipe not available")
        
        print("\n2. Testing model file...")
        model_path = Path("model/model.onnx")
        if model_path.exists():
            print(f"✅ Model found: {model_path}")
        else:
            print(f"❌ Model not found: {model_path}")
            return False
        
        print("\n3. Testing application initialization...")
        app = PoseDetectionApp()
        info = app.get_info()
        print(f"✅ {info['detector_type']} initialized successfully")
        app.release()
        
        print("\n✅ Basic tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="HRNet Pose Detection with NPU Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time NPU pose detection (default)
  python main.py

  # Process image with NPU
  python main.py --image photo.jpg --output result.jpg

  # Use MediaPipe fallback
  python main.py --mediapipe

  # Generate NPU context for faster startup (one-time)
  python main.py --generate-context

  # Run tests
  python main.py --test
        """
    )
    
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--image", help="Process single image file")
    parser.add_argument("--output", help="Output path for processed image")
    parser.add_argument("--mediapipe", action="store_true", help="Force MediaPipe mode")
    parser.add_argument("--model", default="model/model.onnx", help="ONNX model path")
    parser.add_argument("--generate-context", action="store_true", help="Generate NPU context model")
    parser.add_argument("--test", action="store_true", help="Run functionality tests")
    parser.add_argument("--info", action="store_true", help="Show detector information")
    
    args = parser.parse_args()
    
    try:
        if args.test:
            run_tests()
        elif args.generate_context:
            generate_npu_context(args.model)
        elif args.info:
            app = PoseDetectionApp(args.model, args.mediapipe)
            info = app.get_info()
            print("\n📊 Pose Detection Information")
            print("=" * 40)
            for key, value in info.items():
                print(f"{key}: {value}")
            app.release()
        elif args.image:
            app = PoseDetectionApp(args.model, args.mediapipe)
            app.process_image(args.image, args.output)
            app.release()
        else:
            app = PoseDetectionApp(args.model, args.mediapipe)
            app.process_camera(args.camera)
            app.release()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()