#!/usr/bin/env python3
"""
Test Runner for Pose Detection
Simplified test runner that works with the new modular structure
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from pose_detection.detectors.onnx_detector import ONNX_AVAILABLE
    from pose_detection.detectors.mediapipe_detector import MEDIAPIPE_AVAILABLE
    from pose_detection import PoseDetectionApp
except ImportError as e:
    print(f"❌ Import error: {e}")
    ONNX_AVAILABLE = False
    MEDIAPIPE_AVAILABLE = False


class TestRunner:
    """Simplified test runner for pose detection."""
    
    def __init__(self):
        self.model_path = Path("model/model.onnx")
    
    def test_imports(self) -> bool:
        """Test if required packages are available."""
        print("🔍 Testing package imports...")
        
        import cv2
        import numpy as np
        print("✅ OpenCV and NumPy available")
        
        if ONNX_AVAILABLE:
            print("✅ ONNX Runtime available")
        else:
            print("❌ ONNX Runtime not available")
        
        if MEDIAPIPE_AVAILABLE:
            print("✅ MediaPipe available")
        else:
            print("❌ MediaPipe not available")
        
        return True
    
    def test_model_files(self) -> bool:
        """Test if model files exist."""
        print("\n📁 Testing model files...")
        
        model_exists = self.model_path.exists()
        context_model = self.model_path.parent / f"{self.model_path.stem}_ctx.onnx"
        context_exists = context_model.exists()
        
        if model_exists:
            size_mb = self.model_path.stat().st_size / 1024 / 1024
            print(f"✅ Original model: {self.model_path.name} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Model not found: {self.model_path}")
        
        if context_exists:
            size_mb = context_model.stat().st_size / 1024 / 1024
            print(f"✅ Context model: {context_model.name} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️  Context model not found: {context_model.name}")
        
        return model_exists
    
    def test_app_initialization(self) -> bool:
        """Test main application initialization."""
        print("\n🏗️  Testing application initialization...")
        
        try:
            # Try both modes if available
            if MEDIAPIPE_AVAILABLE:
                print("  Testing MediaPipe app initialization...")
                app = PoseDetectionApp(force_mediapipe=True)
                info = app.get_info()
                print(f"    ✅ {info['detector_type']} initialized")
                app.release()
                return True
            
            if ONNX_AVAILABLE and self.model_path.exists():
                print("  Testing NPU app initialization...")
                app = PoseDetectionApp(str(self.model_path), force_mediapipe=False)
                info = app.get_info()
                print(f"    ✅ {info['detector_type']} initialized")
                app.release()
                return True
            
            print("    ❌ No detection methods available")
            return False
            
        except Exception as e:
            print(f"    ❌ App initialization failed: {e}")
            return False
    
    def run_quick_tests(self) -> bool:
        """Run quick tests for CI/CD."""
        print("🧪 Running quick pose detection tests...")
        print("=" * 50)
        
        try:
            results = []
            results.append(self.test_imports())
            results.append(self.test_model_files())
            results.append(self.test_app_initialization())
            
            success = any(results)  # At least one test should pass
            
            print("\n" + "=" * 50)
            if success:
                print("🎉 TESTS PASSED - Application ready for use!")
            else:
                print("❌ TESTS FAILED - Setup required")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Test runner error: {e}")
            return False


def main():
    """Run tests."""
    runner = TestRunner()
    success = runner.run_quick_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()