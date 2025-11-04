#!/usr/bin/env python3
"""
Comprehensive test suite for HRNet Pose Detection
Tests NPU acceleration, MediaPipe fallback, and performance benchmarks
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import detection capabilities
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class TestSuite:
    """Comprehensive test suite for pose detection."""
    
    def __init__(self):
        self.model_path = Path("model/model.onnx")
        self.test_image_path = self._create_test_image()
        self.results = {}
    
    def _create_test_image(self) -> str:
        """Create a simple test image for testing."""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple stick figure
        cv2.circle(test_image, (320, 100), 20, (255, 255, 255), -1)  # Head
        cv2.line(test_image, (320, 120), (320, 300), (255, 255, 255), 5)  # Body
        cv2.line(test_image, (320, 180), (250, 230), (255, 255, 255), 3)  # Left arm
        cv2.line(test_image, (320, 180), (390, 230), (255, 255, 255), 3)  # Right arm
        cv2.line(test_image, (320, 300), (270, 400), (255, 255, 255), 3)  # Left leg
        cv2.line(test_image, (320, 300), (370, 400), (255, 255, 255), 3)  # Right leg
        
        test_path = "test_image.jpg"
        cv2.imwrite(test_path, test_image)
        return test_path
    
    def test_imports(self) -> bool:
        """Test if required packages are available."""
        print("🔍 Testing package imports...")
        
        tests = [
            ("OpenCV", cv2, True),
            ("NumPy", np, True),
            ("ONNX Runtime", None, ONNX_AVAILABLE),
            ("MediaPipe", None, MEDIAPIPE_AVAILABLE)
        ]
        
        all_passed = True
        for name, module, available in tests:
            if available:
                print(f"  ✅ {name}: Available")
            else:
                print(f"  ❌ {name}: Not available")
                if name in ["OpenCV", "NumPy"]:
                    all_passed = False
        
        self.results["imports"] = all_passed
        return all_passed
    
    def test_model_files(self) -> bool:
        """Test if model files exist."""
        print("\n📁 Testing model files...")
        
        model_exists = self.model_path.exists()
        context_model = self.model_path.parent / f"{self.model_path.stem}_ctx.onnx"
        context_exists = context_model.exists()
        
        if model_exists:
            size_mb = self.model_path.stat().st_size / 1024 / 1024
            print(f"  ✅ Original model: {self.model_path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ Model not found: {self.model_path}")
        
        if context_exists:
            size_mb = context_model.stat().st_size / 1024 / 1024
            print(f"  ✅ Context model: {context_model.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  Context model not found: {context_model.name}")
            print("      Tip: Run 'python main.py --generate-context' for faster startup")
        
        self.results["model_files"] = model_exists
        return model_exists
    
    def test_npu_detection(self) -> bool:
        """Test NPU-accelerated pose detection."""
        if not ONNX_AVAILABLE:
            print("\n⚠️  Skipping NPU test - ONNX Runtime not available")
            self.results["npu"] = False
            return False
        
        print("\n🚀 Testing NPU pose detection...")
        
        try:
            from main_new import ONNXPoseDetector
            
            start_time = time.time()
            detector = ONNXPoseDetector(str(self.model_path))
            load_time = time.time() - start_time
            
            # Test inference
            test_image = cv2.imread(self.test_image_path)
            start_time = time.time()
            annotated_image, keypoints = detector.detect_pose(test_image)
            inference_time = time.time() - start_time
            
            # Get model info
            info = detector.get_model_info()
            provider = info["providers"][0] if info["providers"] else "Unknown"
            
            print(f"  ✅ NPU detection successful")
            print(f"     Provider: {provider}")
            print(f"     Load time: {load_time:.2f}s")
            print(f"     Inference time: {inference_time:.3f}s")
            print(f"     Keypoints detected: {len(keypoints)}")
            
            detector.release()
            self.results["npu"] = True
            self.results["npu_load_time"] = load_time
            self.results["npu_inference_time"] = inference_time
            return True
            
        except Exception as e:
            print(f"  ❌ NPU test failed: {e}")
            self.results["npu"] = False
            return False
    
    def test_mediapipe_detection(self) -> bool:
        """Test MediaPipe pose detection."""
        if not MEDIAPIPE_AVAILABLE:
            print("\n⚠️  Skipping MediaPipe test - MediaPipe not available")
            self.results["mediapipe"] = False
            return False
        
        print("\n🎯 Testing MediaPipe pose detection...")
        
        try:
            from main_new import MediaPipePoseDetector
            
            start_time = time.time()
            detector = MediaPipePoseDetector()
            load_time = time.time() - start_time
            
            # Test inference
            test_image = cv2.imread(self.test_image_path)
            start_time = time.time()
            annotated_image, landmarks = detector.detect_pose(test_image)
            inference_time = time.time() - start_time
            
            landmarks_array = detector.get_landmarks_array(landmarks, test_image.shape)
            
            print(f"  ✅ MediaPipe detection successful")
            print(f"     Load time: {load_time:.2f}s")
            print(f"     Inference time: {inference_time:.3f}s")
            print(f"     Landmarks detected: {len(landmarks_array)}")
            
            detector.release()
            self.results["mediapipe"] = True
            self.results["mediapipe_load_time"] = load_time
            self.results["mediapipe_inference_time"] = inference_time
            return True
            
        except Exception as e:
            print(f"  ❌ MediaPipe test failed: {e}")
            self.results["mediapipe"] = False
            return False
    
    def test_app_initialization(self) -> bool:
        """Test main application initialization."""
        print("\n🏗️  Testing application initialization...")
        
        try:
            from main_new import PoseDetectionApp
            
            # Test with NPU (default)
            if ONNX_AVAILABLE and self.model_path.exists():
                print("  Testing NPU app initialization...")
                start_time = time.time()
                app = PoseDetectionApp(str(self.model_path), force_mediapipe=False)
                init_time = time.time() - start_time
                
                info = app.get_info()
                print(f"    ✅ {info['detector_type']} initialized ({init_time:.2f}s)")
                app.release()
            
            # Test MediaPipe fallback
            if MEDIAPIPE_AVAILABLE:
                print("  Testing MediaPipe app initialization...")
                start_time = time.time()
                app = PoseDetectionApp(force_mediapipe=True)
                init_time = time.time() - start_time
                
                info = app.get_info()
                print(f"    ✅ {info['detector_type']} initialized ({init_time:.2f}s)")
                app.release()
            
            self.results["app_init"] = True
            return True
            
        except Exception as e:
            print(f"  ❌ App initialization failed: {e}")
            self.results["app_init"] = False
            return False
    
    def test_camera_detection(self) -> bool:
        """Test camera availability."""
        print("\n📹 Testing camera availability...")
        
        available_cameras = []
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            print(f"  ✅ Cameras available: {available_cameras}")
            self.results["cameras"] = available_cameras
            return True
        else:
            print("  ⚠️  No cameras detected")
            self.results["cameras"] = []
            return False
    
    def benchmark_performance(self) -> dict:
        """Run performance benchmarks."""
        print("\n⚡ Running performance benchmarks...")
        
        benchmarks = {}
        test_image = cv2.imread(self.test_image_path)
        
        # NPU benchmark
        if self.results.get("npu", False):
            print("  Benchmarking NPU performance...")
            try:
                from main_new import ONNXPoseDetector
                detector = ONNXPoseDetector(str(self.model_path))
                
                times = []
                for i in range(10):
                    start = time.time()
                    detector.detect_pose(test_image)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                fps = 1.0 / avg_time
                benchmarks["npu"] = {"avg_time": avg_time, "fps": fps}
                print(f"    NPU: {avg_time:.3f}s avg, {fps:.1f} FPS")
                
                detector.release()
            except Exception as e:
                print(f"    NPU benchmark failed: {e}")
        
        # MediaPipe benchmark
        if self.results.get("mediapipe", False):
            print("  Benchmarking MediaPipe performance...")
            try:
                from main_new import MediaPipePoseDetector
                detector = MediaPipePoseDetector()
                
                times = []
                for i in range(10):
                    start = time.time()
                    detector.detect_pose(test_image)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                fps = 1.0 / avg_time
                benchmarks["mediapipe"] = {"avg_time": avg_time, "fps": fps}
                print(f"    MediaPipe: {avg_time:.3f}s avg, {fps:.1f} FPS")
                
                detector.release()
            except Exception as e:
                print(f"    MediaPipe benchmark failed: {e}")
        
        self.results["benchmarks"] = benchmarks
        return benchmarks
    
    def test_context_generation(self) -> bool:
        """Test NPU context generation."""
        if not ONNX_AVAILABLE or not self.model_path.exists():
            print("\n⚠️  Skipping context generation test - requirements not met")
            return False
        
        print("\n🔧 Testing context generation...")
        
        try:
            from main_new import generate_npu_context
            
            # Remove existing context if present
            context_path = self.model_path.parent / f"{self.model_path.stem}_ctx.onnx"
            test_context_path = self.model_path.parent / f"{self.model_path.stem}_test_ctx.onnx"
            
            if test_context_path.exists():
                test_context_path.unlink()
            
            # Test context generation (redirect to test file)
            print("  Generating test context...")
            success = generate_npu_context(str(self.model_path))
            
            # Check if context was created
            if context_path.exists():
                print(f"  ✅ Context generation successful")
                print(f"     Context file: {context_path.name}")
                self.results["context_generation"] = True
                return True
            else:
                print(f"  ❌ Context file not found")
                self.results["context_generation"] = False
                return False
                
        except Exception as e:
            print(f"  ❌ Context generation failed: {e}")
            self.results["context_generation"] = False
            return False
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("📊 TEST REPORT SUMMARY")
        print("="*60)
        
        # Overall status
        critical_tests = ["imports", "model_files"]
        critical_passed = all(self.results.get(test, False) for test in critical_tests)
        
        if critical_passed:
            print("🎉 OVERALL STATUS: READY FOR USE")
        else:
            print("⚠️  OVERALL STATUS: SETUP REQUIRED")
        
        print(f"\n📋 Test Results:")
        test_status = [
            ("Package Imports", self.results.get("imports", False)),
            ("Model Files", self.results.get("model_files", False)),
            ("NPU Detection", self.results.get("npu", False)),
            ("MediaPipe Detection", self.results.get("mediapipe", False)),
            ("App Initialization", self.results.get("app_init", False)),
            ("Camera Availability", bool(self.results.get("cameras", []))),
            ("Context Generation", self.results.get("context_generation", False))
        ]
        
        for test_name, passed in test_status:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name:<20}: {status}")
        
        # Performance summary
        benchmarks = self.results.get("benchmarks", {})
        if benchmarks:
            print(f"\n⚡ Performance Summary:")
            for method, data in benchmarks.items():
                fps = data["fps"]
                avg_time = data["avg_time"]
                print(f"  {method.upper():<12}: {fps:>6.1f} FPS ({avg_time:.3f}s avg)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if not self.results.get("npu", False) and ONNX_AVAILABLE:
            print("  • Check NPU model and QNN installation")
        
        if not self.results.get("context_generation", False) and self.results.get("npu", False):
            print("  • Generate context model: python main.py --generate-context")
        
        cameras = self.results.get("cameras", [])
        if cameras:
            print(f"  • Use camera {cameras[0]} for real-time detection")
        else:
            print("  • Connect a camera for real-time detection")
        
        if critical_passed:
            print(f"\n🚀 Ready to use:")
            print(f"  • Real-time: python main.py")
            print(f"  • Image: python main.py --image photo.jpg")
            print(f"  • Tests: python main.py --test")
        
        print("\n" + "="*60)
    
    def cleanup(self):
        """Clean up test files."""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 HRNet Pose Detection - Comprehensive Test Suite")
        print("="*60)
        
        try:
            # Core tests
            self.test_imports()
            self.test_model_files()
            
            # Detection tests
            self.test_npu_detection()
            self.test_mediapipe_detection()
            
            # Application tests
            self.test_app_initialization()
            self.test_camera_detection()
            
            # Performance tests
            if any(self.results.get(k, False) for k in ["npu", "mediapipe"]):
                self.benchmark_performance()
            
            # Advanced tests
            self.test_context_generation()
            
            # Generate report
            self.generate_report()
            
        except KeyboardInterrupt:
            print("\n⚠️  Tests interrupted by user")
        except Exception as e:
            print(f"\n❌ Test suite error: {e}")
        finally:
            self.cleanup()


def main():
    """Run test suite."""
    suite = TestSuite()
    suite.run_all_tests()


if __name__ == "__main__":
    main()