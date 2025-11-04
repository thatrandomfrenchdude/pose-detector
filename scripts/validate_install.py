#!/usr/bin/env python3
"""
Installation Validator
Validates that the pose detection system is properly installed and configured
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ Missing core dependency: {e}")
    sys.exit(1)


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\n📦 Checking dependencies...")
    
    deps = {
        'opencv-python': cv2,
        'numpy': np,
    }
    
    optional_deps = {}
    
    # Check ONNX Runtime
    try:
        import onnxruntime as ort
        deps['onnxruntime'] = ort
        print(f"✅ ONNX Runtime: {ort.__version__}")
        
        # Check QNN provider
        providers = ort.get_available_providers()
        if 'QNNExecutionProvider' in providers:
            print("✅ QNN Provider: Available (NPU acceleration)")
        else:
            print("⚠️  QNN Provider: Not available (CPU fallback)")
            
    except ImportError:
        print("❌ ONNX Runtime: Not available")
        optional_deps['onnxruntime'] = False
    
    # Check MediaPipe
    try:
        import mediapipe as mp
        optional_deps['mediapipe'] = mp
        print(f"✅ MediaPipe: {mp.__version__}")
    except ImportError:
        print("❌ MediaPipe: Not available")
        optional_deps['mediapipe'] = False
    
    # Core dependencies must be available
    for name, module in deps.items():
        if name not in ['onnxruntime']:  # ONNX is optional
            try:
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {name}: {version}")
            except:
                print(f"❌ {name}: Error getting version")
                return False
    
    # At least one pose detection method must be available
    if 'onnxruntime' not in deps and 'mediapipe' not in optional_deps:
        print("❌ No pose detection methods available!")
        print("   Install either onnxruntime-qnn or mediapipe")
        return False
    
    return True


def check_project_structure():
    """Check project directory structure."""
    print("\n📁 Checking project structure...")
    
    required_paths = [
        'src/',
        'src/pose_detection/',
        'src/pose_detection/detectors/',
        'src/pose_detection/utils/',
        'src/pose_detection/core/',
        'tests/',
        'docs/',
        'scripts/',
        'main.py',
        'requirements.txt'
    ]
    
    all_good = True
    for path in required_paths:
        full_path = Path(path)
        if full_path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
            all_good = False
    
    return all_good


def check_model_files():
    """Check model files."""
    print("\n🤖 Checking model files...")
    
    model_dir = Path("model")
    if not model_dir.exists():
        print("❌ Model directory not found")
        return False
    
    model_file = model_dir / "model.onnx"
    context_file = model_dir / "model_ctx.onnx"
    
    if model_file.exists():
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"✅ Original model: {model_file.name} ({size_mb:.1f} MB)")
        model_ok = True
    else:
        print(f"❌ Model not found: {model_file}")
        print("   Download HRNet model from Qualcomm AI Hub")
        model_ok = False
    
    if context_file.exists():
        size_mb = context_file.stat().st_size / 1024 / 1024
        print(f"✅ Context model: {context_file.name} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  Context model not found: {context_file.name}")
        print("   Generate with: python main.py --generate-context")
    
    return model_ok


def check_application():
    """Check application functionality."""
    print("\n🔧 Checking application...")
    
    try:
        # Try importing main components
        from pose_detection.detectors.onnx_detector import ONNX_AVAILABLE
        from pose_detection.detectors.mediapipe_detector import MEDIAPIPE_AVAILABLE
        print(f"✅ ONNX detector available: {ONNX_AVAILABLE}")
        print(f"✅ MediaPipe detector available: {MEDIAPIPE_AVAILABLE}")
        
        if not ONNX_AVAILABLE and not MEDIAPIPE_AVAILABLE:
            print("❌ No detectors available!")
            return False
        
        # Try importing main app
        from pose_detection import PoseDetectionApp
        print("✅ Main application can be imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Application check failed: {e}")
        return False


def check_cameras():
    """Check camera availability."""
    print("\n📹 Checking cameras...")
    
    try:
        from pose_detection.utils.camera_utils import get_available_cameras
        cameras = get_available_cameras()
        
        if cameras:
            print(f"✅ Available cameras: {cameras}")
            return True
        else:
            print("⚠️  No cameras detected")
            print("   Connect a camera for real-time detection")
            return True  # Not critical for validation
            
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False


def run_validation():
    """Run complete validation."""
    print("🔍 HRNet Pose Detection - Installation Validator")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Model Files", check_model_files),
        ("Application", check_application),
        ("Cameras", check_cameras),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    critical_failed = False
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20}: {status}")
        
        if result:
            passed += 1
        elif name in ["Python Version", "Dependencies", "Project Structure"]:
            critical_failed = True
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if critical_failed:
        print("\n❌ CRITICAL ISSUES FOUND")
        print("The application cannot run properly.")
        print("Please fix the failed checks and try again.")
        return False
    elif passed == len(results):
        print("\n🎉 ALL CHECKS PASSED!")
        print("The installation is complete and ready to use.")
        print("\nNext steps:")
        print("1. Generate context: python main.py --generate-context")
        print("2. Run tests: python main.py --test")
        print("3. Start detection: python main.py")
        return True
    else:
        print("\n⚠️  SOME ISSUES FOUND")
        print("The application may work but some features might be limited.")
        print("Consider fixing the failed checks for optimal performance.")
        return True


def main():
    """Main validation function."""
    success = run_validation()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()