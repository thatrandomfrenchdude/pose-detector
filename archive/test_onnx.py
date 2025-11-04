#!/usr/bin/env python3
"""
Test script for ONNX model integration.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_onnx_model():
    """Test if ONNX model can be loaded and run inference."""
    try:
        from onnx_pose_detector import ONNXPoseDetector
        import cv2
        
        model_path = "model/model.onnx"
        
        if not Path(model_path).exists():
            logger.error(f"ONNX model not found: {model_path}")
            return False
        
        logger.info(f"Testing ONNX model: {model_path}")
        
        # Initialize detector
        detector = ONNXPoseDetector(model_path, provider="cpu")
        
        # Get model info
        model_info = detector.get_model_info()
        logger.info(f"Model loaded with provider: {model_info.get('providers', ['unknown'])[0]}")
        logger.info(f"Input shape: {model_info.get('input_shape')}")
        logger.info(f"Output shape: {model_info.get('output_shape')}")
        
        # Test with dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        logger.info("Running test inference...")
        annotated_image, keypoints = detector.detect_pose(dummy_image)
        
        logger.info(f"✓ ONNX inference successful - detected {len(keypoints)} keypoints")
        
        detector.release()
        return True
        
    except Exception as e:
        logger.error(f"✗ ONNX model test failed: {e}")
        return False

def test_qnn_provider():
    """Test if QNN provider is available."""
    try:
        import onnxruntime as ort
        
        available_providers = ort.get_available_providers()
        qnn_available = "QNNExecutionProvider" in available_providers
        
        logger.info(f"Available providers: {available_providers}")
        logger.info(f"QNN Provider available: {qnn_available}")
        
        if qnn_available:
            logger.info("✓ QNN Execution Provider is available for NPU acceleration")
            return True
        else:
            logger.warning("✗ QNN Execution Provider not available - NPU acceleration disabled")
            logger.info("  Note: QNN EP requires onnxruntime-qnn package and compatible hardware")
            return False
            
    except Exception as e:
        logger.error(f"✗ QNN provider test failed: {e}")
        return False

def test_npu_model():
    """Test ONNX model with NPU acceleration."""
    try:
        from onnx_pose_detector import ONNXPoseDetector
        import cv2
        
        model_path = "model/model.onnx"
        
        if not Path(model_path).exists():
            logger.error(f"ONNX model not found: {model_path}")
            return False
        
        logger.info(f"Testing ONNX model with NPU: {model_path}")
        
        # Initialize detector with NPU preference
        detector = ONNXPoseDetector(model_path, provider="npu")
        
        # Get model info
        model_info = detector.get_model_info()
        current_provider = model_info.get('providers', ['unknown'])[0]
        logger.info(f"Model loaded with provider: {current_provider}")
        
        if "QNN" in current_provider:
            logger.info("✓ NPU acceleration active")
        else:
            logger.warning(f"⚠ Fallback to {current_provider} - NPU not used")
        
        # Test with dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        logger.info("Running NPU inference test...")
        annotated_image, keypoints = detector.detect_pose(dummy_image)
        
        logger.info(f"✓ NPU inference successful - detected {len(keypoints)} keypoints")
        
        detector.release()
        return True
        
    except Exception as e:
        logger.error(f"✗ NPU model test failed: {e}")
        logger.info("  This may be normal if NPU hardware is not available")
        return False
    """Test the hybrid detector with both ONNX and MediaPipe."""
    try:
        from main import HybridPoseDetector
        import cv2
        
        logger.info("Testing hybrid detector...")
        
        # Test with ONNX first
        detector = HybridPoseDetector(
            onnx_model_path="model/model.onnx",
            use_onnx=True
        )
        
        detector_info = detector.get_detector_info()
        logger.info(f"Current detector: {detector_info['current_detector']}")
        logger.info(f"ONNX available: {detector_info['onnx_available']}")
        logger.info(f"MediaPipe available: {detector_info['mediapipe_available']}")
        
        # Test inference
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_image, landmarks = detector.detect_pose(dummy_image)
        
        landmarks_array = detector.get_landmarks_array(landmarks, dummy_image.shape)
        logger.info(f"✓ Hybrid detector test successful - {len(landmarks_array)} landmarks")
        
        detector.release()
        return True
        
    except Exception as e:
        logger.error(f"✗ Hybrid detector test failed: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'cv2',
        'mediapipe', 
        'numpy',
        'onnxruntime',
        'PIL'
    ]
    
    logger.info("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError as e:
            logger.error(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        logger.info("✓ All packages imported successfully!")
        return True

def test_hybrid_detector():
    """Test the hybrid detector with both ONNX and MediaPipe."""
    try:
        from main import HybridPoseDetector
        import cv2
        
        logger.info("Testing hybrid detector...")
        
        # Test with ONNX first
        detector = HybridPoseDetector(
            onnx_model_path="model/model.onnx",
            use_onnx=True,
            provider_preference="npu"  # Try NPU first
        )
        
        detector_info = detector.get_detector_info()
        logger.info(f"Current detector: {detector_info['current_detector']}")
        logger.info(f"ONNX available: {detector_info['onnx_available']}")
        logger.info(f"MediaPipe available: {detector_info['mediapipe_available']}")
        
        # Test inference
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated_image, landmarks = detector.detect_pose(dummy_image)
        
        landmarks_array = detector.get_landmarks_array(landmarks, dummy_image.shape)
        logger.info(f"✓ Hybrid detector test successful - {len(landmarks_array)} landmarks")
        
        detector.release()
        return True
        
    except Exception as e:
        logger.error(f"✗ Hybrid detector test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ONNX Pose Detection - NPU Integration Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    if test_imports():
        tests_passed += 1
    
    if test_qnn_provider():
        tests_passed += 1
    
    if test_onnx_model():
        tests_passed += 1
    
    if test_npu_model():
        tests_passed += 1
        
    if test_hybrid_detector():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed >= 3:  # Allow some NPU tests to fail on incompatible hardware
        print("✓ Core functionality working! NPU support status:")
        print("  python main.py --model-info")
        print("  python main.py --use-npu    # Try NPU acceleration")
        print("  python main.py --use-cpu    # Force CPU execution")
    else:
        print("✗ Some core tests failed. Check the error messages above.")
        if tests_passed == 0:
            print("Run: pip install -r requirements.txt")
    
    return tests_passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)