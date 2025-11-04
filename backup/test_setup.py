#!/usr/bin/env python3
"""
Test script to verify the pose detection application setup.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'cv2',
        'mediapipe', 
        'numpy',
        'PIL'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_camera():
    """Test if camera can be accessed."""
    try:
        import cv2
        print("\nTesting camera access...")
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera 0 accessible")
            cap.release()
            return True
        else:
            print("✗ Cannot access camera 0")
            
            # Try to find other cameras
            print("Checking for other cameras...")
            found_cameras = []
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    found_cameras.append(i)
                    cap.release()
            
            if found_cameras:
                print(f"✓ Found cameras at indices: {found_cameras}")
                print(f"Try running: python main.py --camera {found_cameras[0]}")
            else:
                print("✗ No cameras found")
            
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe pose detection."""
    try:
        import mediapipe as mp
        import numpy as np
        
        print("\nTesting MediaPipe pose detection...")
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pose.process(dummy_image)
        
        pose.close()
        print("✓ MediaPipe pose detection initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Pose Detection Application - Setup Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_camera():
        tests_passed += 1
        
    if test_mediapipe():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ Setup is complete! You can now run the pose detection application.")
        print("\nTo start:")
        print("  python main.py")
        print("\nFor help:")
        print("  python main.py --help")
    else:
        print("✗ Some tests failed. Please check the installation.")
        if tests_passed < 1:
            print("Run: pip install -r requirements.txt")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)