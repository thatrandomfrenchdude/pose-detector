#!/usr/bin/env python3
"""
Test Basic Dependencies
Tests for basic package imports and dependencies
"""

import unittest


class TestDependencies(unittest.TestCase):
    """Test basic dependencies."""
    
    def test_opencv_import(self):
        """Test OpenCV import."""
        try:
            import cv2
            self.assertTrue(True)
        except ImportError:
            self.fail("OpenCV not available")
    
    def test_numpy_import(self):
        """Test NumPy import."""
        try:
            import numpy as np
            self.assertTrue(True)
        except ImportError:
            self.fail("NumPy not available")
    
    def test_onnx_import(self):
        """Test ONNX Runtime import."""
        try:
            import onnxruntime as ort
            self.assertTrue(True)
        except ImportError:
            self.skipTest("ONNX Runtime not available")
    
    def test_mediapipe_import(self):
        """Test MediaPipe import."""
        try:
            import mediapipe as mp
            self.assertTrue(True)
        except ImportError:
            self.skipTest("MediaPipe not available")


if __name__ == '__main__':
    unittest.main()