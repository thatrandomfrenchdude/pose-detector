#!/usr/bin/env python3
"""
Camera Utilities
Utilities for camera detection and management
"""

import cv2
from typing import List


def get_available_cameras(max_cameras: int = 5) -> List[int]:
    """Get list of available camera indices."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def setup_camera(camera_id: int, width: int = 640, height: int = 480, fps: int = 30) -> cv2.VideoCapture:
    """Setup camera with optimal settings."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        available = get_available_cameras()
        raise ValueError(f"Cannot open camera {camera_id}. Available: {available}")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    return cap