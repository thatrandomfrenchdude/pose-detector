#!/usr/bin/env python3
"""
MediaPipe Pose Detector
MediaPipe pose detection fallback implementation
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available - fallback mode disabled")


class MediaPipePoseDetector:
    """MediaPipe pose detection fallback."""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_pose(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """Detect pose using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        return annotated_image, results.pose_landmarks
    
    def get_landmarks_array(self, landmarks, image_shape) -> np.ndarray:
        """Convert landmarks to coordinate array."""
        if landmarks is None:
            return np.array([])
        
        height, width = image_shape[:2]
        coords = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            coords.append([x, y])
        
        return np.array(coords)
    
    def release(self):
        """Release resources."""
        self.pose.close()