#!/usr/bin/env python3
"""
Main Pose Detection Application
Coordinates NPU and MediaPipe detectors for optimal pose detection
"""

import cv2
import time
import logging
from typing import List

from ..detectors.onnx_detector import ONNXPoseDetector, ONNX_AVAILABLE
from ..detectors.mediapipe_detector import MediaPipePoseDetector, MEDIAPIPE_AVAILABLE
from ..utils.camera_utils import get_available_cameras, setup_camera

logger = logging.getLogger(__name__)


class PoseDetectionApp:
    """Main pose detection application."""
    
    def __init__(self, model_path: str = "model/model.onnx", force_mediapipe: bool = False):
        self.detector = None
        self.detector_type = None
        
        # Try NPU ONNX first (default for Snapdragon X Elite)
        if not force_mediapipe and ONNX_AVAILABLE:
            try:
                self.detector = ONNXPoseDetector(model_path)
                self.detector_type = "NPU-ONNX"
                logger.info("✓ NPU-accelerated pose detection ready")
            except Exception as e:
                logger.warning(f"NPU model failed: {e}")
                logger.info("Falling back to MediaPipe...")
        
        # Fallback to MediaPipe
        if self.detector is None and MEDIAPIPE_AVAILABLE:
            try:
                self.detector = MediaPipePoseDetector()
                self.detector_type = "MediaPipe"
                logger.info("✓ MediaPipe pose detection ready")
            except Exception as e:
                logger.error(f"MediaPipe failed: {e}")
        
        if self.detector is None:
            raise RuntimeError("No pose detection methods available")
    
    def process_camera(self, camera_id: int = 0):
        """Process real-time camera feed."""
        cap = setup_camera(camera_id)
        
        fps_counter = 0
        fps_start_time = time.time()
        
        print(f"Starting {self.detector_type} pose detection... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect pose
                if self.detector_type == "NPU-ONNX":
                    annotated_frame, keypoints = self.detector.detect_pose(frame)
                    num_keypoints = len(keypoints)
                else:  # MediaPipe
                    annotated_frame, landmarks = self.detector.detect_pose(frame)
                    landmarks_array = self.detector.get_landmarks_array(landmarks, frame.shape)
                    num_keypoints = len(landmarks_array)
                
                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                else:
                    fps = 0
                
                # Add info overlay
                if fps > 0:
                    cv2.putText(annotated_frame, f"{self.detector_type} - FPS: {fps:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Keypoints: {num_keypoints}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, "Press 'q' to quit", 
                          (10, annotated_frame.shape[0] - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Pose Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def process_image(self, image_path: str, output_path: str = None):
        """Process single image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect pose
        if self.detector_type == "NPU-ONNX":
            annotated_image, keypoints = self.detector.detect_pose(image)
            num_keypoints = len(keypoints)
        else:  # MediaPipe
            annotated_image, landmarks = self.detector.detect_pose(image)
            landmarks_array = self.detector.get_landmarks_array(landmarks, image.shape)
            num_keypoints = len(landmarks_array)
        
        print(f"Detected {num_keypoints} keypoints using {self.detector_type}")
        
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Result saved to: {output_path}")
        else:
            cv2.imshow('Pose Detection - Press any key to close', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated_image, num_keypoints
    
    def get_info(self) -> dict:
        """Get detector information."""
        info = {"detector_type": self.detector_type}
        
        if hasattr(self.detector, 'get_model_info'):
            info.update(self.detector.get_model_info())
        
        # Add camera info
        info["available_cameras"] = get_available_cameras()
        
        return info
    
    def release(self):
        """Release resources."""
        if self.detector:
            self.detector.release()