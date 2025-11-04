import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import logging
from pathlib import Path
from onnx_pose_detector import ONNXPoseDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridPoseDetector:
    """
    Hybrid pose detection using ONNX model (primary) with MediaPipe fallback.
    """
    
    def __init__(self, onnx_model_path: str = None, use_onnx: bool = True, provider_preference: str = "npu"):
        """
        Initialize the hybrid pose detector.
        
        Args:
            onnx_model_path: Path to ONNX model file
            use_onnx: Whether to try ONNX model first
            provider_preference: Preferred execution provider ("npu", "cpu", "gpu")
        """
        self.use_onnx = use_onnx
        self.provider_preference = provider_preference
        self.onnx_detector = None
        self.mediapipe_detector = None
        self.current_detector = None
        
        # Try to load ONNX model first
        if use_onnx and onnx_model_path:
            try:
                logger.info(f"Loading ONNX model from: {onnx_model_path}")
                logger.info(f"Target execution provider: {provider_preference.upper()}")
                self.onnx_detector = ONNXPoseDetector(onnx_model_path, provider=provider_preference)
                self.current_detector = "onnx"
                logger.info("✓ ONNX model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}")
                logger.info("Falling back to MediaPipe...")
        
        # Load MediaPipe as fallback or primary
        if self.onnx_detector is None:
            try:
                logger.info("Loading MediaPipe model...")
                self.mediapipe_detector = MediaPipePoseDetector()
                self.current_detector = "mediapipe"
                logger.info("✓ MediaPipe model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MediaPipe model: {e}")
                raise RuntimeError("No pose detection models could be loaded")
    
    def detect_pose(self, image):
        """
        Detect pose using the available detector.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            tuple: (processed_image, landmarks_or_keypoints)
        """
        if self.current_detector == "onnx" and self.onnx_detector:
            try:
                annotated_image, keypoints = self.onnx_detector.detect_pose(image)
                return annotated_image, keypoints
            except Exception as e:
                logger.warning(f"ONNX detection failed: {e}, falling back to MediaPipe")
                if self.mediapipe_detector is None:
                    self.mediapipe_detector = MediaPipePoseDetector()
                self.current_detector = "mediapipe"
        
        if self.current_detector == "mediapipe" and self.mediapipe_detector:
            return self.mediapipe_detector.detect_pose(image)
        
        raise RuntimeError("No working pose detector available")
    
    def get_landmarks_array(self, landmarks_or_keypoints, image_shape):
        """
        Convert landmarks to numpy array with pixel coordinates.
        
        Args:
            landmarks_or_keypoints: MediaPipe landmarks or ONNX keypoints
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            numpy.ndarray: Array of landmark coordinates [(x, y), ...]
        """
        if self.current_detector == "onnx":
            # ONNX keypoints are already in pixel coordinates
            return np.array(landmarks_or_keypoints) if landmarks_or_keypoints else np.array([])
        
        elif self.current_detector == "mediapipe":
            return self.mediapipe_detector.get_landmarks_array(landmarks_or_keypoints, image_shape)
        
        return np.array([])
    
    def get_detector_info(self):
        """Get information about the current detector."""
        info = {
            "current_detector": self.current_detector,
            "onnx_available": self.onnx_detector is not None,
            "mediapipe_available": self.mediapipe_detector is not None
        }
        
        if self.onnx_detector:
            info["onnx_model_info"] = self.onnx_detector.get_model_info()
        
        return info
    
    def release(self):
        """Release resources."""
        if self.onnx_detector:
            self.onnx_detector.release()
        if self.mediapipe_detector:
            self.mediapipe_detector.release()


class MediaPipePoseDetector:
    """
    MediaPipe-based pose detection (fallback implementation).
    """
    
    def __init__(self):
        """Initialize the MediaPipe pose detection model."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_pose(self, image):
        """
        Detect pose landmarks in an image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            tuple: (processed_image, landmarks)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform pose detection
        results = self.pose.process(rgb_image)
        
        # Convert back to BGR for OpenCV display
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks if detected
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2
                )
            )
        
        return annotated_image, results.pose_landmarks
    
    def get_landmarks_array(self, landmarks, image_shape):
        """
        Convert landmarks to numpy array with pixel coordinates.
        
        Args:
            landmarks: MediaPipe landmarks object
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            numpy.ndarray: Array of landmark coordinates [(x, y), ...]
        """
        if landmarks is None:
            return np.array([])
        
        height, width = image_shape[:2]
        landmarks_array = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks_array.append([x, y])
        
        return np.array(landmarks_array)
    
    def release(self):
        """Release resources."""
        self.pose.close()


class CameraProcessor:
    """
    Handles camera input and video processing for pose detection.
    """
    
    def __init__(self, camera_id=0, onnx_model_path=None, use_onnx=True, provider_preference="npu"):
        """
        Initialize camera processor.
        
        Args:
            camera_id: Camera index (default: 0)
            onnx_model_path: Path to ONNX model file
            use_onnx: Whether to use ONNX model (default: True)
            provider_preference: Preferred execution provider ("npu", "cpu", "gpu")
        """
        self.camera_id = camera_id
        self.cap = None
        self.pose_detector = HybridPoseDetector(
            onnx_model_path=onnx_model_path,
            use_onnx=use_onnx,
            provider_preference=provider_preference
        )
        
        # Log which detector is being used
        detector_info = self.pose_detector.get_detector_info()
        detector_name = detector_info['current_detector'].upper()
        if detector_name == "ONNX" and detector_info.get('onnx_model_info'):
            providers = detector_info['onnx_model_info'].get('providers', [])
            if providers:
                provider_name = providers[0].replace('ExecutionProvider', '')
                logger.info(f"Using {detector_name} with {provider_name} for pose detection")
            else:
                logger.info(f"Using {detector_name} for pose detection")
        else:
            logger.info(f"Using {detector_name} for pose detection")
        
    def check_available_cameras(self, max_cameras=5):
        """
        Check for available cameras.
        
        Args:
            max_cameras: Maximum number of cameras to check
            
        Returns:
            list: Available camera indices
        """
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def start_camera(self):
        """Start the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            available = self.check_available_cameras()
            raise ValueError(f"Cannot open camera {self.camera_id}. Available cameras: {available}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {self.camera_id} started successfully")
        return True
    
    def process_video_stream(self):
        """
        Main video processing loop with pose detection.
        """
        if not self.start_camera():
            return
        
        fps_counter = 0
        fps_start_time = time.time()
        
        print("Starting pose detection... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect pose
                annotated_frame, landmarks = self.pose_detector.detect_pose(frame)
                
                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                else:
                    fps = 0
                
                # Add detector info and FPS text to frame
                detector_name = self.pose_detector.current_detector.upper()
                if fps > 0:
                    cv2.putText(annotated_frame, f"{detector_name} - FPS: {fps:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, f"{detector_name}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add instruction text
                cv2.putText(annotated_frame, "Press 'q' to quit", 
                          (10, annotated_frame.shape[0] - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow('Pose Detection', annotated_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def process_image(self, image_path, output_path=None):
        """
        Process a single image for pose detection.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect pose
        annotated_image, landmarks = self.pose_detector.detect_pose(image)
        
        # Get landmark coordinates
        landmarks_array = self.pose_detector.get_landmarks_array(landmarks, image.shape)
        
        print(f"Detected {len(landmarks_array)} landmarks")
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Result saved to: {output_path}")
        else:
            cv2.imshow('Pose Detection - Press any key to close', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return landmarks_array
    
    def cleanup(self):
        """Release all resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose_detector.release()
        print("Resources released")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Hybrid Pose Detection Application (ONNX + MediaPipe)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--image", type=str, help="Process single image instead of camera stream")
    parser.add_argument("--output", type=str, help="Output path for processed image")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    parser.add_argument("--model-info", action="store_true", help="Show model information")
    parser.add_argument("--use-mediapipe", action="store_true", help="Force use MediaPipe only")
    parser.add_argument("--use-npu", action="store_true", help="Use NPU (QNN) for ONNX model")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU execution for ONNX model")
    parser.add_argument("--onnx-model", type=str, default="model/model.onnx", 
                       help="Path to ONNX model file (default: model/model.onnx)")
    
    args = parser.parse_args()
    
    # Determine execution provider preference
    if args.use_npu:
        provider_preference = "npu"
    elif args.use_cpu:
        provider_preference = "cpu"
    else:
        provider_preference = "npu"  # Default to NPU for better performance
    
    # Determine ONNX model path
    onnx_model_path = args.onnx_model if not args.use_mediapipe else None
    
    processor = CameraProcessor(
        camera_id=args.camera,
        onnx_model_path=onnx_model_path,
        use_onnx=not args.use_mediapipe,
        provider_preference=provider_preference
    )
    
    try:
        if args.model_info:
            # Show model information
            info = processor.pose_detector.get_detector_info()
            print("\n=== Model Information ===")
            print(f"Current detector: {info['current_detector'].upper()}")
            print(f"ONNX available: {info['onnx_available']}")
            print(f"MediaPipe available: {info['mediapipe_available']}")
            
            if info.get('onnx_model_info'):
                onnx_info = info['onnx_model_info']
                if onnx_info['status'] == 'loaded':
                    print(f"\nONNX Model Details:")
                    print(f"  Model path: {onnx_info['model_path']}")
                    print(f"  Providers: {onnx_info['providers']}")
                    print(f"  Input shape: {onnx_info['input_shape']}")
                    print(f"  Output shape: {onnx_info['output_shape']}")
                    print(f"  Input type: {onnx_info['input_type']}")
            return
        
        if args.list_cameras:
            # List available cameras
            cameras = processor.check_available_cameras()
            print(f"Available cameras: {cameras}")
            return
        
        if args.image:
            # Process single image
            print(f"Processing image: {args.image}")
            landmarks = processor.process_image(args.image, args.output)
            print(f"Processing complete. Found {len(landmarks)} pose landmarks.")
        else:
            # Process video stream
            processor.process_video_stream()
            
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Application error: {e}")
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main()
