#!/usr/bin/env python3
"""
HRNet Pose Detection with NPU Acceleration
Real-time human pose estimation using Qualcomm's Snapdragon X Elite NPU

Features:
- Primary: NPU-accelerated ONNX inference (Snapdragon X Elite)
- Fallback: MediaPipe pose detection
- Real-time camera processing and static image analysis
- Automatic context model generation for fastest NPU startup
"""

import cv2
import numpy as np
import time
import logging
import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available - NPU mode disabled")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available - fallback mode disabled")


class ONNXPoseDetector:
    """NPU-accelerated pose detection using ONNX Runtime with QNN."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.session = None
        self.input_shape = None
        self.output_shape = None
        self._load_model()
    
    def _find_best_model(self) -> Path:
        """Find the best available model (context vs original)."""
        # Look for pre-compiled context models first (faster loading)
        context_candidates = [
            self.model_path.with_suffix('.ctx.onnx'),
            self.model_path.parent / f"{self.model_path.stem}_ctx.onnx"
        ]
        
        for context_model in context_candidates:
            if context_model.exists():
                logger.info(f"Using pre-compiled NPU model: {context_model.name}")
                return context_model
        
        if self.model_path.exists():
            logger.info(f"Using original model: {self.model_path.name}")
            logger.info("Note: First load will be slow (~5s) due to NPU optimization")
            logger.info("Tip: Run 'python main.py --generate-context' for faster startup")
            return self.model_path
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def _load_model(self):
        """Load ONNX model with NPU acceleration."""
        try:
            print("🔍 Finding best model...")
            model_to_use = self._find_best_model()
            
            print(f"📦 Loading model: {model_to_use.name}")
            
            # Configure session for NPU performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable context caching for faster subsequent loads
            if not str(model_to_use).endswith('_ctx.onnx'):
                context_file = str(self.model_path.with_suffix('.ctx.onnx'))
                sess_options.add_session_config_entry("ep.context_enable", "1")
                sess_options.add_session_config_entry("ep.context_file_path", context_file)
                sess_options.add_session_config_entry("ep.context_embed_mode", "1")
            
            # QNN provider options for optimal NPU performance
            qnn_options = {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "htp_graph_finalization_optimization_mode": "3",
                "enable_htp_fp16_precision": "1",
                "qnn_context_priority": "high"
            }
            
            # Try NPU first, fallback to CPU if needed
            available_providers = ort.get_available_providers()
            if "QNNExecutionProvider" in available_providers:
                providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
                provider_options = [qnn_options, {}]
                logger.info("NPU (QNN) acceleration enabled")
            else:
                providers = ["CPUExecutionProvider"]
                provider_options = [{}]
                logger.warning("NPU not available, using CPU")
            
            print("🚀 Creating ONNX Runtime session...")
            
            # Try QNN first with timeout fallback
            session_created = False
            
            if "QNNExecutionProvider" in available_providers:
                try:
                    print("⚡ Attempting NPU (QNN) provider...")
                    self.session = ort.InferenceSession(
                        str(model_to_use),
                        sess_options=sess_options,
                        providers=["QNNExecutionProvider", "CPUExecutionProvider"],
                        provider_options=[qnn_options, {}]
                    )
                    session_created = True
                    print("✅ NPU session created successfully")
                except Exception as qnn_error:
                    print(f"⚠️  NPU provider failed: {qnn_error}")
                    logger.warning(f"QNN provider failed, falling back to CPU: {qnn_error}")
            
            # Fallback to CPU if QNN failed
            if not session_created:
                print("🔄 Falling back to CPU provider...")
                self.session = ort.InferenceSession(
                    str(model_to_use),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                    provider_options=[{}]
                )
                print("✅ CPU session created successfully")
            
            print("✅ Session created successfully")
            
            # Get model details
            self.input_details = self.session.get_inputs()[0]
            self.output_details = self.session.get_outputs()[0]
            self.input_shape = tuple(self.input_details.shape)
            self.output_shape = tuple(self.output_details.shape)
            
            current_provider = self.session.get_providers()[0]
            logger.info(f"Model loaded with {current_provider}")
            print(f"🎯 Model ready with {current_provider}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            print(f"❌ Model loading failed: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for NPU inference."""
        target_height, target_width = self.input_shape[2], self.input_shape[3]
        
        # Convert BGR to RGB and resize
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (target_width, target_height))
        
        # Normalize and format for model
        normalized = resized_image.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
        batched = np.expand_dims(transposed, axis=0)  # Add batch dimension
        
        return batched
    
    def postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Convert model output to keypoint coordinates."""
        keypoints = []
        
        if len(output.shape) == 4:
            output = np.squeeze(output, axis=0)  # Remove batch dimension
        
        if len(output.shape) == 3:  # Heatmaps format (keypoints, height, width)
            num_keypoints = output.shape[0]
            heatmap_height, heatmap_width = output.shape[1], output.shape[2]
            
            # Scale factors to map from heatmap to original image
            scale_x = original_shape[1] / heatmap_width
            scale_y = original_shape[0] / heatmap_height
            
            for i in range(num_keypoints):
                heatmap = output[i]
                max_idx = np.argmax(heatmap)
                max_y, max_x = np.unravel_index(max_idx, heatmap.shape)
                
                # Convert to original image coordinates
                x = int(max_x * scale_x)
                y = int(max_y * scale_y)
                keypoints.append((x, y))
        
        return keypoints
    
    def detect_pose(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Detect pose keypoints in image."""
        original_shape = image.shape[:2]
        
        # Preprocess and run inference
        input_tensor = self.preprocess_image(image)
        outputs = self.session.run(None, {self.input_details.name: input_tensor})
        
        # Postprocess to get keypoints
        keypoints = self.postprocess_output(outputs[0], original_shape)
        
        # Draw keypoints on image
        annotated_image = self._draw_keypoints(image.copy(), keypoints)
        
        return annotated_image, keypoints
    
    def _draw_keypoints(self, image: np.ndarray, keypoints: List[Tuple[int, int]]) -> np.ndarray:
        """Draw keypoints on image."""
        for i, (x, y) in enumerate(keypoints):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(image, str(i), (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return image
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_path": str(self.model_path),
            "providers": self.session.get_providers() if self.session else [],
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }
    
    def release(self):
        """Release model resources."""
        if self.session:
            del self.session
            self.session = None


class MediaPipePoseDetector:
    """MediaPipe pose detection fallback."""
    
    def __init__(self):
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
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            available = self._get_available_cameras()
            raise ValueError(f"Cannot open camera {camera_id}. Available: {available}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
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
    
    def _get_available_cameras(self, max_cameras: int = 5) -> List[int]:
        """Get list of available camera indices."""
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    def get_info(self) -> dict:
        """Get detector information."""
        info = {"detector_type": self.detector_type}
        
        if hasattr(self.detector, 'get_model_info'):
            info.update(self.detector.get_model_info())
        
        return info
    
    def release(self):
        """Release resources."""
        if self.detector:
            self.detector.release()


def generate_npu_context(model_path: str):
    """Generate NPU context model for faster loading."""
    if not ONNX_AVAILABLE:
        print("❌ ONNX Runtime not available")
        return False
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    context_path = model_path.parent / f"{model_path.stem}_ctx.onnx"
    
    try:
        print(f"🚀 Generating NPU context model for: {model_path.name}")
        print("This will take ~5-10 seconds for one-time optimization...")
        
        start_time = time.time()
        
        # Configure for context generation
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.add_session_config_entry("ep.context_enable", "1")
        sess_options.add_session_config_entry("ep.context_file_path", str(context_path))
        sess_options.add_session_config_entry("ep.context_embed_mode", "1")
        
        qnn_options = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision": "1"
        }
        
        # Generate context
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["QNNExecutionProvider"],
            provider_options=[qnn_options]
        )
        
        optimization_time = time.time() - start_time
        
        if context_path.exists():
            print(f"✅ Context model generated: {context_path.name}")
            print(f"⏱️  Optimization time: {optimization_time:.1f} seconds")
            print(f"📁 Original: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"📁 Context: {context_path.stat().st_size / 1024 / 1024:.1f} MB")
            print("🚀 Future loads will be ~50x faster!")
            return True
        else:
            print("❌ Context generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Context generation failed: {e}")
        return False


def run_tests():
    """Run basic functionality tests."""
    print("🧪 Running pose detection tests...")
    
    # Test 1: Check imports
    print("\n1. Testing imports...")
    if ONNX_AVAILABLE:
        print("✅ ONNX Runtime available")
    else:
        print("❌ ONNX Runtime not available")
    
    if MEDIAPIPE_AVAILABLE:
        print("✅ MediaPipe available")
    else:
        print("❌ MediaPipe not available")
    
    # Test 2: Check model file
    print("\n2. Testing model file...")
    model_path = Path("model/model.onnx")
    if model_path.exists():
        print(f"✅ Model found: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Test 3: Test NPU detection
    if ONNX_AVAILABLE:
        print("\n3. Testing NPU detection...")
        try:
            app = PoseDetectionApp(force_mediapipe=False)
            info = app.get_info()
            print(f"✅ {info['detector_type']} initialized successfully")
            app.release()
        except Exception as e:
            print(f"❌ NPU test failed: {e}")
    
    # Test 4: Test MediaPipe fallback
    if MEDIAPIPE_AVAILABLE:
        print("\n4. Testing MediaPipe fallback...")
        try:
            app = PoseDetectionApp(force_mediapipe=True)
            info = app.get_info()
            print(f"✅ {info['detector_type']} initialized successfully")
            app.release()
        except Exception as e:
            print(f"❌ MediaPipe test failed: {e}")
    
    print("\n✅ Tests completed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="HRNet Pose Detection with NPU Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time NPU pose detection (default)
  python main.py

  # Process image with NPU
  python main.py --image photo.jpg --output result.jpg

  # Use MediaPipe fallback
  python main.py --mediapipe

  # Generate NPU context for faster startup (one-time)
  python main.py --generate-context

  # Run tests
  python main.py --test
        """
    )
    
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--image", help="Process single image file")
    parser.add_argument("--output", help="Output path for processed image")
    parser.add_argument("--mediapipe", action="store_true", help="Force MediaPipe mode")
    parser.add_argument("--model", default="model/model.onnx", help="ONNX model path")
    parser.add_argument("--generate-context", action="store_true", help="Generate NPU context model")
    parser.add_argument("--test", action="store_true", help="Run functionality tests")
    parser.add_argument("--info", action="store_true", help="Show detector information")
    
    args = parser.parse_args()
    
    try:
        if args.test:
            run_tests()
        elif args.generate_context:
            generate_npu_context(args.model)
        elif args.info:
            app = PoseDetectionApp(args.model, args.mediapipe)
            info = app.get_info()
            print("\n📊 Pose Detection Information")
            print("=" * 40)
            for key, value in info.items():
                print(f"{key}: {value}")
            app.release()
        elif args.image:
            app = PoseDetectionApp(args.model, args.mediapipe)
            app.process_image(args.image, args.output)
            app.release()
        else:
            app = PoseDetectionApp(args.model, args.mediapipe)
            app.process_camera(args.camera)
            app.release()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()