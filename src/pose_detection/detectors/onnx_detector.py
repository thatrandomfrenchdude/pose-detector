#!/usr/bin/env python3
"""
ONNX NPU Pose Detector
NPU-accelerated pose detection using ONNX Runtime with QNN
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available - NPU mode disabled")


class ONNXPoseDetector:
    """NPU-accelerated pose detection using ONNX Runtime with QNN."""
    
    def __init__(self, model_path: str):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime-qnn")
        
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