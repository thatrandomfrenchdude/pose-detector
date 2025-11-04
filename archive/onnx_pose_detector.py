import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional
from PIL import Image
from pathlib import Path
import onnxruntime as ort
from model_loader import ModelLoader

logger = logging.getLogger(__name__)

class ONNXPoseDetector:
    """
    ONNX-based pose detection using Qualcomm AI Hub models.
    Supports CPU inference with plans for NPU acceleration.
    """
    
    def __init__(self, model_path: str, provider: str = "cpu", prefer_context: bool = True):
        """
        Initialize the ONNX pose detector.
        
        Args:
            model_path: Path to the ONNX model file
            provider: Execution provider ("cpu", "npu", "gpu")
            prefer_context: Whether to prefer pre-compiled context models
        """
        self.original_model_path = model_path
        self.provider = provider
        self.model_loader = None
        self.session = None
        self.input_shape = None
        self.output_shape = None
        
        # Check for existing context model if using NPU
        self.model_path = self._find_best_model(model_path, prefer_context)
        
        # Load the model
        self._load_model()
    
    def _find_best_model(self, model_path: str, prefer_context: bool) -> str:
        """
        Find the best available model (context vs original).
        
        Args:
            model_path: Original model path
            prefer_context: Whether to prefer context models
            
        Returns:
            str: Path to best available model
        """
        model_path_obj = Path(model_path)
        
        # Look for context models (multiple naming conventions)
        context_candidates = [
            model_path_obj.with_suffix('.ctx.onnx'),  # model.ctx.onnx
            model_path_obj.parent / f"{model_path_obj.stem}_ctx.onnx"  # model_ctx.onnx
        ]
        
        if prefer_context and self.provider == "npu":
            for context_model in context_candidates:
                if context_model.exists():
                    logger.info(f"Using pre-compiled context model: {context_model}")
                    return str(context_model)
        
        if model_path_obj.exists():
            if self.provider == "npu":
                logger.info(f"Using original model (will optimize): {model_path}")
                logger.info("Tip: Run 'python generate_context.py model/model.onnx' for faster startup")
            return model_path
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    def _load_model(self):
        """Load the ONNX model."""
        try:
            self.model_loader = ModelLoader(self.model_path, self.provider)
            self.session = self.model_loader.load_model()
            self.input_shape = self.model_loader.input_shape
            self.output_shape = self.model_loader.output_shape
            
            logger.info(f"ONNX model loaded: {self.model_path}")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output shape: {self.output_shape}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX model inference.
        
        Args:
            image: Input image in BGR format (H, W, C)
            
        Returns:
            np.ndarray: Preprocessed image tensor (1, C, H, W)
        """
        if self.input_shape is None:
            raise ValueError("Model not loaded properly")
        
        # Extract target dimensions from input shape
        # Assuming input shape is (batch, channels, height, width)
        target_height = self.input_shape[2] if len(self.input_shape) == 4 else 256
        target_width = self.input_shape[3] if len(self.input_shape) == 4 else 256
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        resized_image = cv2.resize(rgb_image, (target_width, target_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] range
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        transposed_image = np.transpose(normalized_image, (2, 0, 1))
        batched_image = np.expand_dims(transposed_image, axis=0)
        
        return batched_image
    
    def postprocess_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Postprocess ONNX model output to get keypoint coordinates.
        
        Args:
            output: Raw model output
            original_shape: Original image shape (height, width)
            
        Returns:
            List[Tuple[int, int]]: List of (x, y) keypoint coordinates
        """
        keypoints = []
        
        try:
            # Handle different output formats
            if len(output.shape) == 4:  # (batch, channels, height, width)
                output = np.squeeze(output, axis=0)  # Remove batch dimension
            
            if len(output.shape) == 3:  # (channels, height, width) - heatmaps
                num_keypoints = output.shape[0]
                heatmap_height = output.shape[1]
                heatmap_width = output.shape[2]
                
                # Scale factors to map from heatmap to original image
                scale_x = original_shape[1] / heatmap_width
                scale_y = original_shape[0] / heatmap_height
                
                for i in range(num_keypoints):
                    heatmap = output[i]
                    
                    # Find the maximum point in the heatmap
                    max_idx = np.argmax(heatmap)
                    max_y, max_x = np.unravel_index(max_idx, heatmap.shape)
                    
                    # Convert to original image coordinates
                    x = int(max_x * scale_x)
                    y = int(max_y * scale_y)
                    
                    keypoints.append((x, y))
            
            elif len(output.shape) == 2:  # (num_keypoints, 2) - direct coordinates
                num_keypoints = output.shape[0]
                
                for i in range(num_keypoints):
                    x = int(output[i, 0] * original_shape[1])
                    y = int(output[i, 1] * original_shape[0])
                    keypoints.append((x, y))
            
            else:
                logger.warning(f"Unexpected output shape: {output.shape}")
                
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
        
        return keypoints
    
    def detect_pose(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Detect pose in the given image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: (annotated_image, keypoints)
        """
        if self.session is None:
            raise ValueError("Model not loaded")
        
        original_shape = image.shape[:2]  # (height, width)
        
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            input_name = self.model_loader.input_name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # Postprocess output
            keypoints = self.postprocess_output(outputs[0], original_shape)
            
            # Draw keypoints on image
            annotated_image = self._draw_keypoints(image.copy(), keypoints)
            
            return annotated_image, keypoints
            
        except Exception as e:
            logger.error(f"Error during pose detection: {e}")
            return image.copy(), []
    
    def _draw_keypoints(self, image: np.ndarray, keypoints: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw keypoints on the image.
        
        Args:
            image: Input image
            keypoints: List of (x, y) coordinates
            
        Returns:
            np.ndarray: Image with keypoints drawn
        """
        for i, (x, y) in enumerate(keypoints):
            # Skip invalid keypoints
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            
            # Draw keypoint
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
            
            # Optionally add keypoint number
            cv2.putText(image, str(i), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return image
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model_loader:
            return self.model_loader.get_session_info()
        return {"status": "not_loaded"}
    
    def release(self):
        """Release model resources."""
        if self.session:
            del self.session
            self.session = None
        logger.info("ONNX model resources released")