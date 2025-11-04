#!/usr/bin/env python3
"""
Model quantization utility for QNN NPU acceleration.
This script quantizes float32 models to uint16/uint8 for NPU execution.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_quantization_support():
    """Check if quantization tools are available."""
    try:
        import onnxruntime
        from onnxruntime.quantization import QuantType, quantize
        from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
        return True
    except ImportError as e:
        logger.error(f"Quantization tools not available: {e}")
        logger.info("Note: Quantization requires x64 platform and onnxruntime-qnn package")
        return False

class DataReader:
    """Calibration data reader for quantization."""
    
    def __init__(self, model_path: str, num_samples: int = 10):
        """
        Initialize data reader.
        
        Args:
            model_path: Path to the ONNX model
            num_samples: Number of calibration samples to generate
        """
        self.enum_data = None
        
        # Use inference session to get input shape
        import onnxruntime
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        
        self.data_list = []
        
        # Generate random calibration data (should be replaced with real data)
        logger.warning("Using random calibration data - replace with real representative data for better accuracy")
        
        for _ in range(num_samples):
            input_data = {}
            for inp in inputs:
                if inp.type == 'tensor(float)':
                    # Generate random float32 inputs normalized to [0, 1]
                    data = np.random.random(inp.shape).astype(np.float32)
                    # Normalize to typical image range for pose models
                    if len(inp.shape) == 4 and inp.shape[1] == 3:  # NCHW image format
                        data = data  # Already in [0, 1] range
                    input_data[inp.name] = data
                else:
                    logger.warning(f"Unsupported input type: {inp.type}")
            
            if input_data:
                self.data_list.append(input_data)
        
        self.datasize = len(self.data_list)
        logger.info(f"Generated {self.datasize} calibration samples")
    
    def get_next(self):
        """Get next calibration sample."""
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)
    
    def rewind(self):
        """Rewind to beginning of data."""
        self.enum_data = None

def quantize_model_for_qnn(input_model_path: str, output_model_path: str = None):
    """
    Quantize a model for QNN NPU execution.
    
    Args:
        input_model_path: Path to input float32 model
        output_model_path: Path for output quantized model
    """
    if not check_quantization_support():
        return False
    
    input_path = Path(input_model_path)
    if not input_path.exists():
        logger.error(f"Input model not found: {input_model_path}")
        return False
    
    if output_model_path is None:
        output_model_path = str(input_path.with_suffix('.qdq.onnx'))
    
    try:
        from onnxruntime.quantization import QuantType, quantize
        from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
        
        logger.info(f"Quantizing model: {input_model_path}")
        
        # Create data reader for calibration
        data_reader = DataReader(input_model_path)
        
        # Pre-process the original float32 model
        preproc_model_path = str(input_path.with_suffix('.preproc.onnx'))
        logger.info("Pre-processing model for QNN...")
        model_changed = qnn_preprocess_model(str(input_path), preproc_model_path)
        
        model_to_quantize = preproc_model_path if model_changed else str(input_path)
        
        # Generate quantization configuration
        # Using uint16 activations and uint8 weights as recommended for QNN
        logger.info("Generating QNN quantization configuration...")
        qnn_config = get_qnn_qdq_config(
            model_to_quantize,
            data_reader,
            activation_type=QuantType.QUInt16,  # uint16 activations for better accuracy
            weight_type=QuantType.QUInt8        # uint8 weights for efficiency
        )
        
        # Quantize the model
        logger.info("Quantizing model...")
        quantize(model_to_quantize, output_model_path, qnn_config)
        
        # Clean up preprocessed model if it was created
        if model_changed and Path(preproc_model_path).exists():
            os.remove(preproc_model_path)
        
        logger.info(f"✓ Quantized model saved: {output_model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return False

def validate_quantized_model(model_path: str):
    """
    Validate that a quantized model can be loaded.
    
    Args:
        model_path: Path to quantized model
    """
    try:
        import onnxruntime
        
        # Try to load with QNN EP if available
        providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
        available_providers = onnxruntime.get_available_providers()
        
        use_providers = [p for p in providers if p in available_providers]
        
        session = onnxruntime.InferenceSession(model_path, providers=use_providers)
        
        logger.info(f"✓ Quantized model validation successful")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Providers: {session.get_providers()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Quantized model validation failed: {e}")
        return False

def main():
    """Main quantization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize ONNX models for QNN NPU execution")
    parser.add_argument("input_model", help="Path to input float32 ONNX model")
    parser.add_argument("--output", help="Path for output quantized model")
    parser.add_argument("--validate", action="store_true", help="Validate quantized model after creation")
    
    args = parser.parse_args()
    
    if not Path(args.input_model).exists():
        logger.error(f"Input model not found: {args.input_model}")
        sys.exit(1)
    
    # Quantize the model
    success = quantize_model_for_qnn(args.input_model, args.output)
    
    if success and args.validate:
        output_path = args.output or str(Path(args.input_model).with_suffix('.qdq.onnx'))
        validate_quantized_model(output_path)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()