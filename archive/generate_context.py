#!/usr/bin/env python3
"""
QNN Context Binary Generator
Pre-compiles ONNX models for QNN to eliminate startup delays.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_qnn_context(model_path: str, output_dir: str = None):
    """
    Generate QNN context binary for faster model loading.
    
    Args:
        model_path: Path to original ONNX model
        output_dir: Directory for output files (default: same as model)
    
    Returns:
        str: Path to generated context model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None
    
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    context_model_path = output_dir / f"{model_path.stem}_ctx.onnx"
    
    try:
        import onnxruntime as ort
        
        logger.info(f"Generating QNN context binary for: {model_path}")
        logger.info("This will take a few seconds for optimization...")
        
        # Check if QNN EP is available
        available_providers = ort.get_available_providers()
        if "QNNExecutionProvider" not in available_providers:
            logger.error("QNNExecutionProvider not available")
            logger.info("Install: pip install onnxruntime-qnn")
            return None
        
        # Configure session for context generation
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable context binary generation
        sess_options.add_session_config_entry("ep.context_enable", "1")
        sess_options.add_session_config_entry("ep.context_file_path", str(context_model_path))
        sess_options.add_session_config_entry("ep.context_embed_mode", "1")  # Embed for portability
        
        # QNN provider options
        qnn_options = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision": "1",
            "qnn_context_priority": "high"
        }
        
        logger.info("Starting QNN optimization (this may take 5-10 seconds)...")
        start_time = time.time()
        
        # Create session to generate context binary
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["QNNExecutionProvider"],
            provider_options=[qnn_options]
        )
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Verify context model was created
        if context_model_path.exists():
            logger.info(f"✓ QNN context model generated: {context_model_path}")
            logger.info(f"  Optimization time: {optimization_time:.1f} seconds")
            logger.info(f"  Original size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
            logger.info(f"  Context size: {context_model_path.stat().st_size / 1024 / 1024:.1f} MB")
            
            # Test loading the context model
            logger.info("Testing context model loading...")
            test_start = time.time()
            
            test_session = ort.InferenceSession(
                str(context_model_path),
                providers=["QNNExecutionProvider"],
                provider_options=[qnn_options]
            )
            
            test_end = time.time()
            loading_time = test_end - test_start
            
            logger.info(f"✓ Context model loads in {loading_time:.2f} seconds (vs {optimization_time:.1f}s original)")
            logger.info(f"  Speedup: {optimization_time/loading_time:.1f}x faster startup")
            
            return str(context_model_path)
        else:
            logger.error("Context model was not generated")
            return None
            
    except Exception as e:
        logger.error(f"Context generation failed: {e}")
        return None

def main():
    """Main context generation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate QNN context binary for faster model loading")
    parser.add_argument("model", help="Path to ONNX model")
    parser.add_argument("--output-dir", help="Output directory for context model")
    parser.add_argument("--test-loading", action="store_true", help="Test loading performance")
    
    args = parser.parse_args()
    
    # Generate context
    context_model = generate_qnn_context(args.model, args.output_dir)
    
    if context_model and args.test_loading:
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("="*50)
        
        # Test original model loading time
        logger.info("Testing original model loading time...")
        original_start = time.time()
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(
                args.model,
                providers=["QNNExecutionProvider"],
                provider_options=[{"backend_path": "QnnHtp.dll"}]
            )
            original_time = time.time() - original_start
            logger.info(f"Original model load time: {original_time:.2f}s")
        except Exception as e:
            logger.error(f"Original model test failed: {e}")
            original_time = None
        
        # Test context model loading time  
        logger.info("Testing context model loading time...")
        context_start = time.time()
        try:
            session = ort.InferenceSession(
                context_model,
                providers=["QNNExecutionProvider"],
                provider_options=[{"backend_path": "QnnHtp.dll"}]
            )
            context_time = time.time() - context_start
            logger.info(f"Context model load time: {context_time:.2f}s")
            
            if original_time:
                speedup = original_time / context_time
                logger.info(f"Startup speedup: {speedup:.1f}x faster!")
        except Exception as e:
            logger.error(f"Context model test failed: {e}")
    
    success = context_model is not None
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()