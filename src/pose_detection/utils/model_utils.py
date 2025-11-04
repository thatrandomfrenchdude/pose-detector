#!/usr/bin/env python3
"""
Model Utilities
Utilities for model management and NPU context generation
"""

import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")


def generate_npu_context(model_path: str) -> bool:
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


def check_model_files(model_path: str) -> dict:
    """Check model file availability and status."""
    model_path = Path(model_path)
    context_path = model_path.parent / f"{model_path.stem}_ctx.onnx"
    
    info = {
        "original_exists": model_path.exists(),
        "context_exists": context_path.exists(),
        "original_path": str(model_path),
        "context_path": str(context_path)
    }
    
    if info["original_exists"]:
        info["original_size_mb"] = model_path.stat().st_size / 1024 / 1024
    
    if info["context_exists"]:
        info["context_size_mb"] = context_path.stat().st_size / 1024 / 1024
    
    return info