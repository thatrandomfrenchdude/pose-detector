#!/usr/bin/env python3
"""
Startup time comparison: Original model vs Context model
"""

import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def time_model_loading(model_path, provider="npu", description=""):
    """Time how long it takes to load a model."""
    try:
        from onnx_pose_detector import ONNXPoseDetector
        
        logger.info(f"Testing {description}...")
        start_time = time.time()
        
        detector = ONNXPoseDetector(model_path, provider=provider, prefer_context=False)
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        detector.release()
        
        logger.info(f"✓ {description} loaded in {loading_time:.2f} seconds")
        return loading_time
        
    except Exception as e:
        logger.error(f"✗ {description} failed: {e}")
        return None

def main():
    """Compare startup times."""
    print("🚀 NPU Model Loading Speed Comparison")
    print("=" * 50)
    
    # Test original model (forces recompilation)
    original_time = time_model_loading("model/model.onnx", "npu", "Original model (with optimization)")
    
    print("\n" + "-" * 50)
    
    # Test context model (pre-compiled)
    context_time = time_model_loading("model/model.ctx.onnx", "npu", "Context model (pre-compiled)")
    
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    
    if original_time and context_time:
        speedup = original_time / context_time
        time_saved = original_time - context_time
        
        print(f"Original model loading: {original_time:.2f} seconds")
        print(f"Context model loading:  {context_time:.2f} seconds")
        print(f"")
        print(f"🚀 Speedup: {speedup:.1f}x faster")
        print(f"⏱️  Time saved: {time_saved:.2f} seconds per startup")
        print(f"💡 Improvement: {(time_saved/original_time*100):.1f}% faster startup")
        
        if speedup > 10:
            print(f"🔥 Excellent! Context caching is working perfectly!")
        elif speedup > 5:
            print(f"✅ Good speedup achieved with context caching")
        else:
            print(f"⚠️  Modest speedup - context model may not be fully optimized")
    
    print("\n" + "=" * 50)
    print("💡 TIP: Always use the context model for production!")
    print("   The optimization delay happens once during context generation,")
    print("   then every subsequent load is super fast.")

if __name__ == "__main__":
    main()