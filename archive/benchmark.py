#!/usr/bin/env python3
"""
Performance comparison script for CPU vs NPU inference.
"""

import time
import logging
import argparse
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def benchmark_inference(model_path, provider, num_runs=10):
    """
    Benchmark inference performance.
    
    Args:
        model_path: Path to ONNX model
        provider: Execution provider ("cpu" or "npu")
        num_runs: Number of inference runs for averaging
        
    Returns:
        dict: Performance metrics
    """
    try:
        from onnx_pose_detector import ONNXPoseDetector
        
        logger.info(f"Benchmarking {provider.upper()} performance...")
        
        # Initialize detector
        detector = ONNXPoseDetector(model_path, provider=provider)
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup runs
        logger.info("Running warmup iterations...")
        for _ in range(3):
            detector.detect_pose(test_image)
        
        # Benchmark runs
        logger.info(f"Running {num_runs} benchmark iterations...")
        times = []
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            annotated_image, keypoints = detector.detect_pose(test_image)
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(inference_time)
            
            if (i + 1) % 5 == 0:
                logger.info(f"  Completed {i + 1}/{num_runs} runs")
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        detector.release()
        
        return {
            'provider': provider.upper(),
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'fps': fps,
            'keypoints_detected': len(keypoints)
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed for {provider}: {e}")
        return None

def compare_performance(model_path="model/model.onnx", num_runs=10):
    """
    Compare CPU vs NPU performance.
    
    Args:
        model_path: Path to ONNX model
        num_runs: Number of runs for each provider
    """
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    print("🚀 NPU vs CPU Performance Comparison")
    print("=" * 50)
    
    results = {}
    
    # Benchmark NPU
    npu_result = benchmark_inference(model_path, "npu", num_runs)
    if npu_result:
        results['npu'] = npu_result
    
    # Benchmark CPU
    cpu_result = benchmark_inference(model_path, "cpu", num_runs)
    if cpu_result:
        results['cpu'] = cpu_result
    
    # Display results
    print("\n📊 Performance Results")
    print("=" * 50)
    
    if 'npu' in results and 'cpu' in results:
        npu = results['npu']
        cpu = results['cpu']
        
        speedup = cpu['avg_time_ms'] / npu['avg_time_ms']
        fps_improvement = npu['fps'] / cpu['fps']
        
        print(f"🔥 NPU Results:")
        print(f"   Average time: {npu['avg_time_ms']:.2f} ms")
        print(f"   FPS: {npu['fps']:.1f}")
        print(f"   Range: {npu['min_time_ms']:.2f} - {npu['max_time_ms']:.2f} ms")
        
        print(f"\n🖥️  CPU Results:")
        print(f"   Average time: {cpu['avg_time_ms']:.2f} ms") 
        print(f"   FPS: {cpu['fps']:.1f}")
        print(f"   Range: {cpu['min_time_ms']:.2f} - {cpu['max_time_ms']:.2f} ms")
        
        print(f"\n⚡ Performance Improvement:")
        print(f"   NPU is {speedup:.1f}x faster than CPU")
        print(f"   FPS improvement: {fps_improvement:.1f}x")
        print(f"   Time reduction: {((cpu['avg_time_ms'] - npu['avg_time_ms']) / cpu['avg_time_ms'] * 100):.1f}%")
        
    elif 'npu' in results:
        npu = results['npu']
        print(f"🔥 NPU Results:")
        print(f"   Average time: {npu['avg_time_ms']:.2f} ms")
        print(f"   FPS: {npu['fps']:.1f}")
        print(f"   (CPU benchmark failed - NPU only results)")
        
    elif 'cpu' in results:
        cpu = results['cpu']
        print(f"🖥️  CPU Results:")
        print(f"   Average time: {cpu['avg_time_ms']:.2f} ms")
        print(f"   FPS: {cpu['fps']:.1f}")
        print(f"   (NPU benchmark failed - CPU only results)")
        
    else:
        print("❌ Both benchmarks failed")
        
    print("\n" + "=" * 50)

def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark NPU vs CPU performance")
    parser.add_argument("--model", default="model/model.onnx", help="Path to ONNX model")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--npu-only", action="store_true", help="Benchmark NPU only")
    parser.add_argument("--cpu-only", action="store_true", help="Benchmark CPU only")
    
    args = parser.parse_args()
    
    if args.npu_only:
        result = benchmark_inference(args.model, "npu", args.runs)
        if result:
            print(f"🔥 NPU Performance: {result['avg_time_ms']:.2f}ms avg, {result['fps']:.1f} FPS")
    elif args.cpu_only:
        result = benchmark_inference(args.model, "cpu", args.runs)
        if result:
            print(f"🖥️  CPU Performance: {result['avg_time_ms']:.2f}ms avg, {result['fps']:.1f} FPS")
    else:
        compare_performance(args.model, args.runs)

if __name__ == "__main__":
    main()