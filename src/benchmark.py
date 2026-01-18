#!/usr/bin/env python3
"""
Benchmark comparison script: GPU vs CPU object detection
Generates performance reports and speedup metrics

Author: Group U
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

# Import both detectors
sys.path.insert(0, os.path.dirname(__file__))
from object_detector_gpu import GPUObjectDetector
from object_detector_cpu import CPUObjectDetector


def benchmark_detector(detector, image_paths: List[str], name: str) -> Dict:
    """Run benchmark on detector with given images."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {name}")
    print(f"{'='*60}")
    
    results = {
        'name': name,
        'images': [],
        'times': [],
        'detections_count': [],
        'total_time': 0.0,
        'avg_time': 0.0,
        'throughput': 0.0
    }
    
    for img_path in image_paths:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        detections, proc_time = detector.process_image(img_path, output_path=None)
        
        results['images'].append(os.path.basename(img_path))
        results['times'].append(proc_time * 1000)  # Convert to ms
        results['detections_count'].append(len(detections))
        results['total_time'] += proc_time
    
    # Calculate aggregate metrics
    n_images = len(image_paths)
    if n_images > 0:
        results['avg_time'] = (results['total_time'] / n_images) * 1000  # ms
        results['throughput'] = n_images / results['total_time']  # images/sec
    
    return results


def generate_report(gpu_results: Dict, cpu_results: Dict, output_dir: str):
    """Generate comprehensive performance report."""
    
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON REPORT")
    print(f"{'='*60}")
    
    # Summary statistics
    print("\n1. SUMMARY STATISTICS")
    print(f"   Images processed: {len(gpu_results['images'])}")
    print(f"\n   GPU Performance:")
    print(f"   - Total time: {gpu_results['total_time']*1000:.2f} ms")
    print(f"   - Average time: {gpu_results['avg_time']:.2f} ms/image")
    print(f"   - Throughput: {gpu_results['throughput']:.2f} images/sec")
    
    print(f"\n   CPU Performance:")
    print(f"   - Total time: {cpu_results['total_time']*1000:.2f} ms")
    print(f"   - Average time: {cpu_results['avg_time']:.2f} ms/image")
    print(f"   - Throughput: {cpu_results['throughput']:.2f} images/sec")
    
    # Speedup calculation
    if cpu_results['avg_time'] > 0:
        speedup = cpu_results['avg_time'] / gpu_results['avg_time']
        throughput_gain = gpu_results['throughput'] / cpu_results['throughput']
    else:
        speedup = 0.0
        throughput_gain = 0.0
    
    print(f"\n2. SPEEDUP METRICS")
    print(f"   - Time speedup: {speedup:.2f}x")
    print(f"   - Throughput gain: {throughput_gain:.2f}x")
    print(f"   - Time saved: {cpu_results['total_time'] - gpu_results['total_time']:.2f} sec")
    
    # Per-image comparison
    print(f"\n3. PER-IMAGE COMPARISON")
    print(f"   {'Image':<30} {'GPU (ms)':<12} {'CPU (ms)':<12} {'Speedup':<10}")
    print(f"   {'-'*64}")
    
    for i, img_name in enumerate(gpu_results['images']):
        gpu_time = gpu_results['times'][i]
        cpu_time = cpu_results['times'][i]
        img_speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
        print(f"   {img_name:<30} {gpu_time:>10.2f}  {cpu_time:>10.2f}  {img_speedup:>8.2f}x")
    
    # Detection counts
    print(f"\n4. DETECTION COUNTS")
    print(f"   {'Image':<30} {'GPU':<8} {'CPU':<8} {'Match':<8}")
    print(f"   {'-'*54}")
    
    all_match = True
    for i, img_name in enumerate(gpu_results['images']):
        gpu_count = gpu_results['detections_count'][i]
        cpu_count = cpu_results['detections_count'][i]
        match = "✓" if gpu_count == cpu_count else "✗"
        if gpu_count != cpu_count:
            all_match = False
        print(f"   {img_name:<30} {gpu_count:<8} {cpu_count:<8} {match:<8}")
    
    # Save JSON report
    report_data = {
        'gpu': gpu_results,
        'cpu': cpu_results,
        'speedup': speedup,
        'throughput_gain': throughput_gain
    }
    
    json_path = os.path.join(output_dir, 'benchmark_report.json')
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\n   Report saved to: {json_path}")
    
    # Generate plots
    try:
        generate_plots(gpu_results, cpu_results, output_dir)
    except Exception as e:
        print(f"[WARNING] Could not generate plots: {e}")
    
    # Final verdict
    print(f"\n{'='*60}")
    if speedup >= 1.5 and all_match:
        print("[SUCCESS] TEST PASSED")
        print(f"GPU achieves {speedup:.2f}x speedup with consistent detection accuracy")
    elif speedup >= 1.0:
        print("[SUCCESS] TEST PASSED")
        print(f"GPU provides {speedup:.2f}x speedup")
    else:
        print("[WARNING] GPU slower than CPU - check CUDA installation")
    print(f"{'='*60}")


def generate_plots(gpu_results: Dict, cpu_results: Dict, output_dir: str):
    """Generate visualization plots."""
    
    # Plot 1: Time comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Per-image time comparison
    ax1 = axes[0, 0]
    x = np.arange(len(gpu_results['images']))
    width = 0.35
    
    ax1.bar(x - width/2, gpu_results['times'], width, label='GPU', color='green', alpha=0.7)
    ax1.bar(x + width/2, cpu_results['times'], width, label='CPU', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Image')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Processing Time per Image')
    ax1.set_xticks(x)
    ax1.set_xticklabels([img[:15] for img in gpu_results['images']], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Average time comparison
    ax2 = axes[0, 1]
    methods = ['GPU', 'CPU']
    avg_times = [gpu_results['avg_time'], cpu_results['avg_time']]
    colors_avg = ['green', 'blue']
    
    ax2.bar(methods, avg_times, color=colors_avg, alpha=0.7)
    ax2.set_ylabel('Average Time (ms)')
    ax2.set_title('Average Processing Time')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(avg_times):
        ax2.text(i, v + max(avg_times)*0.02, f'{v:.2f} ms', ha='center', va='bottom')
    
    # Throughput comparison
    ax3 = axes[1, 0]
    throughputs = [gpu_results['throughput'], cpu_results['throughput']]
    
    ax3.bar(methods, throughputs, color=colors_avg, alpha=0.7)
    ax3.set_ylabel('Throughput (images/sec)')
    ax3.set_title('Processing Throughput')
    ax3.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(throughputs):
        ax3.text(i, v + max(throughputs)*0.02, f'{v:.2f}', ha='center', va='bottom')
    
    # Speedup per image
    ax4 = axes[1, 1]
    speedups = [cpu_results['times'][i] / gpu_results['times'][i] 
                if gpu_results['times'][i] > 0 else 0 
                for i in range(len(gpu_results['times']))]
    
    ax4.bar(x, speedups, color='orange', alpha=0.7)
    ax4.axhline(y=1.0, color='red', linestyle='--', label='No speedup')
    ax4.set_xlabel('Image')
    ax4.set_ylabel('Speedup (x)')
    ax4.set_title('GPU Speedup per Image')
    ax4.set_xticks(x)
    ax4.set_xticklabels([img[:15] for img in gpu_results['images']], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'benchmark_plots.png')
    plt.savefig(plot_path, dpi=150)
    print(f"   Plots saved to: {plot_path}")
    plt.close()


def main():
    """Main benchmark function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark GPU vs CPU Object Detection')
    parser.add_argument('--templates', type=str, default='../templates',
                        help='Directory containing template images')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image directory for benchmarking')
    parser.add_argument('--output', type=str, default='../results/benchmark',
                        help='Output directory for benchmark results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get test images
    if not os.path.isdir(args.input):
        print(f"[ERROR] Input must be a directory: {args.input}")
        return 1
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        os.path.join(args.input, f) for f in os.listdir(args.input)
        if f.lower().endswith(image_extensions)
    ]
    
    if len(image_files) == 0:
        print(f"[ERROR] No images found in {args.input}")
        return 1
    
    print(f"[INFO] Found {len(image_files)} test images")
    
    # Initialize detectors
    print("\n[INFO] Initializing GPU detector...")
    gpu_detector = GPUObjectDetector(
        template_dir=args.templates,
        min_matches=10,
        ratio_threshold=0.75
    )
    
    print("\n[INFO] Initializing CPU detector...")
    cpu_detector = CPUObjectDetector(
        template_dir=args.templates,
        min_matches=10,
        ratio_threshold=0.75
    )
    
    # Run benchmarks
    gpu_results = benchmark_detector(gpu_detector, image_files, "GPU (CUDA)")
    cpu_results = benchmark_detector(cpu_detector, image_files, "CPU")
    
    # Generate report
    generate_report(gpu_results, cpu_results, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
