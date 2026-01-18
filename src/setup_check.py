#!/usr/bin/env python3
"""
Quick test script to verify installation and setup

Author: Group U
"""

import os
import sys

def check_cuda():
    """Check CUDA availability."""
    print("1. Checking CUDA...")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ CUDA GPU detected")
            return True
        else:
            print("   ✗ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("   ✗ nvidia-smi not found (CUDA may not be installed)")
        return False

def check_opencv():
    """Check OpenCV and CUDA support."""
    print("\n2. Checking OpenCV...")
    try:
        import cv2
        print(f"   ✓ OpenCV version: {cv2.__version__}")
        
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_count > 0:
            print(f"   ✓ OpenCV CUDA support: {cuda_count} device(s)")
            return True
        else:
            print("   ✗ OpenCV CUDA support: NOT available")
            print("   ℹ GPU acceleration will not work. Install opencv-contrib with CUDA.")
            return False
    except ImportError:
        print("   ✗ OpenCV not installed")
        print("   Run: pip install opencv-contrib-python")
        return False
    except Exception as e:
        print(f"   ✗ Error checking OpenCV CUDA: {e}")
        return False

def check_dependencies():
    """Check other Python dependencies."""
    print("\n3. Checking Python dependencies...")
    required = ['numpy', 'matplotlib']
    all_ok = True
    
    for package in required:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} not installed")
            all_ok = False
    
    return all_ok

def check_templates():
    """Check template images."""
    print("\n4. Checking template images...")
    templates_dir = os.path.join('..', 'templates')
    required_templates = ['ball.jpg', 'book.jpg']
    
    if not os.path.exists(templates_dir):
        print(f"   ✗ Templates directory not found: {templates_dir}")
        return False
    
    all_ok = True
    for template in required_templates:
        path = os.path.join(templates_dir, template)
        if os.path.exists(path):
            print(f"   ✓ {template}")
        else:
            print(f"   ✗ {template} not found")
            all_ok = False
    
    return all_ok

def check_test_images():
    """Check test images."""
    print("\n5. Checking test images...")
    test_dir = os.path.join('..', 'test_images')
    
    if not os.path.exists(test_dir):
        print(f"   ✗ Test images directory not found: {test_dir}")
        return False
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(test_dir) 
              if f.lower().endswith(image_extensions)]
    
    if len(images) > 0:
        print(f"   ✓ Found {len(images)} test image(s)")
        return True
    else:
        print("   ⚠ No test images found")
        print("   Add test images to test_images/ directory")
        return False

def check_cuda_compilation():
    """Check if CUDA code can be compiled."""
    print("\n6. Checking CUDA compilation...")
    
    if not os.path.exists('Makefile'):
        print("   ℹ Makefile not found (run from src/ directory)")
        return None
    
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[3] if len(result.stdout.split('\n')) > 3 else result.stdout
            print(f"   ✓ nvcc compiler available: {version_line.strip()}")
            return True
        else:
            print("   ✗ nvcc compiler check failed")
            return False
    except FileNotFoundError:
        print("   ✗ nvcc not found (CUDA toolkit may not be installed)")
        return False

def main():
    print("="*60)
    print("GPU Object Detection - Setup Verification")
    print("="*60)
    
    results = {
        'cuda': check_cuda(),
        'opencv': check_opencv(),
        'dependencies': check_dependencies(),
        'templates': check_templates(),
        'test_images': check_test_images(),
        'cuda_compiler': check_cuda_compilation()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Core requirements
    core_ok = results['opencv'] and results['dependencies']
    gpu_ok = results['cuda'] and results['opencv']
    data_ok = results['templates']
    
    if core_ok and data_ok:
        print("✓ Core requirements met - detector can run")
        if gpu_ok:
            print("✓ GPU acceleration available")
        else:
            print("⚠ GPU acceleration NOT available (will use CPU)")
        
        print("\nNext steps:")
        print("1. Ensure templates are in templates/ directory")
        print("2. Add test images to test_images/")
        print("3. Run detector:")
        print("   python object_detector_gpu.py --input ../test_images/image.jpg --templates ../templates")
        
    else:
        print("✗ Setup incomplete")
        print("\nMissing requirements:")
        if not results['opencv']:
            print("  - OpenCV installation or CUDA support")
        if not results['dependencies']:
            print("  - Python dependencies (numpy, matplotlib)")
        if not results['templates']:
            print("  - Template images (ball.jpg, book.jpg)")
    
    if results['cuda_compiler']:
        print("\n4. Compile and run CUDA kernels:")
        print("   make run")
    
    print("="*60)
    
    return 0 if (core_ok and data_ok) else 1

if __name__ == '__main__':
    sys.exit(main())
