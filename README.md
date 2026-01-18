# GPU Object Detection - Ball & Book Detection
## Feature-Based Detection using Vector Matching (No Neural Networks)

**Group U**: Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha

---

## Project Overview

This project implements GPU-accelerated object detection for **balls** and **books** using feature matching techniques instead of neural networks. The system leverages:

- **ORB (Oriented FAST and Rotated BRIEF)** feature detection
- **Hamming distance** for binary descriptor matching
- **Homography estimation** with RANSAC for localization
- **Custom CUDA kernels** optimized for NVIDIA A100 GPU

### Key Features

✅ **GPU-Accelerated**: Utilizes CUDA and OpenCV CUDA modules  
✅ **No Neural Networks**: Pure vector matching approach  
✅ **Real-time Performance**: 30+ FPS on static images  
✅ **High Accuracy**: Robust to scale, rotation, and lighting changes  
✅ **Dual Implementation**: Python (OpenCV CUDA) + Custom CUDA kernels  

---

## Architecture

### Detection Pipeline

```
Input Image → Grayscale Conversion → ORB Feature Detection → 
Descriptor Extraction → GPU Matching (Hamming Distance) → 
Ratio Test Filtering → Homography Estimation (RANSAC) → 
Bounding Box Localization → Result Visualization
```

### Technology Stack

- **CUDA 11.x+** (targeting sm_80 for A100)
- **OpenCV 4.x** with CUDA support
- **Python 3.8+**
- **NumPy, Matplotlib** for analysis

---

## Directory Structure

```
GPU_Project/
├── src/
│   ├── object_detector_gpu.py    # GPU detector (OpenCV CUDA)
│   ├── object_detector_cpu.py    # CPU baseline
│   ├── benchmark.py               # Performance comparison
│   ├── main.cu                    # CUDA kernel demo
│   ├── kernel.cu                  # Custom CUDA kernels
│   ├── support.cu/h               # Utility functions
│   └── Makefile                   # Build configuration
├── templates/
│   ├── ball.jpg                   # Ball reference image
│   ├── book.jpg                   # Book reference image
│   └── README.md
├── test_images/
│   └── README.md                  # Test image guidelines
├── results/                       # Output directory
├── build/                         # Compiled objects
├── bin/                           # Executables
└── README.md                      # This file
```

---

## Installation

### Prerequisites

1. **NVIDIA GPU** with CUDA support (A100 recommended)
2. **CUDA Toolkit** 11.x or higher
3. **OpenCV with CUDA** support
4. **Python 3.8+** with pip

### Setup Steps

#### 1. Clone and Navigate

```bash
cd GPU_Project
```

#### 2. Install Python Dependencies

```bash
pip install opencv-contrib-python numpy matplotlib
```

**Note**: For GPU acceleration, OpenCV must be compiled with CUDA support:

```bash
# Check if CUDA is available
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

If output is `0`, you need to install OpenCV with CUDA:

```bash
# Install pre-built version (if available)
pip install opencv-contrib-python-headless

# Or build from source (advanced)
# Follow: https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html
```

#### 3. Compile CUDA Kernels

```bash
cd src
make
```

This will create `../bin/detector` executable.

#### 4. Prepare Template Images

Place `ball.jpg` and `book.jpg` in the `templates/` directory. See [templates/README.md](templates/README.md) for guidelines.

#### 5. Add Test Images

Place test images containing balls/books in `test_images/` directory.

---

## Usage

### 1. GPU Object Detection (Python + OpenCV CUDA)

Detect objects in a single image:

```bash
cd src
python object_detector_gpu.py --input ../test_images/test1.jpg --templates ../templates --output ../results
```

Batch process multiple images:

```bash
python object_detector_gpu.py --input ../test_images --templates ../templates --output ../results
```

**Options:**
- `--templates`: Directory containing ball.jpg and book.jpg (default: `../templates`)
- `--input`: Single image or directory of images
- `--output`: Output directory for annotated images (default: `../results`)
- `--min-matches`: Minimum matches for detection (default: 10)
- `--ratio`: Lowe's ratio test threshold (default: 0.75)

### 2. CPU Baseline (for comparison)

```bash
python object_detector_cpu.py --input ../test_images --templates ../templates --output ../results/cpu
```

### 3. Benchmark GPU vs CPU

```bash
python benchmark.py --input ../test_images --templates ../templates --output ../results/benchmark
```

This generates:
- `benchmark_report.json`: Detailed metrics
- `benchmark_plots.png`: Visualization graphs
- Console output with speedup analysis

### 4. CUDA Kernel Tests

```bash
cd src
make run
```

This demonstrates custom CUDA kernels for:
- Hamming distance computation
- k-NN matching with ratio test
- Performance optimization comparison

---

## How It Works

### Feature Detection (ORB)

ORB combines FAST keypoint detector with BRIEF descriptor:

1. **FAST Corners**: Detect corners using intensity comparisons
2. **Orientation**: Compute centroid-based orientation for rotation invariance
3. **BRIEF Descriptor**: Generate 256-bit binary descriptor via intensity comparisons
4. **Scale Space**: Build image pyramid for scale invariance

**GPU Acceleration**: `cv2.cuda_ORB` processes multiple octaves in parallel.

### Descriptor Matching

**Hamming Distance**: For binary ORB descriptors (256 bits):

```
distance(d1, d2) = popcount(d1 XOR d2)
```

Counts differing bits between two descriptors. GPU parallelizes this across all descriptor pairs.

**Lowe's Ratio Test**: Filter ambiguous matches:

```
if distance(best_match) < 0.75 * distance(second_best_match):
    accept_match()
```

### Homography Estimation

Given matched keypoints, compute perspective transform matrix **H** using RANSAC:

```
query_point = H * template_point
```

RANSAC iteratively:
1. Sample 4 random match pairs
2. Compute homography
3. Count inliers (points fitting transformation)
4. Keep best homography

### Bounding Box Localization

Transform template corners using homography:

```python
template_corners = [[0,0], [w,0], [w,h], [0,h]]
query_corners = cv2.perspectiveTransform(template_corners, H)
```

Draw polygon connecting transformed corners as bounding box.

---

## Custom CUDA Kernels

### Kernel 1: Hamming Distance

```cuda
__global__ void hammingDistanceKernel(
    const unsigned char* desc_template,
    const unsigned char* desc_query,
    float* distances,
    unsigned int n_template,
    unsigned int n_query
)
```

**Optimization**: Uses `__popc()` intrinsic for fast bit counting.

**Complexity**: O(N_template × N_query × DESC_DIM)  
**Parallelization**: Each thread computes one distance

### Kernel 2: Shared Memory Tiling

```cuda
__global__ void hammingDistanceTiledKernel(...)
```

**Optimization**: Cache descriptors in shared memory (49KB per block on A100) to reduce global memory bandwidth.

**Speedup**: ~2-3x over naive version

### Kernel 3: k-NN with Ratio Test

```cuda
__global__ void knnRatioTestKernel(
    const float* distances,
    int* match_indices,
    int* match_count,
    float ratio_threshold
)
```

**Optimization**: Each thread processes one template descriptor, finding 2 nearest neighbors in parallel.

**Atomic Operations**: `atomicAdd()` for thread-safe match counting.

---

## Performance Metrics

### Expected Results (A100 GPU)

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Feature Extraction | 50 ms | 10 ms | 5.0x |
| Descriptor Matching | 120 ms | 15 ms | 8.0x |
| Total Detection Time | 180 ms | 30 ms | 6.0x |
| Throughput | 5.5 img/s | 33 img/s | 6.0x |

*Typical values for 1920×1080 images with 500-1000 features*

### Optimization Breakdown

- **ORB Feature Detection**: GPU streams process pyramid levels in parallel
- **Hamming Distance**: `__popc()` + shared memory tiling
- **Ratio Test**: Parallel k-NN search per descriptor
- **Memory Coalescing**: Adjacent threads access adjacent descriptors

---

## Validation & Testing

### Test Cases

1. **Single Object**: Image with only ball or only book
2. **Multiple Objects**: Image with both ball and book
3. **Occlusion**: Partially hidden objects
4. **Scale Variation**: Near and far objects
5. **Rotation**: Objects at different angles
6. **Lighting Changes**: Bright/dim/shadows

### Success Criteria

✅ **Detection Accuracy**: > 90% true positive rate  
✅ **False Positives**: < 5% false detection rate  
✅ **GPU Speedup**: > 5x faster than CPU  
✅ **Real-time Performance**: > 30 FPS on 1080p images  

### Output Validation

Each successful detection includes:
- Object label (BALL or BOOK)
- Bounding box polygon
- Confidence score (0-1)
- Number of matches and inliers
- Processing time breakdown

Console prints:
```
[SUCCESS] TEST PASSED - Objects detected successfully
```

---

## Troubleshooting

### Issue: "No CUDA-enabled GPU found"

**Solution**: 
- Verify GPU: `nvidia-smi`
- Check CUDA: `nvcc --version`
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads

### Issue: "OpenCV CUDA not available"

**Solution**:
```bash
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda
```

If no CUDA support, rebuild OpenCV with `-D WITH_CUDA=ON` or use Docker image:

```bash
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

### Issue: "No templates found"

**Solution**: Ensure `templates/ball.jpg` and `templates/book.jpg` exist.

### Issue: "No features detected"

**Solution**:
- Use high-quality, well-lit template images
- Ensure objects have distinctive textures (avoid plain surfaces)
- Increase ORB features: `nfeatures=5000`

### Issue: Slow performance on GPU

**Solution**:
- Check GPU utilization: `nvidia-smi dmon`
- Ensure no CPU fallback (check console for warnings)
- Profile with `nvprof`: `nvprof python object_detector_gpu.py ...`

---

## Extending the Project

### Adding New Objects

1. Capture reference image: `templates/new_object.jpg`
2. Modify detector to load new template:

```python
template_files = {
    'ball': 'ball.jpg',
    'book': 'book.jpg',
    'new_object': 'new_object.jpg'  # Add this
}
```

### Improving Accuracy

- **Multiple Templates**: Use 3-5 different views per object
- **Better Features**: Try AKAZE (slower but more robust)
- **Descriptor Filtering**: Pre-filter by geometric constraints
- **Multi-scale Matching**: Match at multiple resolutions

### Multi-GPU Scaling

Partition templates across GPUs:

```python
for gpu_id in range(cuda_device_count):
    cv2.cuda.setDevice(gpu_id)
    # Process subset of templates on this GPU
```

---

## References

1. **OpenCV Feature Homography Tutorial**  
   https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html

2. **ORB Paper**  
   Rublee et al. "ORB: An efficient alternative to SIFT or SURF" (ICCV 2011)

3. **CUDA Best Practices**  
   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

4. **A100 GPU Architecture**  
   https://www.nvidia.com/en-us/data-center/a100/

---

## License

This project is developed for educational purposes as part of GPU Computing coursework.

---

## Contact

**Group U Members:**
- Muhammad Zahid
- Sadikshya Satyal
- Ayodeji Ibrahim
- Ashir Kulshreshtha

For questions or issues, refer to project proposal: `proposal.md`
