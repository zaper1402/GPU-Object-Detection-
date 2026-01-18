#!/bin/bash
################################################################################
# SLURM Job Script for GPU Object Detection on Mahti
# CSC Mahti Supercomputer - NVIDIA A100 GPU
# 
# Author: Group U (Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha)
################################################################################

#SBATCH --job-name=gpu_obj_detect
#SBATCH --account=project_2016196          # CSC project account
#SBATCH --partition=gputest                # Partition: gputest (<15min) or gpusmall (<24h)
#SBATCH --time=00:15:00                    # Max runtime: 15 minutes for testing
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=10                 # CPU cores per task
#SBATCH --gres=gpu:a100:1                  # 1 NVIDIA A100 GPU
#SBATCH --mem=32G                          # Memory per node
#SBATCH --output=logs/job_%j.out           # Standard output log
#SBATCH --error=logs/job_%j.err            # Standard error log

################################################################################
# Job Configuration Summary
################################################################################
echo "========================================================================"
echo "GPU Object Detection - SLURM Job"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "========================================================================"
echo ""

################################################################################
# Load Required Modules
################################################################################
echo "[Step 1/7] Loading modules..."
module purge
module load gcc
module load cuda
module load python-data

echo "Loaded modules:"
module list
echo ""

################################################################################
# Activate Virtual Environment
################################################################################
echo "[Step 2/7] Activating Python virtual environment..."
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    echo "Run setup_mahti.sh first"
    exit 1
fi

echo "Python version: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"
echo ""

################################################################################
# Verify GPU Availability
################################################################################
echo "[Step 3/7] Checking GPU availability..."
nvidia-smi

if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi failed - GPU not accessible"
    exit 1
fi
echo ""

################################################################################
# Verify Setup
################################################################################
echo "[Step 4/7] Verifying project setup..."
cd src
python setup_check.py

if [ $? -ne 0 ]; then
    echo "WARNING: Setup verification failed, but continuing..."
fi
echo ""

################################################################################
# Generate Sample Data (if needed)
################################################################################
echo "[Step 5/7] Checking sample data..."
if [ ! -f "../templates/ball.jpg" ] || [ ! -f "../templates/book.jpg" ]; then
    echo "Generating sample templates..."
    python generate_samples.py
else
    echo "Templates already exist, skipping generation..."
fi
echo ""

################################################################################
# Run GPU Object Detection
################################################################################
echo "[Step 6/7] Running GPU object detection..."
echo "========================================================================"

# Test single image detection
if [ -f "../test_images/sample_test.jpg" ]; then
    echo "Testing single image detection..."
    python object_detector_gpu.py \
        --input ../test_images/sample_test.jpg \
        --templates ../templates \
        --output ../results
    
    echo ""
    echo "Single image detection complete"
    echo "----------------------------------------------------------------------"
else
    echo "WARNING: sample_test.jpg not found, skipping single image test"
fi

# Batch process all test images
TEST_IMAGE_COUNT=$(ls ../test_images/*.jpg 2>/dev/null | wc -l)
if [ $TEST_IMAGE_COUNT -gt 0 ]; then
    echo ""
    echo "Batch processing $TEST_IMAGE_COUNT test images..."
    python object_detector_gpu.py \
        --input ../test_images \
        --templates ../templates \
        --output ../results
    
    echo ""
    echo "Batch processing complete"
    echo "----------------------------------------------------------------------"
else
    echo "WARNING: No test images found in test_images/"
fi

echo ""
echo "========================================================================"

################################################################################
# Run Benchmark (GPU vs CPU)
################################################################################
echo "[Step 7/7] Running GPU vs CPU benchmark..."
echo "========================================================================"

if [ $TEST_IMAGE_COUNT -gt 0 ]; then
    python benchmark.py \
        --input ../test_images \
        --templates ../templates \
        --output ../results/benchmark
    
    echo ""
    echo "Benchmark complete"
    echo "----------------------------------------------------------------------"
else
    echo "WARNING: Skipping benchmark - no test images available"
fi

echo ""
echo "========================================================================"

################################################################################
# Compile and Test CUDA Kernels (Optional)
################################################################################
echo ""
echo "Optional: Testing custom CUDA kernels..."
echo "========================================================================"

cd ..
if [ -f "src/Makefile" ]; then
    echo "Compiling CUDA kernels..."
    make clean
    make
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Running CUDA kernel tests..."
        make run
    else
        echo "WARNING: CUDA compilation failed"
    fi
else
    echo "WARNING: Makefile not found, skipping CUDA kernel tests"
fi

echo ""
echo "========================================================================"

################################################################################
# Results Summary
################################################################################
echo ""
echo "========================================================================"
echo "Job Summary"
echo "========================================================================"
echo "End Time: $(date)"
echo ""

# Check results
if [ -d "results" ]; then
    RESULT_COUNT=$(ls results/*.jpg 2>/dev/null | wc -l)
    echo "Results generated: $RESULT_COUNT image(s)"
    echo "Result files:"
    ls -lh results/*.jpg 2>/dev/null || echo "  No result images found"
    echo ""
fi

# Check benchmark results
if [ -f "results/benchmark/benchmark_report.json" ]; then
    echo "Benchmark report: results/benchmark/benchmark_report.json"
    echo ""
    echo "Speedup summary:"
    python -c "
import json
try:
    with open('results/benchmark/benchmark_report.json', 'r') as f:
        data = json.load(f)
    print(f\"  GPU avg time: {data['gpu']['avg_time']:.2f} ms\")
    print(f\"  CPU avg time: {data['cpu']['avg_time']:.2f} ms\")
    print(f\"  Speedup: {data['speedup']:.2f}x\")
    print(f\"  GPU throughput: {data['gpu']['throughput']:.2f} images/sec\")
    print(f\"  CPU throughput: {data['cpu']['throughput']:.2f} images/sec\")
except Exception as e:
    print(f\"  Could not parse benchmark results: {e}\")
" 2>/dev/null
fi

echo ""
echo "========================================================================"
echo "Job completed successfully!"
echo "========================================================================"
echo ""
echo "To download results to your local machine:"
echo "  scp -r $USER@mahti.csc.fi:$(pwd)/results ./gpu_detection_results"
echo ""
echo "To view logs:"
echo "  cat logs/job_$SLURM_JOB_ID.out"
echo "  cat logs/job_$SLURM_JOB_ID.err"
echo ""
echo "========================================================================"

################################################################################
# Resource Usage Statistics
################################################################################
# Print resource usage at end
echo ""
echo "Resource Usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State

exit 0
