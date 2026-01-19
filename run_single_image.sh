#!/bin/bash

################################################################################
# Bash Script: run_single_image.sh
# Description: Runs GPU object detection on a single test image using srun.
#              Allocates compute node, sets up environment, and runs detection.
# Logs each step and progress live.
#
# Author: Group U (Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha)
################################################################################

# Define variables
ACCOUNT="project_2016196"
PARTITION="gputest"
GPU_TYPE="a100:1"
TIME="00:15:00"
MEMORY="16G"
INPUT_IMAGE="./test_images/ball_2.jpg"
TEMPLATE_DIR="./templates"
OUTPUT_DIR="./results"

# Log function
log() {
    echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

echo "========================================================================"
echo "GPU Object Detection - Single Image Test"
echo "========================================================================"
echo ""

# Step 1: Check if files exist
log "Checking input files..."
if [ ! -f "$INPUT_IMAGE" ]; then
    log "ERROR: Input image not found: $INPUT_IMAGE"
    exit 1
fi

if [ ! -d "$TEMPLATE_DIR" ]; then
    log "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi

log "Input image: $INPUT_IMAGE"
log "Templates: $TEMPLATE_DIR"
log "Output: $OUTPUT_DIR"
echo ""

# Step 2: Create output directory
log "Creating output directory..."
mkdir -p $OUTPUT_DIR

# Step 3: Submit job to compute node using srun
log "Requesting GPU compute node..."
log "Account: $ACCOUNT"
log "Partition: $PARTITION"
log "GPU: $GPU_TYPE"
log "Time limit: $TIME"
log "Memory: $MEMORY"
echo ""
log "Waiting for resources allocation..."
echo ""

# Run on compute node with srun
srun --account=$ACCOUNT \
     --partition=$PARTITION \
     --gres=gpu:$GPU_TYPE \
     --time=$TIME \
     --mem=$MEMORY \
     bash -c "
        echo '========================================================================'
        echo 'Compute Node Allocated - Running on: \$(hostname)'
        echo '========================================================================'
        echo ''
        
        echo '[STEP 1/4] Checking GPU availability...'
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        echo ''
        
        echo '[STEP 2/4] Loading required modules...'
        module load python-data gcc cuda cmake
        if [ \$? -ne 0 ]; then
            echo '[ERROR] Failed to load modules'
            exit 1
        fi
        echo '[SUCCESS] Modules loaded: gcc, cuda, python-data'
        echo ''
        
        echo '[STEP 3/4] Checking Python OpenCV CUDA support...'
        python3 -c \"import cv2; print('OpenCV:', cv2.__version__); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())\" || echo '[WARNING] OpenCV check failed'
        echo ''
        
        # Skip venv for now - use system Python with CUDA-enabled OpenCV
        # echo '[STEP 3/4] Activating virtual environment...'
        # source .venv/bin/activate
        # if [ \$? -ne 0 ]; then
        #     echo '[ERROR] Failed to activate virtual environment'
        #     exit 1
        # fi
        # echo '[SUCCESS] Virtual environment activated'
        # echo ''
        
        echo '[STEP 4/4] Running GPU object detection...'
        echo 'Input: $INPUT_IMAGE'
        echo 'Templates: $TEMPLATE_DIR'
        echo 'Output: $OUTPUT_DIR'
        echo ''
        python3 src/object_detector_gpu.py \\
            --input $INPUT_IMAGE \\
            --templates $TEMPLATE_DIR \\
            --output $OUTPUT_DIR
        
        EXIT_CODE=\$?
        echo ''
        if [ \$EXIT_CODE -eq 0 ]; then
            echo '========================================================================'
            echo 'Detection completed successfully!'
            echo '========================================================================'
            echo ''
            echo 'View results:'
            echo '  ls -lh $OUTPUT_DIR/'
            echo ''
        else
            echo '========================================================================'
            echo 'Detection failed with exit code: '\$EXIT_CODE
            echo '========================================================================'
            exit \$EXIT_CODE
        fi
    "

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    log "Job completed successfully"
    log "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View output images:"
    echo "  ls -lh $OUTPUT_DIR/"
    echo ""
else
    log "Job failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

# End of script