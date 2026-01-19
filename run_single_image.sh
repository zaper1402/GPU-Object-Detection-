#!/bin/bash

################################################################################
# Bash Script: run_single_image.sh
# Description: Starts an interactive session, sets up the environment, and runs
#              the object detection script on a single test image.
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
INPUT_IMAGE="../test_images/ball_2.jpg"
TEMPLATE_DIR="../templates"
OUTPUT_DIR="../results"

# Log function
log() {
    echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Step 1: Start interactive session
log "Starting interactive session on partition $PARTITION with $GPU_TYPE GPU..."
sinteractive --account=$ACCOUNT --partition=$PARTITION --gres=gpu:$GPU_TYPE --time=$TIME --mem=$MEMORY

if [ $? -ne 0 ]; then
    log "Failed to start interactive session. Exiting."
    exit 1
fi

# Step 2: Load required modules
log "Loading required modules..."
module load gcc cuda python-data
if [ $? -ne 0 ]; then
    log "Failed to load required modules. Exiting."
    exit 1
fi

# Step 3: Activate virtual environment
log "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    log "Failed to activate virtual environment. Exiting."
    exit 1
fi

# Step 4: Run object detection on single image
log "Running object detection on single test image: $INPUT_IMAGE"
python src/object_detector_gpu.py --input $INPUT_IMAGE --templates $TEMPLATE_DIR --output $OUTPUT_DIR
if [ $? -ne 0 ]; then
    log "Object detection failed. Exiting."
    exit 1
fi

Step 5: Completion log : tail output
tail -f $OUTPUT_DIR/object_detection.log

log "Object detection completed successfully. Results saved to $OUTPUT_DIR."

# End of script