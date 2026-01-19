#!/bin/bash
# Check if system OpenCV has CUDA support

module load python-data gcc cuda cmake

echo "Checking for OpenCV with CUDA support..."
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}'); print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')"
