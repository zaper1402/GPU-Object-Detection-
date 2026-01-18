#!/bin/bash
################################################################################
# Mahti Cluster Setup Script for GPU Object Detection Project
# CSC Mahti Supercomputer - NVIDIA A100 GPU
# 
# Author: Group U (Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha)
################################################################################

echo "========================================================================"
echo "GPU Object Detection - Mahti Cluster Setup"
echo "========================================================================"
echo ""

# Function to print colored output
print_success() {
    echo -e "\e[32m✓ $1\e[0m"
}

print_error() {
    echo -e "\e[31m✗ $1\e[0m"
}

print_info() {
    echo -e "\e[34mℹ $1\e[0m"
}

# Step 1: Load required modules
echo "[1/6] Loading required modules..."
module purge
module load gcc/11.3.0
module load cuda/11.5.0
module load python-data/3.10

if [ $? -eq 0 ]; then
    print_success "Modules loaded: gcc/11.3.0, cuda/11.5.0, python-data/3.10"
else
    print_error "Failed to load modules"
    exit 1
fi
echo ""

# Step 2: Verify CUDA installation
echo "[2/6] Verifying CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_success "CUDA compiler found: version $CUDA_VERSION"
else
    print_error "CUDA compiler (nvcc) not found"
    exit 1
fi
echo ""

# Step 3: Create virtual environment
echo "[3/6] Creating Python virtual environment..."
if [ -d ".venv" ]; then
    print_info "Virtual environment already exists, skipping..."
else
    python -m venv .venv --system-site-packages
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi
echo ""

# Step 4: Activate and install Python packages
echo "[4/6] Installing Python dependencies..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install required packages
print_info "Installing opencv-contrib-python..."
pip install opencv-contrib-python --quiet
print_info "Installing numpy..."
pip install numpy --quiet
print_info "Installing matplotlib..."
pip install matplotlib --quiet
print_info "Installing Pillow..."
pip install Pillow --quiet

if [ $? -eq 0 ]; then
    print_success "Python packages installed successfully"
else
    print_error "Failed to install Python packages"
    exit 1
fi
echo ""

# Step 5: Create project directory structure
echo "[5/6] Creating project directories..."
mkdir -p templates
mkdir -p test_images
mkdir -p results
mkdir -p build
mkdir -p bin
mkdir -p logs

print_success "Project directories created"
echo ""

# Step 6: Generate sample templates and test images
echo "[6/6] Generating sample data..."
cd src

if [ -f "generate_samples.py" ]; then
    python generate_samples.py
    if [ $? -eq 0 ]; then
        print_success "Sample templates and test images generated"
    else
        print_error "Failed to generate samples"
        cd ..
        exit 1
    fi
else
    print_error "generate_samples.py not found"
    cd ..
    exit 1
fi

cd ..
echo ""

# Final verification
echo "========================================================================"
echo "Setup Complete! Running verification..."
echo "========================================================================"
echo ""

cd src
python setup_check.py
SETUP_STATUS=$?
cd ..

echo ""
echo "========================================================================"
if [ $SETUP_STATUS -eq 0 ]; then
    print_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: source .venv/bin/activate"
    echo "  2. Test CPU detection (on login node):"
    echo "     cd src"
    echo "     python object_detector_cpu.py --input ../test_images/sample_test.jpg --templates ../templates"
    echo ""
    echo "  3. Submit GPU job (runs on compute node):"
    echo "     Edit submit_gpu_job.sh (set your project ID)"
    echo "     sbatch submit_gpu_job.sh"
    echo ""
    echo "  4. Monitor job:"
    echo "     squeue -u \$USER"
    echo "     tail -f logs/job_<JOBID>.out"
else
    print_error "Setup completed with warnings"
    echo "  Please resolve issues shown above before submitting GPU jobs"
fi
echo "========================================================================"
