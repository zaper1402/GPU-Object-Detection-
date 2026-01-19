# Quick Start Guide - Mahti Cluster
## GPU Object Detection on CSC Mahti Supercomputer

### Prerequisites
- CSC account with Mahti access
- Project allocation with GPU hours
- SSH access to Mahti login nodes

---

## Initial Setup (10 minutes)

### 1. Connect to Mahti
```bash
ssh username@mahti.csc.fi
cd /scratch/project_XXXXX/$USER  # Use your project number
git clone <your-repo> GPU_Project  # Or upload files
cd GPU_Project
```

### 2. Run Setup Script
```bash
bash setup_mahti.sh
```

This script:
- Loads CUDA 11.5 and Python modules
- Creates virtual environment
- Installs dependencies (OpenCV, NumPy, Matplotlib)
- Generates sample templates
- Creates project directories

### 3. Verify Installation
```bash
source .venv/bin/activate
cd src
python setup_check.py
```

**Expected output:**
```
✓ All checks passed (6/6)
[SUCCESS] TEST PASSED - System ready
```

---

## Running on GPU Nodes

**Important**: GPU operations only work on compute nodes, not login nodes!

### Option 1: Submit Batch Job (Recommended)

Edit `submit_gpu_job.sh` - replace `project_2016196` with your project ID if different:
```bash
#SBATCH --account=project_XXXXX  # Your CSC project
```

Submit job:
```bash
sbatch submit_gpu_job.sh
```

Monitor:
```bash
squeue -u $USER                    # Check queue
tail -f logs/job_<JOBID>.out       # Watch output
```

### Option 2: Interactive GPU Session

```bash
sinteractive --account=project_XXXXX --partition=gputest --gres=gpu:a100:1 --time=00:30:00

# Inside interactive session:
module load cuda/11.5.0 python-data/3.10
source .venv/bin/activate
cd src
python object_detector_gpu.py --input ../test_images/sample_test.jpg --templates ../templates
```

---

## Usage Examples

### Test Single Image Detection
```bash
# In SLURM job or interactive session:
python object_detector_gpu.py \
    --input ../test_images/test_both_separated.jpg \
    --templates ../templates \
    --output ../results
```

### Batch Process Directory
```bash
python object_detector_gpu.py \
    --input ../test_images \
    --templates ../templates \
    --output ../results
```

### GPU vs CPU Benchmark
```bash
python benchmark.py \
    --input ../test_images \
    --templates ../templates \
    --output ../results/benchmark
```
### Srun 
# Quick GPU check without batch script
srun --account=project_2016196 --partition=gputest --gres=gpu:a100:1 --time=00:01:00 nvidia-smi

# Test CUDA compilation
srun --account=project_2016196 --partition=gputest --gres=gpu:a100:1 nvcc --version

# Run device query
srun --account=project_2016196 --partition=gputest --gres=gpu:a100:1 ./deviceQuery

### Test CUDA Kernels
```bash
cd /path/to/GPU_Project
make clean
make run
```

---

## Checking Results

```bash
# View output logs
cat logs/job_<JOBID>.out

# Check result images
ls -lh results/

# Download results to local machine (from your computer):
scp username@mahti.csc.fi:/scratch/project_XXXXX/$USER/GPU_Project/results/*.jpg ./
```

---

## Module Commands Reference

```bash
# List loaded modules
module list

# Load required modules
module load cuda/11.5.0
module load python-data/3.10

# Search available modules
module spider cuda
module spider python

# Unload all modules
module purge
```

---

## SLURM Job Commands

```bash
# Submit job
sbatch submit_gpu_job.sh

# Check queue
squeue -u $USER

# Cancel job
scancel <JOBID>

# Job history
sacct -u $USER --starttime=2026-01-01

# Check GPU allocation
saldo -p project_XXXXX
```

---

## Troubleshooting

### "CUDA module not loaded"
```bash
module load cuda/11.5.0
nvcc --version
```

### "nvidia-smi not found" on login node
**This is normal!** Login nodes don't have GPUs. Solutions:
1. Submit job: `sbatch submit_gpu_job.sh`
2. Interactive session: `sinteractive --gres=gpu:a100:1`

### "OpenCV CUDA not available"
On login nodes, this warning is expected. GPU features activate automatically on compute nodes.

Test CPU version first:
```bash
python object_detector_cpu.py --input ../test_images/sample_test.jpg --templates ../templates
```

### "Out of memory" errors
Reduce image resolution or batch size:
```bash
# Resize images before processing
mogrify -resize 1280x720 test_images/*.jpg
```

Or request more GPU memory:
```bash
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G  # Increase from 32G
```

### Job fails immediately
Check:
```bash
cat logs/job_<JOBID>.err
seff <JOBID>  # Job efficiency report
```

Common issues:
- Wrong project ID in `#SBATCH --account=`
- Insufficient GPU hours
- Module not loaded in job script

---

## Performance Expectations (A100 GPU)

| Metric | Value |
|--------|-------|
| Detection time | 25-35 ms/image |
| Throughput | 30-40 images/sec |
| GPU utilization | 40-60% |
| Memory usage | 250-400 MB VRAM |
| Speedup vs CPU | 5-8x |

---

## File Organization

```
/scratch/project_XXXXX/$USER/GPU_Project/
├── src/                      # Source code
├── templates/                # ball.jpg, book.jpg
├── test_images/              # Query images
├── results/                  # Detection outputs
├── logs/                     # SLURM job logs
├── build/                    # Compiled CUDA objects
├── bin/                      # Executables
├── .venv/                    # Python virtual environment
├── submit_gpu_job.sh         # SLURM batch script
└── setup_mahti.sh            # Setup script
```

---

## Best Practices

1. **Use scratch space**: `/scratch/project_XXXXX/$USER` (not home directory)
2. **Clean up**: Remove old results after downloading
3. **Test locally first**: Use CPU detector on login node before GPU jobs
4. **Monitor quotas**: Check `saldo -p project_XXXXX` regularly
5. **Optimize jobs**: Use `gputest` partition for <15min jobs, `gpusmall` for longer

---

## Resources

- **Mahti Docs**: https://docs.csc.fi/computing/systems-mahti/
- **GPU Usage**: https://docs.csc.fi/computing/running/gpu-jobs/
- **SLURM Guide**: https://docs.csc.fi/computing/running/submitting-jobs/
- **Project README**: [README.md](README.md)
- **Technical Details**: [answers.txt](answers.txt)

---

**Group U**: Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha  
**Target**: CSC Mahti - NVIDIA A100-SXM4-40GB (sm_80)
