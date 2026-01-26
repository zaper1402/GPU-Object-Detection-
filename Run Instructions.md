#### Sinteractive command
```bash
sinteractive --time=05:00:00 --gres=gpu:a100:1,nvme:100 --partition=gpusmall --mem=32G --cpus-per-task=8 --pty bash
``` 
You can use partition 'gpusmall' also. Note : gputest has a limit of 15min max session.

#### Module load 
```bash
module load gcc/10.4.0 cuda/12.6.1 cmake python-data pytorch git
```

### In root folder of this project.

#### Virtual env
```bash
python -m venv .venv
source .venv/bin/activate
```

#### Pip requirements
```bash
pip install -r requirements.txt
```


### install opencv
```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
rm -rf build
mkdir build
cd build
```

#### Cmake
```bash
cmake -D CMAKE_BUILD_TYPE=Release \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="80" \
      -D OPENCV_EXTRA_MODULES_PATH=/users/akulshre/Object-Detection-with-SIFT-ORB/opencv_contrib/modules \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$HOME/.venv/bin/python \
      -D PYTHON3_INCLUDE_DIR=$HOME/.venv/include/python3.12 \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=$HOME/.venv/lib/python3.12/site-packages/numpy/core/include \
      -D PYTHON3_PACKAGES_PATH=$HOME/.venv/lib/python3.12/site-packages \
      -D WITH_VA=OFF \
      -D WITH_VA_INTEL=OFF \
      -D CMAKE_C_COMPILER=$(which gcc) \
      -D CMAKE_CXX_COMPILER=$(which g++) \
      -D CUDA_HOST_COMPILER=$(which g++) \
      -D BUILD_opencv_world=OFF \
```

Fast cmake command:
```bash
cmake ../opencv \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$HOME/opencv-install \
  -D CMAKE_SKIP_INSTALL_RPATH=ON \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=80 \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D CMAKE_C_COMPILER=$(which gcc) \
  -D CMAKE_CXX_COMPILER=$(which g++) \
  -D CUDA_HOST_COMPILER=$(which g++) \
  -D CUDA_NVCC_FLAGS="--use_fast_math" \
  -D BUILD_LIST=core,imgproc,features2d,calib3d,flann,cudev,cudaimgproc,cudafeatures2d \
  -D OPENCV_EXTRA_MODULES_PATH=/users/akulshre/Object-Detection-with-SIFT-ORB/opencv_contrib/modules \
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_EXECUTABLE=$HOME/.venv/bin/python \
  -D PYTHON3_PACKAGES_PATH=$HOME/.venv/lib/python3.12/site-packages \
  -D BUILD_opencv_world=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_DOCS=OFF \
  -D WITH_QT=OFF \
  -D WITH_GTK=OFF \
  -D WITH_FFMPEG=OFF \
  -D WITH_GSTREAMER=OFF \
  -D WITH_OPENGL=OFF \
  -D WITH_TBB=OFF \
  -D WITH_IPP=OFF \
  -D WITH_EIGEN=OFF
```

### Make build
```bash
cmake --build . -- -j$(nproc)
cmake --install .
```

Fast build :
```bash
cmake --build . -j8
cmake --install .
```

#### Commands to run each file

1. Video code original
```bash
python main_video_detection.py
```
The path of object and video are hardcoded.


2. Video code gpu
```bash
python main_video_detection_gpu.py
```




#### Other usefule cmd

To check memory hdd
```bash
df -h
```

To check permision
```bash
ls -ld <directory path>
```

To check no of cpu
```bash
echo "CPUs: $SLURM_CPUS_PER_TASK"
```