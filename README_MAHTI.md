#### Sinteractive command
```bash
sinteractive --time=05:00:00 --gres=gpu:a100:1,nvme:100 --partition=gpusmall --mem=32G --cpus-per-task=8 --pty bash
``` 
You can use partition 'gpusmall' also. Note : gputest has a limit of 15min max session.

#### Module load 
```bash
module load gcc/10.4.0 cuda/12.6.1 cmake pytorch git
```
### In root folder of this project.

#### Virtual env
```bash
python3 -m venv --system-site-packages .venv
python -m venv .venv
source .venv/bin/activate
```
#### Pip requirements
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```


#### Building the project
For building the project, you need to have [CMake](https://cmake.org/) installed.
In `cpu_implem_opencv`, `cpu_implem` and `gpu_implem` directories, run the following commands:
```bash
cmake -B build
cd build
make -j4
```

#### Running the file 
Then, for running the program in `cpu_implem` and `gpu_implem`, you can use the following command:
```bash
./program_name [--save] <image_ref_path> <image_test_path> [image_test_path2] [image_test_path3] ...
```