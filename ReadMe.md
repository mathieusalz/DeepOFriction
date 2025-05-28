# Instructions for Nvidia-PhysicsNeMo (Provided by SCITAS)

```bash
module load gcc
module load openmpi/5.0.3-cuda
module load cuda
module load cudnn/8.9.7.29-12
module load python
module load py-torch
module load py-torchvision
module load py-cython/0.29.36-rc3rfxq
module load git
module load git-lfs
module load py-pybind11
module load py-tensorflow
module list


python -m venv VENV2

source VENV2/bin/activate

[ ! -d physicsnemo-sym ] && GIT_CLONE_PROTECTION_ACTIVE=false git clone git@github.com:NVIDIA/physicsnemo-sym.git
cd physicsnemo-sym

[ -d build ] && rm -r build
```

Then, edit the setup.py to add those 2 lines:

```python
import os
nvcc_args.append(f"-I{os.environ['PY_PYBIND11_ROOT']}/include")
```

after those:
```python
nvcc_args.append("-t=0") # Enable multi-threaded builds
# nvcc_args.append("--time=output.txt")
```

Save the file and run:

```bash
pip install --upgrade pip
pip install --no-build-isolation .
```


# Instructions for running on Kuma

First, you need to get onto a processing node
```bash
Sinteract -a lsms-ddcf -p h100 -g gpu:1 -n 1 -t 00:40:00 -m 10G -q debug -c 1
```