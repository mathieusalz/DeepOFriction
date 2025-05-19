# Instructions for Nvidia-PhysicsNeMo (Provided by SCITAS)

```console
module load gcc
module load openmpi/5.0.3-cuda
module load cuda
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

#python -m pip install nvidia-physicsnemo

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

```console
pip install --upgrade pip
pip install --no-build-isolation .
```
