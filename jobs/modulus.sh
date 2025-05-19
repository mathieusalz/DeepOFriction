#!/bin/bash

#SBATCH --job-name=tensorbord-trial
#SBATCH --nodes=1
#SBATCH -p h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output tensorboard-log-%J.out

module load gcc python openmpi py-tensorflow

ipnport=$(shuf -i8000-9999 -n1)
tensorboard --logdir logs --port=${ipnport} --bind_all

python example.py
