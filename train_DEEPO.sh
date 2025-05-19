#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8

module purge
module load gcc cuda openmpi/4.1.3-cuda  

srun VENV/bin/python training.py