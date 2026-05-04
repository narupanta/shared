#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=forward_para
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere


singularity exec --nv /home/y0113799/container/ma.sif python3 fem_testing_geometry.py