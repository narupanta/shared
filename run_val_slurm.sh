#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=gen-train-val
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere
#SBATCH --cpus-per-task=8
singularity exec --nv /home/y0113799/container/ma.sif ./run_val.sh
