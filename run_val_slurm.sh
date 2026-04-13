#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=gen-train-val
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere20260413T084516_isihara_0.0_0.02_10.0_0.95_5_25.0_0_0
#SBATCH --cpus-per-task=8
singularity exec --nv /home/y0113799/container/ma.sif ./run_val.sh
