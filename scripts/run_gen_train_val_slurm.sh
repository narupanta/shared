#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=gen-train-val
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL     # Events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de
singularity exec --nv /home/y0113799/container/ma.sif ./run_gen_train_val.sh
