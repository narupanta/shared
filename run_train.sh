#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=25:00:00
#SBATCH --job-name=ma_unsup_train
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de
#SBATCH --mail-type=END,FAIL


singularity exec --nv /home/y0113799/container/ma.sif python -u main_SEDFSVGP.py
