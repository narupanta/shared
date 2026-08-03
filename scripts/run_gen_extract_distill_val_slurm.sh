#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=gen-ext-dist-val
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de

SCRIPT_PATH="./scripts/run_gen_extract_distill_val.sh"
if [ ! -f "$SCRIPT_PATH" ] && [ -f "./run_gen_extract_distill_val.sh" ]; then
    SCRIPT_PATH="./run_gen_extract_distill_val.sh"
fi

echo "Launching pipeline via Singularity: $SCRIPT_PATH"
singularity exec --nv /home/y0113799/container/ma.sif "$SCRIPT_PATH" "$@"
