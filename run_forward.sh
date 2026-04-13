#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=train_hydrogel
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere

# Navigate to the directory where your script is located (optional)
# cd /path/to/your/project

echo "Starting the first script..."
python forward_fem_piola_sample.py 

# The second script will only run after the first one completes
echo "First script finished. Starting the second script..."
python forward_fem_piola_traction_sample.py

echo "All tasks completed successfully."