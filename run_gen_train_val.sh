#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=gen-train-val
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere

# Configuration
YAML_FILE="train_val_config.yaml"

# Extracting values from YAML
get_yaml() {
  python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d$1)"
}

# Extracting values (use ['key'] syntax for nested values)
MODEL=$(get_yaml "['material_model_name']")
D_NOISE=$(get_yaml "['disp_noise']")
L_NOISE=$(get_yaml "['load_noise']")
ASYM=$(get_yaml "['asym_factor']")
TOP_LOAD=$(get_yaml "['target_load_true_top']")
STEPS=$(get_yaml "['n_loadsteps']")

MCI_SAMPLING=$(get_yaml "['number_of_mci_sampling']")
N_IP=$(get_yaml "['n_ip']")
BETA=$(get_yaml "['beta']")
FIXED_NOISE=$(get_yaml "['is_fixed_reaction_force_noise']")
PRIOR_MEAN=$(get_yaml "['is_include_prior_mean']")
ITERS=$(get_yaml "['n_iterations']")
LR=$(get_yaml "['learning_rate']")

# Special handling for the list: [1, 5, 9] -> 1 5 9
TRAIN_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(*(d['train_load_steps_indices']))")

# Validation Config Extraction
VAL_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(*(d['val_load_steps_indices']))")
VAL_SAMPLES=$(get_yaml "['number_samples']")


echo "--- Starting Pipeline: $MODEL ---"

# 1. Sequential: Data Generation
python3 dataset_generator_force_control.py \
    --model "$MODEL" \
    --disp_noise "$D_NOISE" \
    --load_noise "$L_NOISE" \
    --target_top "$TOP_LOAD" \
    --asym "$ASYM" \
    --n_steps "$STEPS"

if [ $? -ne 0 ]; then echo "❌ Step 1 failed"; exit 1; fi

echo "Running Step 2: Training..."
MODEL_PATH=$(python3 train_unsupervised.py \
    --number_of_mci_sampling "$MCI_SAMPLING" \
    --train_load_steps_indices $TRAIN_INDICES \
    --n_ip "$N_IP" \
    --beta "$BETA" \
    --is_fixed_reaction_force_noise "$FIXED_NOISE" \
    --is_include_prior_mean "$PRIOR_MEAN" \
    --n_iterations "$ITERS" \
    --disp_noise "$D_NOISE" \
    --load_noise "$L_NOISE" \
    --target_load_true_top "$TOP_LOAD" \
    --asym_factor "$ASYM" \
    --learning_rate "$LR" | tail -n 1)

if [ $? -ne 0 ]; then echo "Training failed"; exit 1; fi

echo "✅ Training finished. Model saved at: $MODEL_PATH"
echo "Running Step 3: Stochastic Forward Sampling for random material model and both---"

# 1. Restrict each process to a sensible number of threads
# (e.g., if you have 8 cores, giving each process 2-4 threads is ideal)
export OMP_NUM_THREADS=4 

# 2. Start Validation in the background
python3 forward_fem_piola_sample.py \
    --model_path $MODEL_PATH \
    --n_sample "$VAL_SAMPLES" &
PID_VAL=$!  # Save the Process ID of the validation task

# 3. Start Analysis in the background
python3 forward_fem_piola_traction_sample.py \
    --model_path $MODEL_PATH \
    --n_sample "$VAL_SAMPLES" &
PID_ANA=$!  # Save the Process ID of the analysis task

echo "Processes started: Validation (PID: $PID_VAL) and Analysis (PID: $PID_ANA)"
echo "Logs are being written to validation.log and analysis.log..."

# 4. Wait for both background processes to finish
wait $PID_VAL $PID_ANA
if [ $? -ne 0 ]; then echo "Stochastic Forward failed"; exit 1; fi