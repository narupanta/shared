#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=gen-ext-dist-val
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere

set -e

# Configuration File & Argument Parsing
YAML_INPUT="distill_pipeline_config.yaml"
SKIP_VAL=false

for arg in "$@"; do
    if [ "$arg" == "--skip-val" ] || [ "$arg" == "--no-val" ] || [ "$arg" == "--skip-validation" ]; then
        SKIP_VAL=true
    elif [[ "$arg" != --* ]]; then
        YAML_INPUT="$arg"
    fi
done

# Recipe shortcut resolution (e.g. 'isihara', 'gentthomas', 'nh2', 'nh4')
if [ -f "$YAML_INPUT" ]; then
    YAML_FILE="$YAML_INPUT"
elif [ -f "configs/recipes/${YAML_INPUT}.yaml" ]; then
    YAML_FILE="configs/recipes/${YAML_INPUT}.yaml"
elif [ -f "configs/recipes/${YAML_INPUT}_benchmark.yaml" ]; then
    YAML_FILE="configs/recipes/${YAML_INPUT}_benchmark.yaml"
else
    echo "❌ Error: Configuration file or recipe '$YAML_INPUT' not found!"
    echo "Available recipes in configs/recipes/:"
    ls -1 configs/recipes/*.yaml 2>/dev/null || echo "  (none)"
    exit 1
fi

YAML_SKIP=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('skip_validation', False) or d.get('run_validation', True) is False or str(d.get('run_validation', 1)) == '0' or str(d.get('skip_validation', 0)) == '1')" 2>/dev/null || echo "False")
if [ "$YAML_SKIP" == "True" ]; then
    SKIP_VAL=true
fi

echo "========================================================================"
echo "=== Loading configuration from $YAML_FILE ==="
echo "========================================================================"

# Helper function to extract scalar values from YAML
get_yaml() {
  python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d$1)"
}

# 1. Dataset Generation Config
MODEL=$(get_yaml "['material_model_name']")
D_NOISE=$(get_yaml "['disp_noise']")
L_NOISE=$(get_yaml "['load_noise']")
ASYM=$(get_yaml "['asym_factor']")
TOP_LOAD=$(get_yaml "['target_load_true_top']")
STEPS=$(get_yaml "['n_loadsteps']")

# 2. UGP Extraction Config
MCI_SAMPLING=$(get_yaml "['number_of_mci_sampling']")
N_IP=$(get_yaml "['n_ip']")
BETA=$(get_yaml "['beta']")
FIXED_NOISE=$(get_yaml "['is_fixed_reaction_force_noise']")
PRIOR_MEAN=$(get_yaml "['is_include_prior_mean']")
EXT_ITERS=$(get_yaml "['extraction_n_iterations']")
EXT_LR=$(get_yaml "['extraction_learning_rate']")
TRAIN_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(*(d['train_load_steps_indices']))")

# 3. Distillation Config
DIST_MODEL=$(get_yaml "['distilled_material_model']")
DIST_ITERS=$(get_yaml "['distillation_n_iterations']")
DIST_TARGET=$(get_yaml "['distill_target']")
SAMPLE_MODE=$(get_yaml "['sample_mode']")
NUM_POINTS=$(get_yaml "['num_points']")
MAX_GAMMA=$(get_yaml "['max_gamma']")
DO_SENSITIVITY=$(get_yaml "['do_sensitivity']")
SOBOL_THRESHOLD=$(get_yaml "['sobol_threshold']")
SOBOL_FACTOR=$(get_yaml "['sobol_samples_factor']")

SENSITIVITY_FLAG=""
if [ "$DO_SENSITIVITY" == "0" ] || [ "$DO_SENSITIVITY" == "False" ] || [ "$DO_SENSITIVITY" == "false" ]; then
    SENSITIVITY_FLAG="--no_sensitivity"
fi

# 4. Validation Config
VAL_SAMPLES=$(get_yaml "['val_number_samples']")


echo "--- Step 1: Sequential Data Generation ($MODEL) ---"
python3 dataset/synthetic/force_control/syn_force_control.py \
    --model "$MODEL" \
    --disp_noise "$D_NOISE" \
    --load_noise "$L_NOISE" \
    --target_top "$TOP_LOAD" \
    --asym "$ASYM" \
    --n_steps "$STEPS"
echo "✅ Step 1 (Data Generation) completed."


echo "--- Step 2: UGP Extraction Training ($MODEL, $EXT_ITERS iterations) ---"
python3 extraction/train_unsupervised.py \
    --material_model_name "$MODEL" \
    --number_of_mci_sampling "$MCI_SAMPLING" \
    --train_load_steps_indices $TRAIN_INDICES \
    --n_ip "$N_IP" \
    --beta "$BETA" \
    --is_fixed_reaction_force_noise "$FIXED_NOISE" \
    --n_iterations "$EXT_ITERS" \
    --disp_noise "$D_NOISE" \
    --load_noise "$L_NOISE" \
    --target_load_true_top "$TOP_LOAD" \
    --asym_factor "$ASYM" \
    --learning_rate "$EXT_LR"

SAVED_DIR=$(ls -td extraction/extracted_models/*_${MODEL}_* | head -n 1)
if [ -z "$SAVED_DIR" ] || [ ! -d "$SAVED_DIR" ]; then
    echo "❌ Error: Could not locate extracted model directory in extraction/extracted_models/"
    exit 1
fi
MODEL_PATH=$(basename "$SAVED_DIR")
echo "✅ Step 2 (Extraction) finished. Model saved at: $SAVED_DIR (Folder: $MODEL_PATH)"


echo "--- Step 3: Distillation ($DIST_MODEL candidate expression, $DIST_ITERS iterations) ---"
python3 distillation/distill_uqmodeldisc.py \
    --saved_model_dir "$SAVED_DIR" \
    --material_model "$DIST_MODEL" \
    --n_iterations "$DIST_ITERS" \
    --distill_target "$DIST_TARGET" \
    --sample_mode "$SAMPLE_MODE" \
    --num_points "$NUM_POINTS" \
    --max_gamma "$MAX_GAMMA" \
    --sobol_threshold "$SOBOL_THRESHOLD" \
    --sobol_samples_factor "$SOBOL_FACTOR" \
    $SENSITIVITY_FLAG

DISTILLED_DIR=$(ls -td distillation/distilled_models/* | head -n 1)
if [ -z "$DISTILLED_DIR" ] || [ ! -d "$DISTILLED_DIR" ]; then
    echo "❌ Error: Could not locate distilled model directory in distillation/distilled_models/"
    exit 1
fi
echo "✅ Step 3 (Distillation) finished. Distilled model saved at: $DISTILLED_DIR"


if [ "$SKIP_VAL" == "true" ]; then
    echo "⏭️ Skipping Step 4 (Stochastic Forward Sampling Validation) as requested."
    echo "🎉 Pipeline (Datagen -> Extraction -> Distillation) Completed Successfully!"
    exit 0
fi

echo "--- Step 4: Stochastic Forward Sampling with Distilled Model ---"

# Restrict each process to a sensible number of threads
export OMP_NUM_THREADS=4 

# Start Validation in the background
python3 validation/forward_fem_distilled_piola_sample.py \
    --model_path "$MODEL_PATH" \
    --distilled_dir "$DISTILLED_DIR" \
    --material_model "$DIST_MODEL" \
    --n_sample "$VAL_SAMPLES" > validation_distilled.log 2>&1 &
PID_VAL=$!

# Start Analysis in the background
python3 validation/forward_fem_distilled_piola_traction_sample.py \
    --model_path "$MODEL_PATH" \
    --distilled_dir "$DISTILLED_DIR" \
    --material_model "$DIST_MODEL" \
    --n_sample "$VAL_SAMPLES" > analysis_distilled.log 2>&1 &
PID_ANA=$!

echo "Processes started: Validation (PID: $PID_VAL) and Analysis (PID: $PID_ANA)"
echo "Logs are being written to validation_distilled.log and analysis_distilled.log..."

# Wait for both background processes to finish
wait $PID_VAL $PID_ANA

echo "🎉 Full Pipeline (Datagen -> Extraction -> Distillation -> Distilled Forward FEM) Completed Successfully!"
