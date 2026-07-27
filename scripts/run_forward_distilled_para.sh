#!/bin/bash

export OMP_NUM_THREADS=4 
MODEL_PATH=20260411T115941_isihara_0.0_0.01_8_0.975_5_40.0_1_0
DISTILLED_DIR="distillation/distilled_models/20260723T084909_isihara_isihara_wasserstein"
MATERIAL_MODEL="isihara"
VAL_SAMPLES=1024

# 2. Start Validation in the background
python3 forward_fem_distilled_piola_sample.py \
    --model_path $MODEL_PATH \
    --distilled_dir $DISTILLED_DIR \
    --material_model $MATERIAL_MODEL \
    --n_sample "$VAL_SAMPLES" > validation_distilled.log 2>&1 &
PID_VAL=$!  

# 3. Start Analysis in the background
python3 forward_fem_distilled_piola_traction_sample.py \
    --model_path $MODEL_PATH \
    --distilled_dir $DISTILLED_DIR \
    --material_model $MATERIAL_MODEL \
    --n_sample "$VAL_SAMPLES" > analysis_distilled.log 2>&1 &
PID_ANA=$!  

echo "Processes started: Validation (PID: $PID_VAL) and Analysis (PID: $PID_ANA)"
echo "Logs are being written to validation_distilled.log and analysis_distilled.log..."

# 4. Wait for both background processes to finish
wait $PID_VAL $PID_ANA

echo "--- All Parallel Tasks Finished ---"
