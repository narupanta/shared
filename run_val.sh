#!/bin/bash

MODEL_PATH=20260410T090545_isihara_0.0_0.01_5_20_1_1
# VAL_SAMPLES=5
VAL_INDICES="9"
# 2. Start Validation in the background
python3 uq_verification_disp.py \
    --model_path $MODEL_PATH \
    --validation_load_step_indices $VAL_INDICES
PID_VAL=$!  # Save the Process ID of the validation task

# 3. Start Analysis in the background
python3 uq_verification_energy.py \
    --model_path $MODEL_PATH \
    --validation_load_step_indices $VAL_INDICES
PID_ANA=$!  # Save the Process ID of the analysis task

echo "Processes started: disp val (PID: $PID_VAL) and energy val (PID: $PID_ANA)"

# 4. Wait for both background processes to finish
wait $PID_VAL $PID_ANA
