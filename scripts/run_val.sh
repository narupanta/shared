#!/bin/bash

MODEL_PATH=20260716T151443_gentthomas_0.0001_0.01_3.0_0.95_5_80.0_1
# VAL_SAMPLES=5
TRAIN_INDICES="1 5 9"
VAL_INDICES="9"
# 2. Start Validation in the background
python3 uq_verification_disp.py \
    --model_path $MODEL_PATH \
    --validation_load_step_indices $VAL_INDICES
PID_VAL=$!  # Save the Process ID of the validation task

# 3. Start Analysis in the background
python3 uq_verification_energy.py \
    --model_path $MODEL_PATH \
    --validation_load_step_indices $TRAIN_INDICES
PID_ANA=$!  # Save the Process ID of the analysis task

echo "Processes started: disp val (PID: $PID_VAL) and energy val (PID: $PID_ANA)"

# 4. Wait for both background processes to finish
wait $PID_VAL $PID_ANA
