export OMP_NUM_THREADS=4 
MODEL_PATH=20260410T172507_isihara_0.0_0.01_8_0.975_5_40_0_0
VAL_SAMPLES=5
# VAL_INDICES= 2 3 4 6 8
# 2. Start Validation in the background
python3 forward_fem_piola_sample.py \
    --model_path $MODEL_PATH \
    --n_sample "$VAL_SAMPLES" >> ffp.log 2>&1 &
PID_VAL=$!  # Save the Process ID of the validation task

# 3. Start Analysis in the background
python3 forward_fem_piola_traction_sample.py \
    --model_path $MODEL_PATH \
    --n_sample "$VAL_SAMPLES" >> ffpt.log 2>&1 &
PID_ANA=$!  # Save the Process ID of the analysis task

echo "Processes started: Validation (PID: $PID_VAL) and Analysis (PID: $PID_ANA)"
echo "Logs are being written to validation.log and analysis.log..."

# 4. Wait for both background processes to finish
wait $PID_VAL $PID_ANA

echo "--- All Parallel Tasks Finished ---"