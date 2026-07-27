#!/bin/bash
export OMP_NUM_THREADS=4
MODEL_PATH=20260716T151443_gentthomas_0.0001_0.01_3.0_0.95_5_80.0_1

python3 forward_fem_piola_sample.py --model_path $MODEL_PATH --n_sample 1024 &
PID1=$!

python3 forward_fem_piola_traction_sample.py --model_path $MODEL_PATH --n_sample 1024 &
PID2=$!

wait $PID1 $PID2
echo "Sampling complete!"
