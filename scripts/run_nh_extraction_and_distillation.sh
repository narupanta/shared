#!/bin/bash
set -e

echo "=== Step 1: Starting UGP Extraction Training (nh, load=1.5, mci=8, n_iterations=50000) ==="
python3 extraction/train_unsupervised.py \
    --material_model_name nh \
    --disp_noise 0.0001 \
    --load_noise 0.01 \
    --target_load_true_top 1.5 \
    --asym_factor 0.95 \
    --number_of_mci_sampling 8 \
    --train_load_steps_indices 1 11 20 \
    --n_ip 5 \
    --beta 80 \
    --is_fixed_reaction_force_noise 1 \
    --n_iterations 50000 \
    --learning_rate 0.01

SAVED_DIR=$(ls -td extraction/extracted_models/*_nh_* | head -n 1)

echo "=== Step 2: Extraction Complete. Starting Distillation (GMR) on $SAVED_DIR ==="
python3 distillation/distill_uqmodeldisc.py \
    --saved_model_dir "$SAVED_DIR" \
    --material_model gmr \
    --n_iterations 5000

echo "=== Pipeline Completed Successfully! ==="
