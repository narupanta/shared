import os
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    
    # run FEM with the learned institute law
    base_save_path = "selected_model"  # change as needed
    os.makedirs(base_save_path, exist_ok=True)

    # Select geometry of the problem 1. training geometry (quarter holeat origin) 2. testing geometry (2 eliptical holes)

    # run ground truth model

    # run n samples of strain erngy function realization through FEM simulation to get the displacement field, nodal force residual, reaction force
    
    # evaluate the displacement field mean and std
    # 
    # evaluate the nodal force residual mean and std

    # evaluate the reaction force mean and std
    # generate plots for the final loadstep
    # depends on geometry
    # 1.training geometry -> uq capability
    #   - UQ -> extract nodal force residual from FEM and compare them with learned residual distribution
    #   - UQ -> extract reaction force from FEM and compare them with the learned reaction distriburion
    #   - displacement fields plot -> pred_mean_u vs true_u and disp error vs uncertainty 
    # 2.testing geometry -> geometry generalization capability
    #   - invariant space plot -> shows that the invariant space getting from simulating FEM on testing geometry is not overlapped with training geometry or less overlapped 
    #   - displacement fields plot -> pred_mean_u vs true_u and disp error vs uncertainty 
    # 