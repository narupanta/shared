import jax 
import gpjax as gpx
import jax.numpy as jnp
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from core.model import SparseHyperelasticityGP
from core.material_models import get_material
import jax
import jax.numpy as jnp
from core.utils import *
import datetime
import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jr
import jax
import os
from core.datasetclass import TractionDataset
from core.utils import invariants_and_derivatives
from core.loss_function import physical_loss, elbo_loss
# helper: per-element edge-based neumann traction contribution
import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
def plot_loss_analysis(loss_components_hist, params_hist, steps_history, save_path) :
    fig1, axs = plt.subplots(1, 4, figsize=(22, 5))
    fig1.suptitle("Optimization Objectives and Physics Noise", fontsize=16)

    # Total Loss
    axs[0].plot(steps_history, loss_components_hist["total_loss"], 'k-')
    axs[0].set_title("Total ELBO")
    axs[0].set_yscale('symlog')

    # Log-Likelihood (Data Fit)
    axs[1].plot(steps_history, loss_components_hist["log_like"], color='blue')
    axs[1].set_title("Log-Likelihood")

    # KL Divergence (Regularization)
    axs[2].plot(steps_history, loss_components_hist["kl"], color='green')
    axs[2].set_title("KL Divergence")

    # Physics Residual & Physics Noise Scale
    axs[3].plot(steps_history, loss_components_hist["phy"], color='red', label="Residual")
    ax3_twin = axs[3].twinx()
    ax3_twin.plot(steps_history, params_hist["sigma_physic"], color='orange', linestyle='--', label=r"$\sigma_{physic}$")
    axs[3].set_title("Physics (Resid vs Noise)")
    axs[3].set_yscale('log')
    ax3_twin.set_yscale('log')
    axs[3].legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    plt.tight_layout()
    fig1.savefig(os.path.join(save_path, "loss_and_physics.png"))

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_parameters_hist(params_hist, steps_history, save_path):
    # --- FIGURE 1: Inducing Variables & Positions (2x3 Grid) ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle(r"Evolution of Inducing Variables and Positions ($Z, \mathbf{u}$)", fontsize=16)

    # ROW 0: DEVIATORIC GP
    # 0,0: Deviatoric Mean (m_dev)
    axes1[0, 0].plot(steps_history, np.array(params_hist["dev_u_mean"]))
    axes1[0, 0].set_title(r"Deviatoric Mean ($\mathbf{m}_{dev}$)")
    
    # 0,1: Deviatoric Variance (S_dev)
    axes1[0, 1].plot(steps_history, np.array(params_hist["dev_u_var"]))
    axes1[0, 1].set_title(r"Deviatoric Variance ($\mathbf{S}_{dev}$)")
    
    # 0,2: Deviatoric Inducing Positions (dev_z)
    # dev_z is (M, 2) -> We plot the first column (I1_bar) or a norm
    # Applying the softplus transformation used in your model for accurate viz
    dev_z_1 = np.array(params_hist["dev_z"])[:, :, 0]
    dev_z_2 = np.array(params_hist["dev_z"])[:, :, 1]
    axes1[0, 2].plot(steps_history, dev_z_1) 
    axes1[0, 2].plot(steps_history, dev_z_2) # Plotting I1_bar positions
    axes1[0, 2].set_title(r"Dev. Inducing Positions ($Z_{dev, I_1}$)")

    # ROW 1: VOLUMETRIC GP
    # 1,0: Volumetric Mean (m_vol)
    axes1[1, 0].plot(steps_history, np.array(params_hist["vol_u_mean"]))
    axes1[1, 0].set_title(r"Volumetric Mean ($\mathbf{m}_{vol}$)")
    
    # 1,1: Volumetric Variance (S_vol)
    axes1[1, 1].plot(steps_history, np.array(params_hist["vol_u_var"]))
    axes1[1, 1].set_title(r"Volumetric Variance ($\mathbf{S}_{vol}$)")
    
    # 1,2: Volumetric Inducing Positions (vol_z)
    actual_vol_z = np.array(params_hist["vol_z"])[:, :, 0]
    axes1[1, 2].plot(steps_history, actual_vol_z)
    axes1[1, 2].set_title(r"Vol. Inducing Positions ($Z_{vol, J}$)")

    for ax in axes1.flatten():
        ax.set_xlabel("Iteration Step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(os.path.join(save_path, "inducing_state_evolution.png"))

    # --- FIGURE 2: Kernel Hyperparameters (2x2 Grid) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("Evolution of Kernel Hyperparameters", fontsize=16)

    # 0,0: Deviatoric Lengthscales
    axes2[0, 0].plot(steps_history, np.array(params_hist["dev_gp_lengthscales"]))
    axes2[0, 0].set_title(r"Deviatoric Lengthscales ($\ell_{dev}$)")
    
    # 0,1: Deviatoric Sigma Scaling
    axes2[0, 1].plot(steps_history, np.array(params_hist["dev_gp_sigma_scaling"]))
    axes2[0, 1].set_title(r"Deviatoric Signal Scale ($\sigma_{dev}$)")
    
    # 1,0: Volumetric Lengthscales
    axes2[1, 0].plot(steps_history, np.array(params_hist["vol_gp_lengthscales"]))
    axes2[1, 1].set_yscale('log') # Useful if lengthscales vary widely
    axes2[1, 0].set_title(r"Volumetric Lengthscales ($\ell_{vol}$)")
    
    # 1,1: Volumetric Sigma Scaling
    axes2[1, 1].plot(steps_history, np.array(params_hist["vol_gp_sigma_scaling"]))
    axes2[1, 1].set_title(r"Volumetric Signal Scale ($\sigma_{vol}$)")

    for ax in axes2.flatten():
        ax.set_xlabel("Iteration Step")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(os.path.join(save_path, "hyperparameters_evolution.png"))

    # Optional: If you want to track the physics noise parameter separately:
    if "sigma_physic" in params_hist:
        plt.figure(figsize=(8, 4))
        plt.plot(steps_history, np.array(params_hist["sigma_physic"]))
        plt.title(r"Physics Residual Noise ($\sigma_{physic}$)")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, "physics_noise_evolution.png"))

    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle("Evolution of Trend Function (Mean) Parameters", fontsize=16)

    # Subplot 1: Deviatoric Trend Parameters (c20, c02, c11, c10, c01)
    dev_params = ["c10", "c01", "c20", "c02", "c11"]
    for p in dev_params:
        if p in params_hist:
            axes3[0].plot(steps_history, np.array(params_hist[p]), label=fr"${p}$")
    
    axes3[0].set_title("Deviatoric Trend Parameters")
    axes3[0].set_xlabel("Iteration Step")
    axes3[0].set_ylabel("Value")
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    # Subplot 2: Volumetric Trend Parameters (k, q)
    vol_params = ["k", "q"]
    for p in vol_params:
        if p in params_hist:
            axes3[1].plot(steps_history, np.array(params_hist[p]), label=fr"${p}$")
    
    axes3[1].set_title("Volumetric Trend Parameters")
    axes3[1].set_xlabel("Iteration Step")
    axes3[1].set_ylabel("Value")
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(os.path.join(save_path, "trend_parameters_evolution.png"))

def plot_r2_strain_energy_function(psi_pred, psi_true, psi_dev_pred, psi_dev_true, psi_vol_pred, psi_vol_true, save_path) :
    plt.figure(figsize=(8, 6))
    
    # Flatten the arrays for plotting and R2 calculation
    psi_pred_flat = psi_pred.flatten()
    psi_true_flat = psi_true.flatten()

    
# --- Combined Energy Parity Plots (1x3 Figure) ---
    fig_energy, axs = plt.subplots(1, 3, figsize=(22, 6))
    fig_energy.suptitle(f"Strain Energy Prediction Parity (R² Analysis)", fontsize=16)

    # Data configurations for the loop
    plot_configs = [
        (psi_true, psi_pred, "Total Energy", r"$\psi$"),
        (psi_dev_true, psi_dev_pred, "Deviatoric Energy", r"$\psi_{dev}$"),
        (psi_vol_true, psi_vol_pred, "Volumetric Energy", r"$\psi_{vol}$")
    ]

    for i, (true_data, pred_data, title, label) in enumerate(plot_configs):
        # Flatten and calculate R2
        t_flat = true_data.flatten()
        p_flat = pred_data.flatten()
        r2_val = jnp.corrcoef(t_flat, p_flat)[0, 1]**2
        
        # Scatter plot
        axs[i].scatter(t_flat, p_flat, alpha=0.6, s=10, color='tab:blue')
        
        # Identity line (y=x)
        min_val = min(t_flat.min(), p_flat.min())
        max_val = max(t_flat.max(), p_flat.max())
        axs[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
        
        # Formatting
        axs[i].set_title(f"{title}\n$R^2 = {r2_val:.4f}$")
        axs[i].set_xlabel(f"True {label}")
        axs[i].set_ylabel(f"Predicted {label}")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to your timestamped folder
    energy_plot_path = os.path.join(save_path, "energy_parity_combined.png")
    plt.savefig(energy_plot_path)
    print(f"Energy parity plots saved to: {energy_plot_path}")

def plot_ut_ebt_ps_uc_ebc_ss(learned_gp, true_model, save_path):
    num_points = 100
    num_samples = 200 # Number of GP posterior samples to draw
    gamma = jnp.linspace(0.0, 1.0, num_points)
    
    # Define deformation modes in a dictionary for easy iteration
    modes = {
        "Uniaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Pure Shear": jnp.zeros((num_points, 3, 3)),
        "Uniaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Simple Shear": jnp.zeros((num_points, 3, 3))
    }

    # Populate Deformation Gradients (F)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)

    modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma)
    modes["Pure Shear"] = modes["Pure Shear"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 1].set(gamma) # Standard simple shear gamma

    # JIT compile the vmapped prediction function
    psi_pred_func = jax.jit(jax.vmap(learned_gp.psi, in_axes=(0, None)))
    psi_dist = jax.jit(jax.vmap(learned_gp.psi_dist))
    # psi_mean = psi_dist.mean
    # psi_var = psi_dist.var

    # psi_var_func =  jax.jit(jax.vmap(lambda f: learned_gp.psi(f).var))
    # psi_pred_sample = jax.jit(jax.vmap(learned_gp.psi_sample, in_axes=(0, None)))

    # Create 2x3 grid for all modes
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Material Model Validation: Strain Energy Density ($\psi$) vs Standard Deformation Modes", fontsize=16)
    axes = axes.flatten()

    for idx, (name, F_mode) in enumerate(modes.items()):
        ax = axes[idx]
        
        # 1. Calculate and plot True Energy
        psi_true = true_model.phi(F_mode)
        ax.plot(gamma, psi_true, label="True", color="grey", linewidth=2.5, zorder=5)

        # 2. Plot GP Samples
        for i in range(num_samples):
            # Pass a unique key for each sample
            psi_sample = psi_pred_func(F_mode, jr.PRNGKey(i))
            label = "GP Samples" if i == 0 else None
            ax.plot(gamma, psi_sample, color="royalblue", alpha=0.1, linewidth=0.8, label=label, zorder=1)
        psi_mean = psi_dist(F_mode).mean
        psi_var = psi_dist(F_mode).var
        psi_std = jnp.sqrt(psi_var)
        lower_bound = psi_mean - 1.96 * psi_std
        upper_bound = psi_mean + 1.96 * psi_std

        # Plot the Mean

        # psi_mean = psi_pred_func(F_mode, None)
        ax.plot(gamma, psi_mean, color="navy", alpha=0.9, linewidth=2, label="GP Mean", zorder=3)

        # Plot the 95% Confidence Interval
        ax.fill_between(gamma, lower_bound, upper_bound, 
                        color="navy", alpha=0.2, label="95% CI", zorder=2)
        

        # Formatting
        ax.set_title(name)
        ax.set_xlabel("Deformation Measure ($\gamma$)")
        ax.set_ylabel("Energy ($\psi$)")
        ax.grid(True, linestyle='--', alpha=0.5)
        y_min = jnp.min(jnp.array([psi_true.min(), psi_mean.min()]))
        y_max = jnp.max(jnp.array([psi_true.max(), psi_mean.max()]))

        # Add a 10% buffer so the lines aren't touching the edge
        padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)
        if idx == 0: # Only show legend on first plot to avoid clutter
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the combined validation figure
    save_file = os.path.join(save_path, "material_modes_validation.png")
    plt.savefig(save_file)
    print(f"Loading mode validation plots saved to: {save_file}")

    
def plot_inducing_points(dev_z, vol_z, dev_I, vol_I, save_path):
    # Setup Figure 1: Inducing points in Feature Space
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: I1_dev vs I2_dev (Inducing points for the Deviatoric GP)
    
    axes1[0].scatter(dev_I[:, 0], dev_I[:, 1], marker='o', label='Invariants (Dev)')
    axes1[0].scatter(dev_z[:, 0], dev_z[:, 1], c='red', marker='x', label='Inducing Points (Dev)')
    axes1[0].set_xlabel(r"$\bar{I}_1$")
    axes1[0].set_ylabel(r"$\bar{I}_2$")
    axes1[0].set_title("Deviatoric Inducing Points")
    axes1[0].legend()

    # Plot 2: J vs -2*J (Inducing points for the Volumetric GP)
    axes1[1].scatter(vol_I[:, 0], vol_I[:, 1], marker='o', label='J and -2 * J (Vol)')
    axes1[1].scatter(vol_z[:, 0], vol_z[:, 1], c='red', marker='x', label='Inducing Points (Vol)')
    # Reference constraint line
    axes1[1].set_xlabel(r"$J$")
    axes1[1].set_ylabel(r"$-2J$")
    axes1[1].set_title("Volumetric Inducing Points")
    axes1[1].legend()
    
    fig1.savefig(os.path.join(save_path, "inducing_points_features.png"))

    # --- Setup Figure 2: Standard Load Paths ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    num_points = 100
    gamma = jnp.linspace(0.0, 1.0, num_points)
    modes = {
        "Uniaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Pure Shear": jnp.zeros((num_points, 3, 3)),
        "Uniaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Simple Shear": jnp.zeros((num_points, 3, 3))
    }

    # Populate Deformation Gradients (F)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)

    modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma)
    modes["Pure Shear"] = modes["Pure Shear"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 1].set(gamma) # Standard simple shear gamma
    # Reuse your 'modes' dictionary logic here (assuming 'modes' is accessible)
    # For brevity, we compute and plot the lines for each mode:
    axes2[0].scatter(dev_I[:, 0] - 3, dev_I[:, 1] - 3, marker='o')
    axes2[1].scatter(dev_I[:, 0] - 3, (vol_I[:, 0] - 1)**2, marker='o')
    axes2[2].scatter(dev_I[:, 1] - 3, (vol_I[:, 0] - 1)**2, marker='o')

    axes2[0].scatter(dev_z[:, 0] - 3, dev_z[:, 1] - 3, color = "red", marker='x')
    axes2[1].scatter(dev_z[:, 0] - 3, (vol_z[:, 0] - 1)**2, color = "red", marker='x')
    axes2[2].scatter(dev_z[:, 1] - 3, (vol_z[:, 0] - 1)**2, color = "red", marker='x')

    for mode_name, F_stack in modes.items():
        i, _  = jax.vmap(invariants_and_derivatives)(F_stack)
        js = jnp.sqrt(i[:, 2])
        i1_bar = js**(-2/3) * i[:, 0]
        i2_bar = js**(-4/3) * i[:, 1]
        # Plot 1: I1_bar - 3 vs I2_bar - 3
        axes2[0].plot(i1_bar - 3, i2_bar - 3, label=mode_name)

        # Plot 2: I1_bar - 3 vs (J - 1)**2
        axes2[1].plot(i1_bar - 3, (js - 1)**2, label=mode_name)
        
        # Plot 3: I2_bar - 3 vs (J - 1)**2
        axes2[2].plot(i2_bar - 3, (js - 1)**2, label=mode_name)
    # Labeling Figure 2
    axes2[0].set_title(r"$\bar{I}_1-3$ vs $\bar{I}_2-3$")
    axes2[1].set_title(r"$\bar{I}_1-3$ vs $(J-1)^2$")
    axes2[2].set_title(r"$\bar{I}_2-3$ vs $(J-1)^2$")
    
    for ax in axes2:
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    fig2.savefig(os.path.join(save_path, "standard_loading_paths.png"))


def plot_stress_validation(gp_model, true_model, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Piola Stress Validation: Model Discovery vs. True Physics", fontsize=16)
    axes = axes.flatten()
    num_points = 100
    gamma = jnp.linspace(0.0, 1.0, num_points)
    modes = {
        "Uniaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Pure Shear": jnp.zeros((num_points, 3, 3)),
        "Uniaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Simple Shear": jnp.zeros((num_points, 3, 3))
    }

    # Populate Deformation Gradients (F)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)

    modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma)
    modes["Pure Shear"] = modes["Pure Shear"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 1].set(gamma) # Standard simple shear gamma
    
    for i, (mode_name, F_stack) in enumerate(modes.items()):
        # 1. Compute Piola Stress for the whole stack
        # Piola_stress function uses jax.grad(psi)
        P_predicted = jax.vmap(lambda f: gp_model.piola(f, None))(F_stack)
        P_predicted = jnp.array(P_predicted)

        P_var = jax.vmap(lambda f: gp_model.piola_dist(f).var)(F_stack)
        P_std = jnp.sqrt(P_var)
        P_lower_bound = P_predicted - 1.96 * P_std
        P_upper_bound = P_predicted + 1.96 * P_std
        
        P_true = jax.vmap(true_model.P)(F_stack)
        # 2. Select the relevant component based on the mode
        if mode_name == "Pure Shear":
            y_pred = P_predicted[:, 1, 1] # P22
            y_true = P_true[:, 1, 1]
            lower = P_lower_bound[:, 1, 1]
            upper = P_upper_bound[:, 1, 1]
            label = r"$P_{22}$"
        elif mode_name == "Simple Shear":
            y_pred = P_predicted[:, 0, 1] # P12
            y_true = P_true[:, 0, 1]
            lower = P_lower_bound[:, 0, 1]
            upper = P_upper_bound[:, 0, 1]
            label = r"$P_{12}$"
        else:
            y_pred = P_predicted[:, 0, 0] # P11
            y_true = P_true[:, 0, 0]
            lower = P_lower_bound[:, 0, 0]
            upper = P_upper_bound[:, 0, 0]
            label = r"$P_{11}$"

        # 3. Plotting
        gamma = jnp.linspace(0, 1, len(F_stack)) # Match your gamma range
        axes[i].plot(gamma, y_pred, color='blue', label='GP Predicted')
        axes[i].fill_between(gamma, lower, upper, color='blue', alpha=0.2, label='95% CI')
        # Assuming you have ground truth stress 'y_true'
        axes[i].plot(gamma, y_true, 'k--', alpha=0.6, label='True')
        
        axes[i].set_title(mode_name)
        axes[i].set_ylabel(label)
        axes[i].set_xlabel(r"$\gamma$")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "piola_stress_validation.png"))