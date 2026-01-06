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
def plot_parameters_hist(params_hist, steps_history, save_path) :
    
    # --- FIGURE 2: Parameters (2x3 Grid) ---
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("Kernel Hyperparameters (Matern52 + Polynomial)", fontsize=14)
    axes = axes.flatten()

    keys = ["sigma_poly", "sigma_scaling", "lengthscales", "offset", "degree", "sigma_physic"]
    titles = [r"$\sigma_{poly}$", r"$\sigma_{scaling}$", "Lengthscales", "Offset", "Poly Degree", r"$\sigma_{physic}$"]

    for i, key in enumerate(keys):
        axes[i].plot(steps_history, np.array(params_hist[key]))
        axes[i].set_title(titles[i])
        axes[i].grid(True, alpha=0.3)
        if key == "sigma_physic" :
            axes[i].set_yscale('log')


    plt.tight_layout()
    fig2.savefig(os.path.join(save_path, "parameter_evolution.png"))

    # --- FIGURE 3: Inducing Latent Mean (Single plot) ---
    plt.figure(figsize=(12, 6))
    plt.plot(steps_history, np.array(params_hist["inducing_mean"]))
    plt.title("Inducing Latent Variable Mean ($g$)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "inducing_latent_mean.png"))

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
        psi_mean = psi_pred_func(F_mode, None)
        ax.plot(gamma, psi_mean, color="navy", alpha=0.9, linewidth=2, label="GP Mean", zorder=2)
        

        # Formatting
        ax.set_title(name)
        ax.set_xlabel("Deformation Measure ($\gamma$)")
        ax.set_ylabel("Energy ($\psi$)")
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0: # Only show legend on first plot to avoid clutter
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the combined validation figure
    save_file = os.path.join(save_path, "material_modes_validation.png")
    plt.savefig(save_file)
    print(f"Loading mode validation plots saved to: {save_file}")

def plot_inducing_points() :
    # 1 figure -> 2 plots
    # 1. 
    # 2. plot on I1_dev and I2_dev of . plus data points
    # 3. path of the inducings points. from first step to final step every 50 steps
    pass

    
