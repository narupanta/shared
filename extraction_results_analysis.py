import jax 
import jax.numpy as jnp
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from core.model import SparseHyperelasticityGP, transform_input_features, GPRawParams, GPParams, GPWeights
from core.material_models import get_material
import jax
import jax.numpy as jnp
from core.utils import *
import datetime
import os
from tqdm import tqdm
from core.datasetclass import TractionDataset
from core.training_loop import stochastic_training_loop, deterministic_training_loop 
from core.loss_function import total_stochastic_loss, total_physical_loss, vfm_loss, ell
# from core.plotter import \
#     plot_loss_analysis, \
#     plot_parameters_hist, plot_inducing_points, plot_combined_validation
# helper: per-element edge-based neumann traction contribution
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Falls back to DejaVu if Times isn't found
    "font.size": 14,                # Base font size
    "axes.titlesize": 22,           # Subplot titles
    "axes.labelsize": 20,           # X and Y labels
    "legend.fontsize": 16,          # Legend text
    "xtick.labelsize": 16,          # Axis tick numbers
    "ytick.labelsize": 16,
    "figure.dpi": 600,              # High resolution for the screen and save
    "savefig.dpi": 600,             # Ensures saved file is high quality
    "text.usetex": False            # Set to True only if you have a full LaTeX install
})

def vfm_obs(cells, n_nodes, f_neu_nodes, node_type, piola2x2, dNdx, dA) :

    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola2x2, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)    # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2)
    f_int_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_int_cell)

    # --- Residual R = int(grad v : P) dx  -  int(v·T) ds(Neumann)
    R_nodes = f_int_nodes - f_neu_nodes
    free_node_in = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
    free_node_on_dbc_left = (node_type[:, 1] == 1)
    free_node_on_dbc_bottom = (node_type[:, 2] == 1)
    # only free DOFs contribute to the residual loss (bc == 0)
    free_dof_domain_loss = R_nodes[free_node_in]
    free_dof_on_dbc_left_loss = R_nodes[free_node_on_dbc_left, 1]
    free_dof_on_dbc_bottom_loss = R_nodes[free_node_on_dbc_bottom, 0]
    neu_nodes_right = (node_type[:, 3] == 1)
    neu_nodes_top = (node_type[:, 4] == 1)
    total_traction_force = f_neu_nodes[neu_nodes_right|neu_nodes_top].sum(axis = 0)
    fixed_nodes_loss1 = jnp.sum(R_nodes[node_type[:, 1] == 1, 0]) + total_traction_force[0]
    fixed_nodes_loss2 = jnp.sum(R_nodes[node_type[:, 2] == 1, 1]) + total_traction_force[1]

    free_x_loss = jnp.concat([free_dof_domain_loss[:, 0], free_dof_on_dbc_bottom_loss]) 
    free_y_loss = jnp.concat([free_dof_domain_loss[:, 1], free_dof_on_dbc_left_loss])

    free_loss = jnp.stack([free_x_loss, free_y_loss], axis = -1)
    fix_loss = jnp.stack([fixed_nodes_loss1, fixed_nodes_loss2])
    return f_int_nodes, f_neu_nodes, free_loss, fix_loss

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os

def plot_combined_validation(learned_gp, true_model, save_path, step):
    num_points = 50
    num_samples = 20
    max_gamma = 1.5
    gamma = jnp.linspace(0.0, max_gamma, num_points)
    
    # --- 1. Deformation Gradients & Mode Setup ---
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((num_points, 3, 3))
        arr = arr.at[:, 0, 0].set(f11); arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33); arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = jnp.stack([
        set_F(1 + gamma, 1.0, 1.0),            # Uniaxial Tension
        set_F(1 + gamma, 1 + gamma, 1.0),      # Equibiaxial Tension
        set_F(1 + gamma, 1/(1 + gamma), 1.0),  # Pure Shear
        set_F(1/(1 + gamma), 1.0, 1.0),        # Uniaxial Compression
        set_F(1/(1 + gamma), 1/(1/(1 + gamma)), 1.0), # Equibiaxial Compression
        set_F(1.0, 1.0, 1.0, f12=gamma)        # Simple Shear
    ])

    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    # --- 2. Vectorized Computations ---
    psi_vmap = jax.vmap(jax.vmap(learned_gp.psi, in_axes=(0, None)), in_axes=(0, None))
    piola_vmap = jax.vmap(jax.vmap(learned_gp.piola, in_axes=(0, None)), in_axes=(0, None))

    psi_true = jax.vmap(jax.vmap(true_model.psi))(F_all)
    P_true = jax.vmap(jax.vmap(true_model.P))(F_all)

    # Extraction for GP Stats
    psi_means, psi_vars, p_means, p_vars = [], [], [], []
    for m in range(len(mode_names)):
        psi_d = learned_gp.psi_dist(F_all[m])
        p_d = learned_gp.piola_dist(F_all[m])
        psi_means.append(psi_d.mean); psi_vars.append(psi_d.var)
        p_means.append(p_d.mean); p_vars.append(p_d.var)

    keys = jax.random.split(jax.random.PRNGKey(step), num_samples)
    psi_samples = jax.vmap(psi_vmap, in_axes=(None, 0))(F_all, keys)
    P_samples = jax.vmap(piola_vmap, in_axes=(None, 0))(F_all, keys)

    # --- 3. Figure 1: Energy Density (Psi) - BLUE ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
    fig1.suptitle(r"SEDF Distribution Extraction ($\Psi$)", fontsize=28, fontweight='bold')

    for i, name in enumerate(mode_names):
        ax = axes1[i // 3, i % 3]
        ax.set_box_aspect(1)
        ax.plot(gamma, psi_true[i], 'k--', lw=1.5, label="True", zorder=5)
        ax.plot(gamma, psi_samples[:, i, :].T, color="lightblue", lw=0.6, alpha=0.15, zorder=1)
        ax.plot(gamma, psi_means[i], color="blue", lw=1.8, label="GP Mean", zorder=3)
        ax.fill_between(gamma, psi_means[i] - 1.96*jnp.sqrt(psi_vars[i]), 
                        psi_means[i] + 1.96*jnp.sqrt(psi_vars[i]), color="blue", alpha=0.1)
        ax.set_title(name); ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\Psi$")
        ax.grid(True, alpha=0.2, ls=':'); ax.set_xlim(0, max_gamma)
        if i == 0: ax.legend(fontsize=16)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(os.path.join(save_path, "val_psi.png"), dpi=600)

    # --- 4. Figure 2: Piola Stress (P) - RED ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(r"Piola Stress Distribution ($P_{ij}$)", fontsize=28, fontweight='bold')

    for i, name in enumerate(mode_names):
        ax = axes2[i // 3, i % 3]
        ax.set_box_aspect(1)
        
        # Component indexing
        idx = (1, 1) if name == "Pure Shear" else (0, 1) if name == "Simple Shear" else (0, 0)
        label_P = r"$P_{22}$" if name == "Pure Shear" else r"$P_{12}$" if name == "Simple Shear" else r"$P_{11}$"

        p_t = P_true[i, :, idx[0], idx[1]]
        p_m = p_means[i][:, idx[0], idx[1]]
        p_s = jnp.sqrt(p_vars[i][:, idx[0], idx[1]])
        p_samp = P_samples[:, i, :, idx[0], idx[1]]

        ax.plot(gamma, p_t, 'k--', lw=1.5, label="True", zorder=5)
        ax.plot(gamma, p_samp.T, color="salmon", lw=0.6, alpha=0.15, zorder=1)
        ax.plot(gamma, p_m, color="red", lw=1.8, label="GP Mean", zorder=3)
        ax.fill_between(gamma, p_m - 1.96*p_s, p_m + 1.96*p_s, color="red", alpha=0.1)
        
        ax.set_title(name); ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(label_P)
        ax.grid(True, alpha=0.2, ls=':'); ax.set_xlim(0, max_gamma)
        if i == 0: ax.legend(fontsize=16)

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(os.path.join(save_path, "val_stress.png"), dpi=600)



import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os

def plot_int_force_distributions(f_int_nodes, mesh_pos, node_type, sigma_free, num_val_samples, save_path):
    # --- 1. Masking & Data Prep ---
    dbc_node = (node_type[:, 1] == 1) | (node_type[:, 2] == 1)
    nbc_node = (node_type[:, 3] == 1) | (node_type[:, 4] == 1)
    not_dbc_nbc = ~dbc_node & ~nbc_node
    
    # Flatten samples and loadsteps: (Samples * Loadsteps, Valid_Nodes)
    g_x = f_int_nodes[:, :, not_dbc_nbc, 0].reshape(-1, not_dbc_nbc.sum())
    g_y = f_int_nodes[:, :, not_dbc_nbc, 1].reshape(-1, not_dbc_nbc.sum())
    
    sig_x = sigma_free[not_dbc_nbc, 0] if sigma_free.ndim > 1 else sigma_free[0]
    sig_y = sigma_free[not_dbc_nbc, 1] if sigma_free.ndim > 1 else sigma_free[1]
    
    valid_positions = mesh_pos[not_dbc_nbc]
    distances = np.linalg.norm(valid_positions[:, :2], axis=1)
    sort_idx = np.argsort(distances)
    
    dist_sorted = distances[sort_idx]
    g_x_sorted = g_x[:, sort_idx]
    g_y_sorted = g_y[:, sort_idx]
    
    # --- 2. Density Plot Data Selection (r ~ 0.2) ---
    target_r = 0.2
    tol = 0.05
    mask_r = (distances >= (target_r - tol)) & (distances <= (target_r + tol))
    
    gx_r = g_x[:, mask_r].flatten()
    gy_r = g_y[:, mask_r].flatten()
    sig_x_r = np.mean(sig_x[mask_r]) if sigma_free.ndim > 1 else sig_x
    sig_y_r = np.mean(sig_y[mask_r]) if sigma_free.ndim > 1 else sig_y

    # --- 3. Plotting (3x2 Grid) ---
    # Increased global font size for better readability
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    
    def plot_hit_rate(ax, dist, force_data, sig_data, label_prefix, color):
        ci_val = 1.96 * sig_data
        within_ci = (force_data >= -ci_val) & (force_data <= ci_val)
        hit_rate = np.mean(within_ci, axis=0) * 100

        for i in range(min(force_data.shape[0], 50)):
            ax.scatter(dist, force_data[i, :], color=color, alpha=0.05, s=2)
        
        ax.fill_between(dist, -ci_val, ci_val, color="green", alpha=0.15, label='Theoretical 95% CI')
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_ylabel(f'$f_{{int, {label_prefix}}}$', fontsize=18)
        
        ax_tw = ax.twinx()
        ax_tw.scatter(dist, hit_rate, color='green', s=1.5, marker="x", label='Local Hit Rate %')
        ax_tw.set_ylim(0, 105)
        ax_tw.set_ylabel('Samples in CI (%)', fontsize=18)
        ax.set_title(f'Internal Force ({label_prefix}) Distribution vs Distance', fontsize=20)

    def plot_density(ax, data, sigma, label_prefix, title, color):
        ax.hist(data, bins=60, density=True, alpha=0.3, color=color, label='Samples Density')
        ci_bound = 1.96 * sigma
        x_range = np.linspace(-ci_bound*1.1, ci_bound*1.1, 200)
        pdf = norm.pdf(x_range, 0, sigma)
        ax.plot(x_range, pdf, 'g-', lw=2, label='Learned $N(0, \sigma^2)$')
        
        ax.axvline(ci_bound, color='green', linestyle='--', lw=1.5, label='95% CI Bound')
        ax.axvline(-ci_bound, color='green', linestyle='--', lw=1.5)
        ax.axvspan(-ci_bound, ci_bound, color='green', alpha=0.1)
        
        hit_rate = np.mean((data >= -ci_bound) & (data <= ci_bound)) * 100
        
        textstr = f'Samples in CI: {hit_rate:.2f}%'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(f'Force Value ($f_{{int, {label_prefix}}}$)', fontsize=18)
        ax.legend(fontsize=10, loc='upper left')

    # Column 0: X-Forces | Column 1: Y-Forces
    # Row 0: Hit Rates vs Distance
    plot_hit_rate(axes[0, 0], dist_sorted, g_x_sorted, sig_x, 'x', 'blue')
    plot_hit_rate(axes[0, 1], dist_sorted, g_y_sorted, sig_y, 'y', 'red')

    # Row 1: Local Density (at r = 0.2)
    plot_density(axes[1, 0], gx_r, sig_x_r, 'x', f'X-Density at r \u2248 {target_r}', 'blue')
    plot_density(axes[1, 1], gy_r, sig_y_r, 'y', f'Y-Density at r \u2248 {target_r}', 'red')

    # Row 2: Global Density
    plot_density(axes[2, 0], g_x.flatten(), np.mean(sig_x), 'x', 'Global X-Density', 'blue')
    plot_density(axes[2, 1], g_y.flatten(), np.mean(sig_y), 'y', 'Global Y-Density', 'red')

    # Set X-labels for the distance plots
    for ax in axes[0, :]:
        ax.set_xlabel('Distance from Origin $r$', fontsize=20)

    fig.suptitle("Internal Force Residual Distribution at Free Nodes", fontsize=28, y=1.02)

    plt.tight_layout()
    
    save_file = os.path.join(save_path, "internal_force_distributions.png")
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.show()
# Example call:
# plot_force_distributions(f_int_nodes, mesh_pos, node_type, 100, 'force_spatial_dist.pdf')

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os

# --- Helper Functions for Consistency ---

def plot_res_column(ax, res_data, sigma, color, label_prefix):
    """Plots scatter of residuals over steps with a 95% Confidence Interval band."""
    steps = np.arange(res_data.shape[1])
    for i in range(res_data.shape[0]):
        ax.scatter(steps, res_data[i, :], color=color, alpha=0.15, s=25)
    
    ci_bound = 1.96 * np.mean(sigma)
    ax.fill_between(steps, -ci_bound, ci_bound, color="green", alpha=0.2, label='95% CI')
    ax.axhline(0, color='black', lw=1.2, ls='--')
    
    # Step-wise hit rate calculation
    within_ci = (res_data >= -ci_bound) & (res_data <= ci_bound)
    hit_rate = np.mean(within_ci, axis=0) * 100
    for s_idx, step_val in enumerate(steps):
        ax.text(step_val, ci_bound * 1.1, f'{hit_rate[s_idx]:.0f}%', 
                color='green', fontsize=14, ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Force Residual', fontsize=20)
    ax.set_xlabel('Loadstep', fontsize=20)
    ax.legend(loc='upper left', fontsize=16)

def plot_reaction_density(ax, data, sigma, label_prefix, title, color):
    """Plots a histogram of residuals against the theoretical Gaussian distribution."""
    ax.hist(data.flatten(), bins=40, density=True, alpha=0.3, color=color, label='Samples')
    
    mean_sig = np.mean(sigma)
    ci_bound = 1.96 * mean_sig
    x_range = np.linspace(-ci_bound*1.3, ci_bound*1.3, 150)
    pdf = norm.pdf(x_range, 0, mean_sig)
    ax.plot(x_range, pdf, 'g-', lw=3, label=f'$N(0, \sigma^2)$')
    
    ax.axvline(ci_bound, color='green', ls='--', lw=2)
    ax.axvline(-ci_bound, color='green', ls='--', lw=2)
    ax.axvspan(-ci_bound, ci_bound, color='green', alpha=0.1)
    
    hit_rate = np.mean((data >= -ci_bound) & (data <= ci_bound)) * 100
    textstr = f'In CI: {hit_rate:.1f}%'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel('Residual Value', fontsize=20)
    ax.legend(fontsize=16, loc='upper left')

# --- Main Plotting Functions ---

def plot_reaction_x_2x2(f_int_left, f_neu_total, sigma_fix_x, steps, num_val_samples, save_path):
    """Generates a 2x2 summary for the X-direction (Left Boundary)."""
    res_x = f_neu_total[:, 0] + f_int_left
    step_idx = 2 if 2 in steps else min(4, len(steps)-1)

    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    
    # [0, 0] Force Balance
    for i in range(num_val_samples):
        label = "Sample $-\sum f_{int,x}$" if i == 0 else None
        axs[0, 0].scatter(steps, -f_int_left[i, :], color='blue', alpha=0.3, marker="x", s=50, label=label)
    axs[0, 0].plot(steps, f_neu_total[:, 0], color='cyan', lw=4, label='Total Traction $\sum f_{neu,x}$')
    axs[0, 0].set_title('Force Balance', fontsize=20, fontweight='bold')
    axs[0, 0].set_ylabel('Force', fontsize=20)
    axs[0, 0].set_xlabel('Loadstep', fontsize=20)
    axs[0, 0].legend(loc='lower right', fontsize=20)

    # [0, 1] Residual Scatter
    plot_res_column(axs[0, 1], res_x, sigma_fix_x, 'blue', 'x')
    axs[0, 1].set_title('Reaction Residual Coverage', fontsize=22, fontweight='bold')

    # [1, 0] Local Density
    plot_reaction_density(axs[1, 0], res_x[:, step_idx], sigma_fix_x, 'x', 
                          f'Density (Step {steps[step_idx]})', 'blue')

    # [1, 1] Global Density
    plot_reaction_density(axs[1, 1], res_x, sigma_fix_x, 'x', 'Global Residual Density', 'blue')

    fig.suptitle("Reaction Force Analysis: X-Direction (Left DBC)", fontsize=28, fontweight='bold', y=0.98)
    for ax in axs.flat: ax.grid(True, ls=':', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "reaction_x_2x2.png"), dpi=600, bbox_inches='tight')
    plt.show()

def plot_reaction_y_2x2(f_int_bottom, f_neu_total, sigma_fix_y, steps, num_val_samples, save_path):
    """Generates a 2x2 summary for the Y-direction (Bottom Boundary)."""
    res_y = f_neu_total[:, 1] + f_int_bottom
    step_idx = 2 if 2 in steps else min(4, len(steps)-1)

    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    
    # [0, 0] Force Balance
    for i in range(num_val_samples):
        label = "Sample $-\sum f_{int,y}$" if i == 0 else None
        axs[0, 0].scatter(steps, -f_int_bottom[i, :], color='red', alpha=0.3, marker="x", s=50, label=label)
    axs[0, 0].plot(steps, f_neu_total[:, 1], color='orange', lw=4, label='Total Traction $\sum f_{neu,y}$')
    axs[0, 0].set_title('Force Balance', fontsize=20, fontweight='bold')
    axs[0, 0].set_ylabel('Force', fontsize=20)
    axs[0, 0].set_xlabel('Loadstep', fontsize=20)
    axs[0, 0].legend(loc='lower right', fontsize=20)

    # [0, 1] Residual Scatter
    plot_res_column(axs[0, 1], res_y, sigma_fix_y, 'red', 'y')
    axs[0, 1].set_title('Reaction Residual Coverage', fontsize=22, fontweight='bold')

    # [1, 0] Local Density
    plot_reaction_density(axs[1, 0], res_y[:, step_idx], sigma_fix_y, 'y', 
                          f'Density (Step {steps[step_idx]})', 'red')

    # [1, 1] Global Density
    plot_reaction_density(axs[1, 1], res_y, sigma_fix_y, 'y', 'Global Y-Residual Density', 'red')

    fig.suptitle("Reaction Force Analysis: Y-Direction (Bottom DBC)", fontsize=28, fontweight='bold', y=0.98)
    for ax in axs.flat: ax.grid(True, ls=':', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "reaction_y_2x2.png"), dpi=600, bbox_inches='tight')
    plt.show()

# Example Call usage:
# plot_reaction_x_2x2(f_int_left, f_neu_total, sigma_fix_x, steps, num_val_samples, save_path)
# plot_reaction_y_2x2(f_int_bottom, f_neu_total, sigma_fix_y, steps, num_val_samples, save_path)
def plot_reaction_analysis(f_int_left, f_int_bottom, f_neu_total, sigma_fix_x, sigma_fix_y, num_val_samples, steps, save_path):
    # --- 1. Residual Calculation ---
    # f_neu + f_int should be ~0. Note: f_neu_total shape: (samples, steps, 2)
    reaction_res_x = f_neu_total[:, 0] + f_int_left
    reaction_res_y = f_neu_total[:, 1] + f_int_bottom

    # --- 2. Plotting Setup (4 Rows, 2 Columns) ---
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(4, 2, figsize=(16, 24))
    
    # Identify a specific loadstep for local density (e.g., middle or step 4)
    step_idx = 2 if 2 in steps else min(4, len(steps)-1)

    # --- Helper: Density Plotting ---
    def plot_reaction_density(ax, data, sigma, label_prefix, title, color):
        ax.hist(data.flatten(), bins=40, density=True, alpha=0.3, color=color, label='Samples')
        
        mean_sig = np.mean(sigma)
        ci_bound = 1.96 * mean_sig
        x_range = np.linspace(-ci_bound*1.2, ci_bound*1.2, 100)
        pdf = norm.pdf(x_range, 0, mean_sig)
        ax.plot(x_range, pdf, 'g-', lw=2.5, label='Learned Residual $N(0, \sigma^2)$')
        
        ax.axvline(ci_bound, color='green', ls='--', lw=1.5)
        ax.axvline(-ci_bound, color='green', ls='--', lw=1.5)
        ax.axvspan(-ci_bound, ci_bound, color='green', alpha=0.1, label='95% CI')
        
        hit_rate = np.mean((data >= -ci_bound) & (data <= ci_bound)) * 100
        textstr = f'In CI: {hit_rate:.1f}%'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(f'Residual Value ($\sum f_{{int, {label_prefix}}} + \sum f_{{neu, {label_prefix}}}$)', fontsize=13)
        ax.legend(fontsize=10, loc='upper left')

    # --- Helper: Residual Scatter/CI Plotting ---
    def plot_res_column(ax, res_data, sigma, color, label_prefix):
        for i in range(num_val_samples):
            ax.scatter(steps, res_data[i, :], color=color, alpha=0.15, s=15)
        
        ci_bound = 1.96 * sigma
        if ci_bound.ndim > 1: ci_bound = np.mean(ci_bound, axis=0)
        
        ax.fill_between(steps, -ci_bound, ci_bound, color="green", alpha=0.2, label='95% CI')
        ax.axhline(0, color='black', lw=1.0, ls='--')
        
        within_ci = (res_data >= -ci_bound) & (res_data <= ci_bound)
        hit_rate = np.mean(within_ci, axis=0) * 100
        for s_idx, step_val in enumerate(steps):
            ax.text(step_val, ci_bound * 1.05, f'{hit_rate[s_idx]:.0f}%', 
                    color='green', fontsize=12, ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Force Residual', fontsize=13)
        ax.set_xlabel('Loadstep', fontsize=13)

    # --- Row 0: Force Balance (Raw Values) ---
    # X Direction
    for i in range(num_val_samples):
        label = "Sample $-\sum f_{int}$" if i == 0 else None
        axs[0, 0].scatter(steps, -f_int_left[i, :], color='blue', alpha=0.3, marker="x", s=20, label=label)
    axs[0, 0].plot(steps, f_neu_total[:, 0], color='cyan', lw=3, label='Total Traction $\sum f_{neu}$')
    axs[0, 0].set_title('Force Balance: X-Direction', fontsize=16, fontweight='bold')
    axs[0, 0].legend(loc='lower right', fontsize=10)

    # Y Direction
    for i in range(num_val_samples):
        label = "Sample $-\sum f_{int}$" if i == 0 else None
        axs[0, 1].scatter(steps, -f_int_bottom[i, :], color='red', alpha=0.3, marker="x", s=20, label=label)
    axs[0, 1].plot(steps, f_neu_total[:, 1], color='orange', lw=3, label='Total Traction $\sum f_{neu}$')
    axs[0, 1].set_title('Force Balance: Y-Direction', fontsize=16, fontweight='bold')
    axs[0, 1].legend(loc='lower right', fontsize=10)

    # --- Row 1: Residual Coverage (Scatter + CI) ---
    plot_res_column(axs[1, 0], reaction_res_x, sigma_fix_x, 'blue', 'x')
    axs[1, 0].set_title('X Residual Coverage over Steps', fontsize=16, fontweight='bold')
    
    plot_res_column(axs[1, 1], reaction_res_y, sigma_fix_y, 'red', 'y')
    axs[1, 1].set_title('Y Residual Coverage over Steps', fontsize=16, fontweight='bold')

    # --- Row 2: Local Density (Specific Step) ---
    plot_reaction_density(axs[2, 0], reaction_res_x[:, step_idx], sigma_fix_x, 'x', f'X-Density (Step {steps[step_idx]})', 'blue')
    plot_reaction_density(axs[2, 1], reaction_res_y[:, step_idx], sigma_fix_y, 'y', f'Y-Density (Step {steps[step_idx]})', 'red')

    # --- Row 3: Global Density (All Steps) ---
    plot_reaction_density(axs[3, 0], reaction_res_x, sigma_fix_x, 'x', 'Global X-Residual Density', 'blue')
    plot_reaction_density(axs[3, 1], reaction_res_y, sigma_fix_y, 'y', 'Global Y-Residual Density', 'red')

    # Global Formatting
    fig.suptitle("Reaction Force Residual Analysis: X (Left) vs Y (Bottom)", fontsize=22, y=1.02)
    for ax in axs.flat:
        ax.grid(True, ls=':', alpha=0.6)
    
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "reaction_residual_analysis.png")
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.show()

def plot_inducing_points(dev_z, vol_z, dev_I, vol_I, save_path):
    """
    Plots standard loading paths with background training data (hollow circles)
    and specific red cross markers for inducing point locations.
    """
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Falls back to DejaVu if Times isn't found
    "font.size": 14,                # Base font size
    "axes.titlesize": 22,           # Subplot titles
    "axes.labelsize": 20,           # X and Y labels
    "legend.fontsize": 16,          # Legend text
    "xtick.labelsize": 13,          # Axis tick numbers
    "ytick.labelsize": 13,
    "figure.dpi": 600,              # High resolution for the screen and save
    "savefig.dpi": 600,             # Ensures saved file is high quality
    "text.usetex": False            # Set to True only if you have a full LaTeX install
    })
    # --- Setup Figure: 3 subplots for Invariant relationships ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5))
    num_points = 100
    gamma = jnp.linspace(0.0, 1.0, num_points)
    
    # 1. Define Deformation Gradient Modes
    modes = {
        "Uniaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Pure Shear": jnp.zeros((num_points, 3, 3)),
        "Uniaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Simple Shear": jnp.zeros((num_points, 3, 3))
    }

    # Populate F for each mode (Mechanics logic)
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
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 1].set(gamma)

    # 2. Plot Training Data (Hollow Blue Circles)
    scatter_kwargs = {
        'facecolors': 'none', 
        'edgecolors': 'blue', 
        'marker': 'o', 
        'alpha': 0.5, 
        's': 12, 
        'linewidth': 0.7,
        'label': 'Training Data'
    }
    
    axes2[0].scatter(dev_I[:, 0] - 3, dev_I[:, 1] - 3, **scatter_kwargs)
    axes2[1].scatter(dev_I[:, 0] - 3, (vol_I[:, 0] - 1)**2, **scatter_kwargs)
    axes2[2].scatter(dev_I[:, 1] - 3, (vol_I[:, 0] - 1)**2, **scatter_kwargs)

    # 3. Plot Inducing Points (Red Crosses)
    z_i1 = dev_z[:, 0] - 3
    z_i2 = dev_z[:, 1] - 3
    z_vol = (vol_z[:, 0] - 1)**2

    # Plot red crosses specifically at the z coordinates
    dev_cross_kwargs = {'color': 'black', 'marker': 'x', 's': 30, 'linewidths': 1.0, 'zorder': 4}
    vol_cross_kwargs = {'color': 'black', 'marker': 'x', 's': 30, 'linewidths': 1.0, 'zorder': 4}
    
    axes2[0].scatter(z_i1, z_i2, **dev_cross_kwargs, label='Inducing Points')
    axes2[1].scatter(z_i1, jnp.zeros_like(z_i1), **dev_cross_kwargs)
    axes2[1].scatter(np.zeros_like(z_vol), z_vol, **vol_cross_kwargs, label='Inducing Points')
    axes2[2].scatter(z_i2, jnp.zeros_like(z_i2), **dev_cross_kwargs)
    axes2[2].scatter(np.zeros_like(z_vol), z_vol, **vol_cross_kwargs)
    for i in range(len(dev_z)):
            # Plot 1: Vertical (I1) and Horizontal (I2)
            axes2[0].axvline(z_i1[i], color='black', linestyle='--', alpha=0.15, linewidth=1.2)
            axes2[0].axhline(z_i2[i], color='black', linestyle='--', alpha=0.15, linewidth=1.2)

            # Plot 2: Vertical (I1) and Horizontal (Vol)
            axes2[1].axvline(z_i1[i], color='black', linestyle='--', alpha=0.15, linewidth=1.2)
            axes2[1].axhline(z_vol[i], color='black', linestyle='--', alpha=0.15, linewidth=1.2)

            # Plot 3: Vertical (I2) and Horizontal (Vol)
            axes2[2].axvline(z_i2[i], color='black', linestyle='--', alpha=0.15, linewidth=1.2)
            axes2[2].axhline(z_vol[i], color='black', linestyle='--', alpha=0.15, linewidth=1.2)
    # 4. Plot Standard Loading Paths
    linestyles = {
        "Uniaxial Tension": "-",
        "Uniaxial Compression": "--",
        "Equibiaxial Tension": "-.",
        "Equibiaxial Compression": ":",
        "Pure Shear": (0, (3, 1, 1, 1)),
        "Simple Shear": (0, (5, 2))
    }

    for mode_name, F_stack in modes.items():
        # Note: 'invariants_and_derivatives' must be defined in your script
        i, _ = jax.vmap(invariants_and_derivatives)(F_stack)
        js = jnp.sqrt(i[:, 2])
        i1_bar = js**(-2/3) * i[:, 0]
        i2_bar = js**(-4/3) * i[:, 1]

        # Use zorder=5 to ensure load paths are on top of data and inducing points
        axes2[0].plot(i1_bar - 3, i2_bar - 3, label=mode_name, linestyle=linestyles[mode_name], linewidth=1.5, zorder=5)
        axes2[1].plot(i1_bar - 3, (js - 1)**2, linestyle=linestyles[mode_name], linewidth=1.5, zorder=5)
        axes2[2].plot(i2_bar - 3, (js - 1)**2, linestyle=linestyles[mode_name], linewidth=1.5, zorder=5)

    # 5. Labeling and Formatting
    axes2[0].set_title(r"$\bar{I}_1-3$ vs $\bar{I}_2-3$", fontsize = 16)
    axes2[1].set_title(r"$\bar{I}_1-3$ vs $(J-1)^2$", fontsize = 16)
    axes2[2].set_title(r"$\bar{I}_2-3$ vs $(J-1)^2$", fontsize = 16)

    axes2[0].set_xlabel(r"$\bar{I}_1-3$", fontsize = 14)
    axes2[1].set_xlabel(r"$\bar{I}_1-3$", fontsize = 14)
    axes2[2].set_xlabel(r"$\bar{I}_2-3$", fontsize = 14)
    axes2[0].set_ylabel(r"$\bar{I}_2-3$", fontsize = 14)
    axes2[1].set_ylabel(r"$(J-1)^2$", fontsize = 14)
    axes2[2].set_ylabel(r"$(J-1)^2$", fontsize = 14)

    fig2.suptitle("Training Datapoints, Standard Deformation paths and Optimized Inducing Points", fontsize=18)

    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', 
                bbox_to_anchor=(0.5, 0.125), # Adjust 0.05 to move up/down
                ncol=4, fontsize='small', frameon=True)
    

    for ax in axes2:
        # ax.legend(fontsize='x-small', loc='upper left', framealpha=0.8)
        # REMOVE GRID as requested
        ax.grid(False) 
        # Add thin axis lines for reference at 0,0
        ax.axhline(0, color='black', lw=0.5, alpha=0.3)
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    os.makedirs(save_path, exist_ok=True)
    fig2.savefig(os.path.join(save_path, "standard_loading_paths_inducing_red.png"), dpi=600)
    plt.show()

if __name__ == "__main__" :
    # extraction results analysis

    material_model_name = "isihara"
    disp_noise = 0.0
    load_noise = 0.005
    train_load_step_indices = [0, 5, 9]
    validation_load_step_indices = [2, 4, 6, 8]
    num_val_samples = 64

    # load result 
    analysis_dir = Path("analysis") 
    extraction_result_dir = Path("selected_model") 
    case_name = f"{material_model_name}_{disp_noise}_{load_noise}"

    # get I_obs_all.npy
    I_obs_all = np.load(extraction_result_dir / case_name / "I_obs_all.npy")
    I_z = np.load(extraction_result_dir / case_name / "I_z.npy")
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)

    # get best_params.npy
    best_raw_params = np.load(extraction_result_dir / case_name / "best_params.npy", allow_pickle=True).item()
    best_raw_params = GPRawParams(**best_raw_params)
    model = SparseHyperelasticityGP(best_raw_params, I_z, min_dev, min_vol, max_dev, max_vol)
    model.params = model.load_params(best_raw_params)
    model.gpweight = model.precompute_weights(best_raw_params)

    true_mat_model = get_material(material_model_name)
    psi_true_func = lambda f: true_mat_model.psi(f)
    piola_true_func = lambda f: true_mat_model.P(f)


    # get optimization_log.txt
    with open(extraction_result_dir / case_name / "optimization_log.txt", "r") as f:
        log = f.readlines()
    # check if analysis_dir / case_name is exist
    if not os.path.exists(analysis_dir / case_name):
        os.makedirs(analysis_dir / case_name)

    # plot_combined_validation(model, true_mat_model, analysis_dir / case_name, 0)


    # learned residual plot
    # load validation data
    data_dir = Path("precomputed_vfm") 
    npz_files = list(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in {data_dir}")
    prep_dataset_dir = data_dir / f"{material_model_name}_{disp_noise}_{load_noise}.npz"
    prep_data = jnp.load(prep_dataset_dir)

    f2x2 = prep_data["F"][validation_load_step_indices] 
    f3x3 = jax.vmap(jax.vmap(fto3x3))(f2x2)

    mesh_pos = prep_data["mesh_pos"]
    cells = prep_data["cells"]
    n_nodes = cells.max() + 1
    node_type = prep_data["node_type"]
    f_neu_nodes = prep_data["f_neu"][validation_load_step_indices] 
    dA = prep_data["dA"]
    dNdX = prep_data["dNdX"]
    
    main_key = jax.random.PRNGKey(256)
    keys = jr.split(main_key, num_val_samples)
    sampling_keys = keys[:]

    piola2x2 = lambda f,k : model.piola(f, k)[:2, :2]
    piola_cells = jax.vmap(piola2x2, in_axes=(0, None))
    piola_steps = jax.vmap(piola_cells, in_axes=(0, None))
    piola_sampling = jax.vmap(piola_steps, in_axes=(None, 0))
    piola2x2_cells = piola_sampling(f3x3, sampling_keys)
    
    per_step_res = jax.vmap(vfm_obs, in_axes=(None, None, 0, None, 0, None, None))
    per_sampling_res = jax.vmap(per_step_res, in_axes=(None, None, None, None, 0, None, None))(cells, n_nodes, f_neu_nodes, node_type, piola2x2_cells, dNdX, dA)
    f_int_nodes, f_neu_nodes_, _, _= per_sampling_res

    sigma_free_x = model.params.sigma_free_x * jnp.ones((n_nodes, 1))
    sigma_free_y = model.params.sigma_free_y * jnp.ones((n_nodes, 1))

    sigma_free = jnp.concatenate([sigma_free_x, sigma_free_y], axis=1)
    # plot_int_force_distributions(f_int_nodes, mesh_pos, node_type, sigma_free, num_val_samples, analysis_dir / case_name)

    # plot internal force along dbc compare with neumann contribution per load step

    left_dbc = node_type[:, 1] == 1
    bottom_dbc = node_type[:, 2] == 1
    right_nbc = node_type[:, 3] == 1
    top_nbc = node_type[:, 4] == 1

    f_int_left = jnp.sum(f_int_nodes[:, :, left_dbc, 0], axis=2)
    f_int_bottom = jnp.sum(f_int_nodes[:, :, bottom_dbc, 1], axis = 2)

    f_neu_total = jnp.sum(f_neu_nodes[:, right_nbc|top_nbc, :], axis=1)

    # sigma_fix_x = model.params.sigma_fix_x
    # sigma_fix_y = model.params.sigma_fix_y
    
    sigma_fix_x = 0.05 * 0.95
    sigma_fix_y = 0.05
    # reaction_res_x = f_neu_total[:, 0] - f_int_left
    # reaction_res_y = f_neu_total[:, 1] - f_int_bottom
    # plot_reaction_analysis(f_int_left, f_int_bottom, f_neu_total, sigma_fix_x, sigma_fix_y, num_val_samples, jnp.array(validation_load_step_indices), analysis_dir / case_name)
    plot_reaction_x_2x2(f_int_left, f_neu_total, sigma_fix_x, jnp.array(validation_load_step_indices), num_val_samples, analysis_dir / case_name)
    plot_reaction_y_2x2(f_int_bottom, f_neu_total, sigma_fix_y, jnp.array(validation_load_step_indices), num_val_samples, analysis_dir / case_name)
    # plot training progress analysis -> how inducing points move 
    best_dev_z = model.params.dev_z
    best_vol_z = model.params.vol_z
    dev, vol = jax.vmap(jax.vmap(transform_input_features))(I_obs_all)
    dev_flat =  dev.reshape(-1, dev.shape[-1]) 
    vol_flat = vol.reshape(-1, vol.shape[-1])
    # plot_inducing_points(best_dev_z, best_vol_z, dev_flat, vol_flat, analysis_dir / case_name)

    pass

