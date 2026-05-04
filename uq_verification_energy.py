
# Import some useful modules.
import jax
import jax.numpy as jnp
import os
from pathlib import Path
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
import jax.random as jr 
jax.config.update("jax_enable_x64", True)
import argparse
from core.utils import *
from core.model import SparseHyperelasticityGP, GPParams, GPRawParams
from core.material_models import get_material
from core.datasetclass import BenchmarkDataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score


def plot_combined_validation(learned_gp, true_model, save_path, step):
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "font.size": 16,                # Slightly reduced for wider layout
        "axes.titlesize": 18,           # Adjusted for smaller subplots
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "text.usetex": False
    })
    
    num_points = 512
    num_samples = 32
    gamma = jnp.linspace(0.0, 1.0, num_points)
    
    # --- 1. Deformation Gradients (Logic remains same) ---
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((num_points, 3, 3))
        arr = arr.at[:, 0, 0].set(f11); arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33); arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = jnp.zeros((6, num_points, 3, 3))
    F_all = F_all.at[0].set(set_F(1 + gamma, 1.0, 1.0))            
    F_all = F_all.at[1].set(set_F(1 + gamma, 1 + gamma, 1.0))    
    F_all = F_all.at[2].set(set_F(1 + gamma, 1/(1 + gamma), 1.0)) 
    F_all = F_all.at[3].set(set_F(1/(1 + gamma), 1.0, 1.0))       
    F_all = F_all.at[4].set(set_F(1/(1 + gamma), 1/(1 + gamma), 1.0)) 
    F_all = F_all.at[5].set(set_F(1.0, 1.0, 1.0, f12=gamma))      

    mode_names = ["UT", "EBT", "PS", 
                  "UC", "EBC", "SS"]

    # --- 2. Vectorized Computations ---
    psi_vmap = jax.vmap(jax.vmap(learned_gp.psi, in_axes=(0, None)), in_axes=(0, None))
    piola_vmap = jax.vmap(jax.vmap(learned_gp.piola, in_axes=(0, None)), in_axes=(0, None))

    psi_true = jax.vmap(true_model.psi)(F_all)
    P_true = jax.vmap(jax.vmap(true_model.P))(F_all)
    
    psi_dist_mean = [learned_gp.psi_dist(F_all[m]).mean for m in range(6)]
    psi_dist_var = [learned_gp.psi_dist(F_all[m]).var for m in range(6)]
    P_dist_mean = [learned_gp.piola_dist(F_all[m]).mean for m in range(6)]
    P_dist_var = [learned_gp.piola_dist(F_all[m]).var for m in range(6)]

    keys = jax.random.split(jax.random.PRNGKey(step), num_samples)
    psi_samples = jax.vmap(psi_vmap, in_axes=(None, 0))(F_all, keys)
    P_samples = jax.vmap(piola_vmap, in_axes=(None, 0))(F_all, keys)

    # --- 3. Plotting (2 Rows, 6 Columns) ---
    fig, axes = plt.subplots(2, 6, figsize=(24/1.5, 8/1.5)) # Wider aspect ratio

    def calc_coverage_pct(true, mean, var):
        std = jnp.sqrt(var)
        is_inside = jnp.logical_and(true >= mean - 1.96*std, true <= mean + 1.96*std)
        return jnp.mean(is_inside) * 100

    cov_psis, cov_ps = [], []

    for i, name in enumerate(mode_names):
        # Determine Stress Component
        if name == "PS": idx_comp = (1, 1); label_P = r"$P_{22}$"
        elif name == "SS": idx_comp = (0, 1); label_P = r"$P_{12}$"
        else: idx_comp = (0, 0); label_P = r"$P_{11}$"

        # ROW 0: Energy
        ax_psi = axes[0, i]
        c_psi = calc_coverage_pct(psi_true[i], psi_dist_mean[i], psi_dist_var[i])
        cov_psis.append(c_psi)
        
        ax_psi.plot(gamma, psi_true[i], 'k--', lw=1.5, label="True", zorder=5)
        ax_psi.plot(gamma, psi_samples[:, i, :].T, color="lightblue", lw=0.6, alpha=0.3, zorder=1)
        ax_psi.plot(gamma, psi_dist_mean[i], color="blue", lw=2, label="GP Mean", zorder=3)
        ax_psi.fill_between(gamma, psi_dist_mean[i] - 1.96*jnp.sqrt(psi_dist_var[i]), 
                           psi_dist_mean[i] + 1.96*jnp.sqrt(psi_dist_var[i]), color="blue", alpha=0.1)
        
        ax_psi.set_title(f"{name}\n$\Psi$", pad=10)
        ax_psi.text(0.05, 0.85, f"Cov: {c_psi:.1f}%", transform=ax_psi.transAxes, fontsize=16, 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))

        # ROW 1: Piola Stress
        ax_p = axes[1, i]
        p_t = P_true[i, :, idx_comp[0], idx_comp[1]]
        p_m = P_dist_mean[i][:, idx_comp[0], idx_comp[1]]
        p_v = P_dist_var[i][:, idx_comp[0], idx_comp[1]]
        c_p = calc_coverage_pct(p_t, p_m, p_v)
        cov_ps.append(c_p)
        
        ax_p.plot(gamma, p_t, 'k--', lw=1.5, label="True", zorder=5)
        ax_p.plot(gamma, P_samples[:, i, :, idx_comp[0], idx_comp[1]].T, color="salmon", lw=0.6, alpha=0.3, zorder=1)
        ax_p.plot(gamma, p_m, color="red", lw=2, label="GP Mean", zorder=3)
        ax_p.fill_between(gamma, p_m - 1.96*jnp.sqrt(p_v), p_m + 1.96*jnp.sqrt(p_v), color="red", alpha=0.1)
        
        ax_p.set_title(f"{label_P}")
        ax_p.text(0.05, 0.85, f"Cov: {c_p:.1f}%", transform=ax_p.transAxes, fontsize=16, 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

        # Common Formatting
        for row, ax in enumerate([ax_psi, ax_p]):
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.2)
        ax_p.set_xlabel(r"$\gamma$")
            # if i == 0: # Only label Y for the first column
            #     ax.set_ylabel(r"$\Psi$" if row == 0 else label_P)
            # if i == 5 and row == 0: # Place legend on a corner plot
            #     ax.legend(loc='lower right')
    # h_true, _ = axes[0, 0].get_legend_handles_labels() 
    
    # Manually create labels for the combined legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, ls='--', label='True'),
        Line2D([0], [0], color='blue', lw=2, label=r'Pred Mean $\Psi$'),
        Line2D([0], [0], color='red', lw=2, label=r'Pred Mean $P_{ij}$')
    ]

    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), # Adjusted to sit above titles
               ncol=3, frameon=True, columnspacing=3.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_file = os.path.join(save_path, "psi_piola_validation.pdf")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()

# def plot_combined_validation(learned_gp, true_model, save_path, step):
#     plt.rcParams.update({
#     "mathtext.fontset": "stix",
    
#     # 2. Set the main font to STIX or Times New Roman
#     "font.family": "serif",
#     "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
#     "font.size": 14,                # Base font size
#     "axes.titlesize": 22,           # Subplot titles
#     "axes.labelsize": 20,           # X and Y labels
#     "legend.fontsize": 16,          # Legend text
#     "xtick.labelsize": 13,          # Axis tick numbers
#     "ytick.labelsize": 13,
#     "figure.dpi": 600,              # High resolution for the screen and save
#     "savefig.dpi": 600,             # Ensures saved file is high quality
#     "text.usetex": True            # Set to True only if you have a full LaTeX install
#     })
#     num_points = 512
#     num_samples = 20
#     max_gamma = 1.0
#     gamma = jnp.linspace(0.0, max_gamma, num_points)
    
#     # --- 1. Deformation Gradients & Mode Setup ---
#     def set_F(f11, f22, f33, f12=0.0):
#         arr = jnp.zeros((num_points, 3, 3))
#         arr = arr.at[:, 0, 0].set(f11); arr = arr.at[:, 1, 1].set(f22)
#         arr = arr.at[:, 2, 2].set(f33); arr = arr.at[:, 0, 1].set(f12)
#         return arr

#     # Defining the standard deformation modes
#     F_all = jnp.stack([
#         set_F(1 + gamma, 1.0, 1.0),                   # Uniaxial Tension
#         set_F(1 + gamma, 1 + gamma, 1.0),             # Equibiaxial Tension
#         set_F(1 + gamma, 1/(1 + gamma), 1.0),         # Pure Shear
#         set_F(1/(1 + gamma), 1.0, 1.0),               # Uniaxial Compression
#         set_F(1/(1 + gamma), 1/(1/(1 + gamma)), 1.0),# Equibiaxial Compression
#         set_F(1.0, 1.0, 1.0, f12=gamma)               # Simple Shear
#     ])

#     mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
#                   "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

#     # --- 2. Vectorized Computations ---
#     psi_vmap = jax.vmap(jax.vmap(learned_gp.psi, in_axes=(0, None)), in_axes=(0, None))
#     piola_vmap = jax.vmap(jax.vmap(learned_gp.piola, in_axes=(0, None)), in_axes=(0, None))

#     psi_true = jax.vmap(jax.vmap(true_model.psi))(F_all)
#     P_true = jax.vmap(jax.vmap(true_model.P))(F_all)

#     # Extraction for GP Stats
#     psi_means, psi_vars, p_means, p_vars = [], [], [], []
#     for m in range(len(mode_names)):
#         psi_d = learned_gp.psi_dist(F_all[m])
#         p_d = learned_gp.piola_dist(F_all[m])
#         psi_means.append(psi_d.mean); psi_vars.append(psi_d.var)
#         p_means.append(p_d.mean); p_vars.append(p_d.var)

#     keys = jax.random.split(jax.random.PRNGKey(step), num_samples)
#     psi_samples = jax.vmap(psi_vmap, in_axes=(None, 0))(F_all, keys)
#     P_samples = jax.vmap(piola_vmap, in_axes=(None, 0))(F_all, keys)

#     # Helper function for coverage calculation
#     def calc_coverage_pct(true, mean, var):
#         std = jnp.sqrt(var)
#         upper = mean + 1.96 * std
#         lower = mean - 1.96 * std
#         is_inside = jnp.logical_and(true >= lower, true <= upper)
#         return jnp.mean(is_inside) * 100

#     # --- 3. Figure 1: Energy Density (Psi) ---
#     fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
#     # fig1.suptitle(r"SEDF Distribution Extraction ($\Psi$)", fontsize=28, fontweight='bold')
#     cov_psis = []
#     for i, name in enumerate(mode_names):
#         ax = axes1[i // 3, i % 3]
#         ax.set_box_aspect(1)
        
#         # Calculate coverage for this specific mode
#         cov_psi = calc_coverage_pct(psi_true[i], psi_means[i], psi_vars[i])
#         cov_psis.append(cov_psi)

#         ax.plot(gamma, psi_true[i], 'k--', lw=1.5, label="True", zorder=5)
#         ax.plot(gamma, psi_samples[:, i, :].T, color="lightblue", lw=0.6, alpha=0.15, zorder=1)
#         ax.plot(gamma, psi_means[i], color="blue", lw=1.8, label="GP Mean", zorder=3)
#         ax.fill_between(gamma, psi_means[i] - 1.96*jnp.sqrt(psi_vars[i] + 1e-8), 
#                         psi_means[i] + 1.96*jnp.sqrt(psi_vars[i] + 1e-8), color="blue", alpha=0.1)
        
#         ax.set_title(name)
#         ax.set_xlabel(r"γ"); ax.set_ylabel(r"Ψ")
#         ax.grid(True, alpha=0.2, ls=':'); ax.set_xlim(0, max_gamma)
        
#         # Add Coverage Text Box
#         ax.text(0.05, 0.92, f"Coverage: {cov_psi:.1f}%", transform=ax.transAxes, 
#                 fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))

#         if i == 0: ax.legend(fontsize=14)
#     # np.array(cov_psis).mean()
#     fig1.suptitle(r"SEDF Distribution Extraction (Ψ)" + f" Coverage: {np.array(cov_psis).mean():.1f}%", 
#                   fontsize=22)

#     fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
#     fig1.savefig(os.path.join(save_path, f"val_psi_step_{step}.pdf"), dpi=300)

#     # --- 4. Figure 2: Piola Stress (P) ---
#     fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
#     # fig2.suptitle(r"Piola Stress Distribution ($P_{ij}$)", fontsize=28, fontweight='bold')
#     cov_ps = []
#     for i, name in enumerate(mode_names):
#         ax = axes2[i // 3, i % 3]
#         ax.set_box_aspect(1)
        
#         # Component indexing logic
#         idx = (1, 1) if name == "Pure Shear" else (0, 1) if name == "Simple Shear" else (0, 0)
#         label_P = r"$P_{22}$" if name == "Pure Shear" else r"$P_{12}$" if name == "Simple Shear" else r"$P_{11}$"

#         p_t = P_true[i, :, idx[0], idx[1]]
#         p_m = p_means[i][:, idx[0], idx[1]]
#         p_v = p_vars[i][:, idx[0], idx[1]]
#         p_s = jnp.sqrt(p_v)
#         p_samp = P_samples[:, i, :, idx[0], idx[1]]

#         # Calculate coverage for this specific stress component
#         cov_p = calc_coverage_pct(p_t, p_m, p_v)
#         cov_ps.append(cov_p)

#         ax.plot(gamma, p_t, 'k--', lw=1.5, label="True", zorder=5)
#         ax.plot(gamma, p_samp.T, color="salmon", lw=0.6, alpha=0.15, zorder=1)
#         ax.plot(gamma, p_m, color="red", lw=1.8, label="GP Mean", zorder=3)
#         ax.fill_between(gamma, p_m - 1.96*p_s, p_m + 1.96*p_s, color="red", alpha=0.1)
        
#         ax.set_title(name)
#         ax.set_xlabel(r"γ"); ax.set_ylabel(label_P)
#         ax.grid(True, alpha=0.2, ls=':'); ax.set_xlim(0, max_gamma)
        
#         # Add Coverage Text Box
#         ax.text(0.05, 0.92, f"Coverage: {cov_p:.1f}%", transform=ax.transAxes, 
#                 fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))
        
#         if i == 0: ax.legend(fontsize=14)
#     fig2.suptitle(r"Piola Stress Distribution ($P_{ij}$)"  + f" Coverage: {np.array(cov_ps).mean():.1f}%", fontsize=22)

#     fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
#     fig2.savefig(os.path.join(save_path, f"val_stress_step_{step}.pdf"), dpi=300)
    
#     plt.close('all')
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_full_piola_validation(learned_gp, true_model, save_path, step):
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Falls back to DejaVu if Times isn't found
    "font.size": 12,                # Base font size
    "axes.titlesize": 18,           # Subplot titles
    "axes.labelsize": 12,           # X and Y labels
    "legend.fontsize": 10,          # Legend text
    "xtick.labelsize": 10,          # Axis tick numbers
    "ytick.labelsize": 10,
    "figure.dpi": 600,              # High resolution for the screen and save
    "savefig.dpi": 600,             # Ensures saved file is high quality
    "text.usetex": False            # Set to True only if you have a full LaTeX install
    })
    num_points = 512
    max_gamma = 1.0
    gamma = jnp.linspace(0.0, max_gamma, num_points)
    
    # --- 1. Deformation Gradients ---
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((num_points, 3, 3))
        arr = arr.at[:, 0, 0].set(f11); arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33); arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = jnp.stack([
        set_F(1 + gamma, 1.0, 1.0),                  # Uniaxial Tension
        set_F(1 + gamma, 1 + gamma, 1.0),            # Equibiaxial Tension
        set_F(1 + gamma, 1/(1 + gamma), 1.0),        # Pure Shear
        set_F(1/(1 + gamma), 1.0, 1.0),              # Uniaxial Compression
        set_F(1/(1 + gamma), 1/(1 + gamma), 1.0),# Equibiaxial Compression
        set_F(1.0, 1.0, 1.0, f12=gamma)              # Simple Shear
    ])

    mode_names = ["UT", "EBT", "PS", 
                  "UC", "EBC", "SS"]

    # --- 2. Stress Computations ---
    P_true = jax.vmap(jax.vmap(true_model.P))(F_all)
    
    p_means, p_vars = [], []
    for m in range(len(mode_names)):
        p_d = learned_gp.piola_dist(F_all[m])
        p_means.append(p_d.mean)
        p_vars.append(p_d.var)
    
    p_means = jnp.array(p_means) # (Modes, Points, 3, 3)
    p_vars = jnp.array(p_vars)   # (Modes, Points, 3, 3)

    # Helper for coverage
    def get_coverage(true, mean, var):
        std = jnp.sqrt(jnp.maximum(var, 1e-9))
        lower, upper = mean - 1.96 * std, mean + 1.96 * std
        is_inside = (true >= lower) & (true <= upper)
        return jnp.mean(is_inside) * 100

    # --- 3. Figure Setup ---
    fig, axes = plt.subplots(9, 6, figsize=(26/1.5, 32/1.5), sharex=True)
    avg_mode_covs = []
    for col in range(6):
        mode_covs = []
        
        # Pre-calculate individual coverages to find the average for the title
        for row in range(9):
            i, j = divmod(row, 3)
            c = get_coverage(P_true[col, :, i, j], p_means[col, :, i, j], p_vars[col, :, i, j])
            mode_covs.append(c)
        
        avg_mode_cov = sum(mode_covs) / 9.0
        avg_mode_covs.append(avg_mode_cov)
        
        for row in range(9):
            i, j = divmod(row, 3)
            ax = axes[row, col]
            
            pt = P_true[col, :, i, j]
            pm = p_means[col, :, i, j]
            ps = jnp.sqrt(jnp.maximum(p_vars[col, :, i, j], 1e-9))
            cov_val = mode_covs[row]

            # Plotting
            ax.plot(gamma, pt, 'k--', lw=1.3, label="True" if row==0 and col==0 else "")
            ax.plot(gamma, pm, color="red", lw=1.6, label="GP Mean" if row==0 and col==0 else "")
            ax.fill_between(gamma, pm - 1.96*ps, pm + 1.96*ps, color="red", alpha=0.15)

            # Subplot Coverage Text
            ax.text(0.05, 0.88, f"Cov: {cov_val:.1f}%", transform=ax.transAxes, 
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # Formatting
            if row == 0:
                ax.set_title(f"{mode_names[col]}\n(Avg: {avg_mode_cov:.1f}%)", 
                             fontsize=18, pad=15)
            if col == 0:
                ax.set_ylabel(f"$P_{{{i+1}{j+1}}}$", fontsize=16)
            
            ax.grid(True, alpha=0.25, ls=':')
    
    fig.suptitle(f"Full Piola Stress Tensor Components ($P_{{ij}}$) - Coverage: {np.array(avg_mode_covs).mean():.1f}%", 
                 fontsize=18, y=0.98)
    fig.legend(["True Model", "GP Mean (95% CI)"], loc='upper center', 
               bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=16, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fn = os.path.join(save_path, f"full_piola_9x6_step_{step}.pdf")
    fig.savefig(save_fn, dpi=200)
    plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import r2_score

def plot_energy_r2_coverage(e_true, e_pred_mean, e_pred_std, save_path=None):
    """
    Plots Predicted vs True values for scalar Energy with 95% Confidence Intervals.
    """
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "font.size": 16,                # Slightly reduced for wider layout
        "axes.titlesize": 18,           # Adjusted for smaller subplots
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 13,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "text.usetex": False
    })
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Ensure inputs are flattened for scalar plotting
    y_true = e_true.flatten()
    y_mean = e_pred_mean.flatten()
    y_std = e_pred_std.flatten()
    
    # Calculate 95% Confidence Interval (1.96 * std)
    ci_bound = 1.96 * y_std
    lower_bound = y_mean - ci_bound
    upper_bound = y_mean + ci_bound
    
    # Calculate Coverage & R2
    inside_ci = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage_pct = np.mean(inside_ci) * 100
    r2 = r2_score(y_true, y_mean)
    
    # Scatter with error bars
    ax.errorbar(y_true, y_mean, yerr=ci_bound, fmt='o', ecolor='lightblue', 
                alpha=0.5, label='Pred Mean with 95% CI', markersize=4, capsize=0)
    
    # Identity line (45 degree)
    min_val = min(y_true.min(), y_mean.min())
    max_val = max(y_true.max(), y_mean.max())
    limits = [min_val, max_val]
    ax.plot(limits, limits, 'r--', linewidth=2, label='Isoline', zorder=3)
    
    # Formatting
    ax.set_title(f'Energy Prediction Accuracy\nCoverage: {coverage_pct:.2f}% | $R^2$: {r2:.4f}', fontsize=14)
    ax.set_xlabel('True Energy ($\Psi_{true}$)', fontsize=12)
    ax.set_ylabel('Predicted Energy ($\Psi_{pred}$)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')

    # Force square aspect ratio so the 45-degree line looks correct
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        save_file = os.path.join(save_path, "energy_r2_coverage.pdf")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Energy plot saved to {save_file}")
    
    return fig, ax

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
    fig2.savefig(os.path.join(save_path, "standard_loading_paths_inducing_red.pdf"), dpi=600)
    plt.show()
    plt.close('all')
def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--model_path', type=str, default="20260418T133514_isihara_0.0_0.01_8.0_0.9_5_50.0_1_0")
    parser.add_argument('--validation_load_step_indices', type=int, nargs='+', default=[9])
    parser.add_argument('--n_sample', type=int, default=128)


    return parser.parse_args()

if __name__ == "__main__" :
    args = parse_args()
    # read file from coverage/{mat_model}_{disp_noise}_{load_noise}
    # material_model_name = "isihara"
    validation_load_step_indices = args.validation_load_step_indices
    n_sample = args.n_sample
    model_path = args.model_path
    # load dataset 
    true_data_dir = Path("precomputed_vfm")
    case_name = args.model_path
    material_model_name = case_name.split("_")[1]
    precomputed_vfm_name = f"{case_name.split("_")[1]}_{case_name.split("_")[2]}_{case_name.split('_')[3]}_{case_name.split('_')[4]}_{case_name.split('_')[5]}"
    true_data = np.load(true_data_dir / f"{precomputed_vfm_name}.npz")
    F_val = true_data["F"][validation_load_step_indices]
    F_val_3x3 = jax.vmap(jax.vmap(fto3x3))(F_val)
    I_obs_all_val, _ = jax.vmap(jax.vmap(invariants_and_derivatives))(F_val_3x3)
    I_obs_all_val_flat = I_obs_all_val.reshape(-1, 3)
    dev_val, vol_val = jax.vmap(transform_input_features)(I_obs_all_val_flat)
    # load result 
    analysis_dir = Path("direct_psi_uq_analysis") 
    extraction_result_dir = Path("saved_model") 
    save_path = analysis_dir / case_name
    save_path.mkdir(parents=True, exist_ok=True)
    # get I_obs_all.npy

    # load true model
    true_material_model = get_material(material_model_name)
    psi_true = lambda f: true_material_model.psi(f)
    # load gp model
    I_obs_all = np.load(extraction_result_dir / case_name / "I_obs_all.npy")
    I_z = np.load(extraction_result_dir / case_name / "I_z.npy")
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)

    best_raw_params = np.load(extraction_result_dir / case_name / "best_params.npy", allow_pickle=True).item()
    best_raw_params = GPRawParams(**best_raw_params)
    model = SparseHyperelasticityGP(best_raw_params, I_z, min_dev, min_vol, max_dev, max_vol)
    model.params = model.load_params(best_raw_params)
    model.gpweight = model.precompute_weights(best_raw_params)

    # psi_dist = model.psi_dist(f)
    # discretize the true model
    F_val_3x3_flat =F_val_3x3.reshape(-1, 3, 3)
    psi_true = jax.vmap(psi_true)(F_val_3x3_flat)
    psi_pred_dist = model.psi_dist(F_val_3x3_flat)
    psi_pred_mean = psi_pred_dist.mean
    psi_pred_std = jnp.sqrt(psi_pred_dist.var)




    # evaluate the distribution of the learned model at the same discretizations

    # now we get the diag term from the covariance to get variance then cal the 95% interval
    plot_energy_r2_coverage(psi_true, psi_pred_mean, psi_pred_std, save_path)
    # count the discretized points from true model if they are in the interval (just simple le, gt logic)
    plot_combined_validation(model, true_material_model, save_path, 0)
    plot_full_piola_validation(model, true_material_model, save_path, 0)
    plot_inducing_points(dev_z, vol_z, dev_val, vol_val, save_path)

    
    pass