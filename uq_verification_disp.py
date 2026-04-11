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

from core.utils import *
from core.model import SparseHyperelasticityGP, GPParams, GPRawParams
from core.material_models import get_material
from core.datasetclass import BenchmarkDataset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_comprehensive_analysis(u_true, u_pred_samples, node_type, node_to_plot, save_path):
    """
    Generates and saves two separate 2x2 figures (X and Y directions).
    Includes statistical text annotations and a zero-line for error distributions.
    """
    # Filter for free nodes
    free_nodes = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
    
    u_pred_free = u_pred_samples[:, free_nodes, :]  # [Samples, Nodes, 2]
    u_true_free = u_true[free_nodes, :]             # [Nodes, 2]
    
    num_samples, num_nodes, _ = u_pred_free.shape
    node_indices = np.arange(num_nodes)
    
    directions = [
        {'idx': 0, 'label': 'x', 'color': 'purple'},
        {'idx': 1, 'label': 'y', 'color': 'teal'}
    ]

    os.makedirs(save_path, exist_ok=True)

    for dir_info in directions:
        d = dir_info['idx']
        label = dir_info['label']
        main_color = dir_info['color']
        u_p = u_pred_samples[:, :, d]
        u_t = u_true[:, d]
        
        u_p_dim = u_pred_free[:, :, d]
        u_t_dim = u_true_free[:, d]
        
        u_std_per_node = np.std(u_p_dim, axis=0)
        u_true_repeated = np.tile(u_t_dim, num_samples)
        u_pred_flat = u_p_dim.flatten()
        errors_flat = (u_t_dim[None, :] - u_p_dim).flatten()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(wspace=0.25, hspace=0.3)
        (ax1, ax2), (ax3, ax4) = axes

        # --- PLOT 1: UNCERTAINTY (LINE CHART) ---
        ax1.plot(node_indices, u_std_per_node, color=main_color, linewidth=1.5)
        ax1.fill_between(node_indices, 0, u_std_per_node, color=main_color, alpha=0.15, label='$\sigma$ per Node')
        ax1.set_title(f'Predictive Uncertainty - {label.upper()}', fontsize=13)
        ax1.set_xlabel('Node Index')
        ax1.set_ylabel('Standard Deviation')
        ax1.grid(alpha=0.3, linestyle='--')

        # --- PLOT 2: GLOBAL ACCURACY ---
        ax2.scatter(u_true_repeated, u_pred_flat, color='seagreen', s=1, alpha=0.05)
        mn, mx = u_t_dim.min(), u_t_dim.max()
        ax2.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
        ax2.set_title(f'Sample-wise Accuracy - {label.upper()}', fontsize=13)
        ax2.set_xlabel(f'True $u_{label}$')
        ax2.set_ylabel(f'Predicted $u_{label}$')
        ax2.legend(markerscale=10)

        # --- PLOT 3: LOCAL DISTRIBUTION (Selected Node) ---
        node_samples = u_p[:, node_to_plot]
        node_true = u_t[node_to_plot]
        n_mean = np.mean(node_samples)
        n_std = np.std(node_samples)
        
        ax3.hist(node_samples, bins=40, density=True, alpha=0.3, color='dodgerblue', label='Samples')
        ax3.axvline(node_true, color='red', linestyle='--', label=f'True: {node_true:.4f}')
        
        # Add stats text to Plot 3
        stats_text_local = f'Mean: {n_mean:.4e}\nStd: {n_std:.4e}'
        ax3.text(0.05, 0.95, stats_text_local, transform=ax3.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax3.set_title(f'Local Dist. at Node {node_to_plot} - {label.upper()}', fontsize=13)
        ax3.set_xlabel('Displacement Value')
        ax3.legend()

        # --- PLOT 4: GLOBAL ERROR DISTRIBUTION ---
        e_mean = np.mean(errors_flat)
        e_std = np.std(errors_flat)
        
        ax4.hist(errors_flat, bins=60, density=True, alpha=0.3, color='orange', label='Error')
        ax4.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.6) # Vertical line at 0
        
        # Add stats text to Plot 4
        stats_text_global = f'Mean Error: {e_mean:.4e}\nError Std: {e_std:.4e}'
        ax4.text(0.05, 0.95, stats_text_global, transform=ax4.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax4.set_title(f'Global Error Distribution - {label.upper()}', fontsize=13)
        ax4.set_xlabel('Error Value (True - Pred)')
        ax4.set_ylabel('Density')

        # Save and Show
        save_file = os.path.join(save_path, f"analysis_{label}_{node_to_plot}_direction.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.show()
def plot_node_distributions(u_true, u_pred_samples, u_pred_piola_traction_samples, node_to_plot, save_path):
    """
    Plots local distributions with 95% Quantile CIs for a list of nodes.
    """
    os.makedirs(save_path, exist_ok=True)
    
    num_nodes_to_plot = len(node_to_plot)
    fig, axes = plt.subplots(num_nodes_to_plot, 2, figsize=(14, 5 * num_nodes_to_plot))
    
    if num_nodes_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    directions = [
        {'idx': 0, 'label': 'X', 'color1': 'dodgerblue', 'color2': 'violet'},
        {'idx': 1, 'label': 'Y', 'color1': 'teal', 'color2': 'green'}
    ]

    for i, node_idx in enumerate(node_to_plot):
        for d_info in directions:
            d = d_info['idx']
            ax = axes[i, d]
            
            # 1. Extract Data
            samples = u_pred_samples[:, node_idx, d]
            pt_samples = u_pred_piola_traction_samples[:, node_idx, d]
            u_true_node = u_true[node_idx, d]
            
            # 2. Calculate Robust Statistics (Quantiles)
            # Piola Samples
            p_median = np.median(samples)
            p_low, p_high = np.quantile(samples, [0.025, 0.975])
            
            # Piola Traction Samples
            pt_median = np.median(pt_samples)
            pt_low, pt_high = np.quantile(pt_samples, [0.025, 0.975])

            # 3. Plot Histograms
            ax.hist(samples, bins=40, density=True, alpha=0.3, 
                    color=d_info['color1'], label='Piola Samples')
            ax.hist(pt_samples, bins=40, density=True, alpha=0.3, 
                    color=d_info['color2'], label='Piola Traction Samples')

            # 4. Add Vertical Lines for Medians
            ax.axvline(u_true_node, color='red', linestyle='-', linewidth=2, 
                       label=f'True: {u_true_node:.4e}')
            ax.axvline(p_median, color=d_info['color1'], linestyle='--', linewidth=1.5, 
                       label=f'Piola Median: {p_median:.4e}')
            ax.axvline(pt_median, color=d_info['color2'], linestyle='--', linewidth=1.5, 
                       label=f'PT Median: {pt_median:.4e}')

            # 5. Plot Confidence Intervals as Shaded Regions (95% CI)
            ax.axvspan(p_low, p_high, color=d_info['color1'], alpha=0.1, 
                       label='Piola 95% CI')
            ax.axvspan(pt_low, pt_high, color=d_info['color2'], alpha=0.1, 
                       label='PT 95% CI')
            
            # Optional: Add faint boundary lines for the CIs
            ax.axvline(p_low, color=d_info['color1'], linestyle=':', alpha=0.5, linewidth=1)
            ax.axvline(p_high, color=d_info['color1'], linestyle=':', alpha=0.5, linewidth=1)
            
            # Formatting
            if i == 0:
                ax.set_title(f'Direction {d_info["label"]}', fontsize=15)
            if d == 0:
                ax.set_ylabel(f'Node {node_idx}\nDensity', fontsize=12, fontweight='bold')
                
            ax.legend(fontsize=7, loc='upper right', frameon=True, framealpha=0.8)
            ax.grid(alpha=0.2)

    plt.suptitle('Local Displacement Distributions: Median & 95% Quantile CI', fontsize=18, y=1.02)
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "local_node_distributions_quantile.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    # plt.show()
# Example: plot_comprehensive_analysis(u_true, u_pred_samples, 50, "plots/")

# Usage:
# plot_global_std_analysis(u_true, u_pred_samples, "results/")

# Usage:
# plot_global_samples_analysis(u_true, u_pred_samples, "results/")

# Example Usage:
# plot_global_index_analysis(u_true, u_pred_samples, "plots/")
# Example Usage:
# plot_global_analysis(u_true, u_pred_samples, node_coords, "plots/")

def plot_disp_field(node_coords, cells, u_true, u_pred_mean, u_pred_std, node_indices, save_path):
    # --- Data Preparation ---
    node_indices = np.array(node_indices)
    
    # Calculate deformed coordinates
    coords_true = node_coords + u_true
    coords_pred = node_coords + u_pred_mean
    
    # Calculate magnitudes/errors
    def get_mag(u): return np.linalg.norm(u, axis=1)
    mag_true = get_mag(u_true)
    mag_pred = get_mag(u_pred_mean)
    error = np.linalg.norm(u_true - u_pred_mean, axis=1)
    mag_std = get_mag(u_pred_std) if u_pred_std.ndim > 1 else u_pred_std

    # Extract coordinates for the red crosses
    marker_coords_true = coords_true[node_indices]
    marker_coords_pred = coords_pred[node_indices]

    fig, axes = plt.subplots(2, 2, figsize=(10, 12)) 
    plt.suptitle('Deformed Field: Accuracy & Uncertainty', fontsize=20)

    # --- Updated Helper Function with Text Annotation ---
    def add_markers_with_labels(ax, coords, indices):
        # Plot the crosses
        ax.scatter(coords[:, 0], coords[:, 1], 
                   color='red', marker='x', s=60, linewidths=1.5, 
                   label='Probe Nodes', zorder=6)
        
        # Add text labels for each node index
        for i, idx in enumerate(indices):
            ax.annotate(f'ID: {idx}', 
                        (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", 
                        xytext=(5, 5),          # Offset text by 5 points right and up
                        fontsize=9, 
                        fontweight='bold',
                        color='red',
                        zorder=7)

    # 1,1: True Material
    tri_true = tri.Triangulation(coords_true[:, 0], coords_true[:, 1], cells)
    im1 = axes[0, 0].tripcolor(tri_true, mag_true, cmap='Blues')
    add_markers_with_labels(axes[0, 0], marker_coords_true, node_indices)
    axes[0, 0].set_title('True Material')
    fig.colorbar(im1, ax=axes[0, 0])

    # 1,2: Predicted Material
    tri_pred = tri.Triangulation(coords_pred[:, 0], coords_pred[:, 1], cells)
    im2 = axes[0, 1].tripcolor(tri_pred, mag_pred, cmap='Blues')
    add_markers_with_labels(axes[0, 1], marker_coords_pred, node_indices)
    axes[0, 1].set_title('Predicted Material')
    fig.colorbar(im2, ax=axes[0, 1])

    # 2,1: Nodal error
    im3 = axes[1, 0].tripcolor(tri_pred, error, cmap='inferno')
    add_markers_with_labels(axes[1, 0], marker_coords_pred, node_indices)
    axes[1, 0].set_title(r'$||u_{true} - u_{pred}||$')
    fig.colorbar(im3, ax=axes[1, 0])

    # 2,2: Uncertainty
    im4 = axes[1, 1].tripcolor(tri_pred, mag_std, cmap='magma')
    add_markers_with_labels(axes[1, 1], marker_coords_pred, node_indices)
    axes[1, 1].set_title(r'Uncertainty ($\sigma_u$)')
    fig.colorbar(im4, ax=axes[1, 1])

    # Standardize labels
    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "displacement_analysis.png"), dpi=300, bbox_inches='tight')


import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import r2_score

def plot_disp_r2_coverage(u_true, u_pred_med, u_pred_lower, u_pred_upper, save_path, suffix="_"):
    """
    Plots Predicted vs True values for both X and Y directions in subplots.
    Expects arrays of shape (N, 2) for x and y components.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = ['X-Direction', 'Y-Direction']
    
    for i, ax in enumerate(axes):
        # Extract components
        y_true = u_true[:, i]
        y_mean = u_pred_med[:, i]
        y_lower = u_pred_lower[:, i]
        y_upper = u_pred_upper[:, i]
        
        # Calculate 95% Confidence Interval (1.96 * std)
        # ci_bound = 1.96 * y_std
        lower_bound = y_lower
        upper_bound = y_upper
        ci_bound = y_upper - y_lower
        
        # Instead of y_mean and y_std, pass the raw samples (samples_array)
        # lower_bound = np.quantile(samples_array, 0.025, axis=0)
        # upper_bound = np.quantile(samples_array, 0.975, axis=0)
        # y_mean = np.median(samples_array, axis=0) # Median is more robust for non-normal
        # Calculate Coverage & R2
        inside_ci = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage_pct = np.mean(inside_ci) * 100
        r2 = r2_score(y_true, y_mean)
        
        # Scatter with error bars
        ax.errorbar(y_true, y_mean, yerr=ci_bound, fmt='o', ecolor='lightgray', 
                    alpha=0.4, label='Pred Mean with 95% CI', markersize=3)
        
        # Identity line (45 degree)
        limits = [
            min(y_true.min(), y_mean.min()),
            max(y_true.max(), y_mean.max())
        ]
        ax.plot(limits, limits, 'r--', linewidth=1.5, label='Perfect Match')
        
        # Formatting each subplot
        ax.set_title(f'{labels[i]}\nCoverage: {coverage_pct:.2f}% | $R^2$: {r2:.4f}')
        ax.set_xlabel(f'True $u_{labels[i][0].lower()}$')
        ax.set_ylabel(f'Predicted $\mu_{labels[i][0].lower()}$')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(prop={'size': 8})

    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        save_file = os.path.join(save_path, f"disp_r2_coverage_xy_{suffix}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_file}")
    
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--model_path', type=str, default="20260411T115941_isihara_0.0_0.01_8_0.975_5_40.0_1_0")
    parser.add_argument('--validation_load_step_indices', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--n_sample', type=int, default=128)


    return parser.parse_args()

if __name__ == "__main__" :
    args = parse_args()
    # read file from coverage/{mat_model}_{disp_noise}_{load_noise}
    # material_model_name = "isihara"
    validation_load_step_indices = args.validation_load_step_indices
    n_sample = args.n_sample
    model_path = args.model_path

    # load result 

    true_data_dir = Path("precomputed_vfm")

    analysis_dir = Path("coverage_test") 
    extraction_result_dir = Path("selected_model") 
    case_name = args.model_path
    precomputed_vfm_name = f"{case_name.split("_")[1]}_{case_name.split("_")[2]}_{case_name.split('_')[3]}_{case_name.split('_')[4]}_{case_name.split('_')[5]}"

    save_path = analysis_dir / case_name
    save_path.mkdir(parents=True, exist_ok=True)

    step = validation_load_step_indices[-1]
    pred_dir_name = analysis_dir / case_name
    # gt_samples_dir_name = analysis_dir / f"{material_model_name}_gt_{load_noise}"
    files = os.listdir(pred_dir_name / "piola_samples")
    pt_files = os.listdir(pred_dir_name / "piola_traction_samples")
    true_data = np.load(true_data_dir / f"{precomputed_vfm_name}.npz")
    u_true = true_data["u"][step]

    u_pred_piola_samples = [] 
    for f in files :
        data = np.load(pred_dir_name/ "piola_samples" / f)
        u_pred_piola_samples.append(data["u_pred"][step])
    u_pred_piola_samples = jnp.array(u_pred_piola_samples)

    u_pred_piola_traction_samples = []
    for f in pt_files :
        data = np.load(pred_dir_name/ "piola_traction_samples" / f)
        u_pred_piola_traction_samples.append(data["u_pred"][step])
    u_pred_piola_traction_samples = jnp.array(u_pred_piola_traction_samples)

    # err = u_samples - u_true[None, :, :]
    import numpy as np

    # Define target locations on the r = 0.1 circle
    targets = np.array([
        # [0.0866, 0.05],
        [0.0707, 0.0707],
        # [0.05, 0.0866],
        [0.25, 0.75],
        [0.5, 0.5],
        [0.75, 0.25],
        [1, 1]


    ])
    # u_samples = u_true_samples
    # Find the indices of the closest nodes in your mesh
    node_indices = []
    for target in targets:
        dist = np.linalg.norm(data["node_coords"] - target, axis=1)
        node_indices.append(np.argmin(dist).item())

    print(f"Closest node indices: {node_indices}")
    # plot_disp_field(data["node_coords"], data["cells"], u_true, u_pred_piola_samples.mean(axis=0), u_pred_piola_samples.std(axis=0), node_indices, save_path)
    # plot_disp_field(data["node_coords"], data["cells"], u_true, u_pred_piola_samples.mean(axis=0), u_pred_piola_samples.std(axis=0), node_indices, save_path)
    # plot disp field
    # node_index = 100
    # u_pred_samples = u_pred_piola_samples
    # u_pt_lower_bound = np.quantile(u_pred_piola_samples, 0.025, axis=0)
    # u_pt_upper_bound = np.quantile(u_pred_piola_samples, 0.975, axis=0)
    node_type = true_data["node_type"]
    plot_disp_field(data["node_coords"], data["cells"], u_true, u_pred_piola_samples.mean(axis=0), u_pred_piola_samples.std(axis=0), node_indices, save_path)

    plot_node_distributions(u_true, u_pred_piola_samples, u_pred_piola_traction_samples, node_indices, save_path)
    # plot_disp_r2_coverage(u_true, u_pred_samples.mean(axis=0), u_pred_samples.std(axis=0), save_path)

    true_data = np.load(true_data_dir / f"{precomputed_vfm_name}.npz")
    u_true_val = true_data["u"][validation_load_step_indices]

    u_pred_piola_samples_val = [] 
    for f in files :
        data = np.load(pred_dir_name/ "piola_samples" / f)
        u_pred_piola_samples_val.append(data["u_pred"][validation_load_step_indices])
    u_pred_piola_samples_val = jnp.array(u_pred_piola_samples_val)
    p_val_shape = u_pred_piola_samples_val.shape
    u_pred_piola_samples_val_flat = u_pred_piola_samples_val.reshape(p_val_shape[0], -1, 2)
    u_pred_piola_traction_samples_val = []
    for f in pt_files :
        data = np.load(pred_dir_name/ "piola_traction_samples" / f)
        u_pred_piola_traction_samples_val.append(data["u_pred"][validation_load_step_indices])
    u_pred_piola_traction_samples_val = jnp.array(u_pred_piola_traction_samples_val)
    pt_val_shape = u_pred_piola_traction_samples_val.shape

    u_pred_piola_traction_samples_val_flat = u_pred_piola_traction_samples_val.reshape(pt_val_shape[0], -1, 2)
    u_true_val_flat = u_true_val.reshape(-1, 2)
    

    u_pt_lower_bound = np.quantile(u_pred_piola_traction_samples_val_flat, 0.025, axis=0)
    u_pt_upper_bound = np.quantile(u_pred_piola_traction_samples_val_flat, 0.975, axis=0)
    u_pt_median = np.quantile(u_pred_piola_traction_samples_val_flat, 0.5, axis=0)
    plot_disp_r2_coverage(u_true_val_flat, u_pt_median, u_pt_lower_bound, u_pt_upper_bound, save_path, suffix ="_piola_traction")


    u_p_lower_bound = np.quantile(u_pred_piola_samples_val_flat, 0.025, axis=0)
    u_p_upper_bound = np.quantile(u_pred_piola_samples_val_flat, 0.975, axis=0)
    u_p_median = np.quantile(u_pred_piola_samples_val_flat, 0.5, axis=0)
    plot_disp_r2_coverage(u_true_val_flat, u_p_median, u_p_lower_bound, u_p_upper_bound, save_path, suffix ="_piola")

    # for n_idx in node_indices :
    #     plot_comprehensive_analysis(u_true, u_pred_samples, node_type, n_idx, save_path)
    pass