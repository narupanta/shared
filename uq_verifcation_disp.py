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

def plot_node_distributions(u_true_samples, u_pred_samples, node_to_plot, save_path):
    """
    Plots local distributions for a list of nodes.
    Rows: Each node in node_to_plot
    Columns: X-direction, Y-direction
    """
    os.makedirs(save_path, exist_ok=True)
    
    num_nodes_to_plot = len(node_to_plot)
    fig, axes = plt.subplots(num_nodes_to_plot, 2, figsize=(14, 5 * num_nodes_to_plot))
    
    # Ensure axes is 2D even if only one node is plotted
    if num_nodes_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    directions = [
        {'idx': 0, 'label': 'X', 'color': 'dodgerblue'},
        {'idx': 1, 'label': 'Y', 'color': 'teal'}
    ]

    for i, node_idx in enumerate(node_to_plot):
        for d_info in directions:
            d = d_info['idx']
            ax = axes[i, d]
            
            # 1. Extract Data
            samples = u_pred_samples[:, node_idx, d]
            true_samples = u_true_samples[:, node_idx, d]
            pred_mean = np.mean(samples)
            pred_std = np.std(samples)
            true_mean = np.mean(true_samples)
            true_std = np.std(true_samples)

            
            # 2. Plot Histogram of Predicted Samples
            ax.hist(samples, bins=40, density=True, alpha=0.3, 
                    color=d_info['color'], label='Predicted Samples')
            ax.hist(true_samples, bins=40, density=True, alpha=0.3, 
                    color='red', label='True Samples')
            # 3. Add Vertical Lines
            ax.axvline(true_mean, color='red', linestyle='-', linewidth=2, label=f'Mean True: {true_mean:.4e}, Std True: {true_std:.4e}')
            ax.axvline(pred_mean, color=d_info['color'], linestyle='--', linewidth=2, label=f'Mean Pred: {pred_mean:.4e}, Std True: {pred_std:.4e}')
            
            # 4. Add Gaussian Distribution for True (std = 1e-4)
            # sigma_target = 1e-4
            # Create a range for the PDF centered at true_val
            # x_min = min(samples.min(), true_mean - 4*true_std)
            # x_max = max(samples.max(), true_mean + 4*true_std)
            # x_axis = np.linspace(x_min, x_max, 200)
            
            # ax.plot(x_axis, norm.pdf(x_axis, true_mean, true_std), 
            #         color='red', linestyle=':', alpha=0.8, label=f'Target $\sigma$ {true_std}')
            
            # Formatting
            if i == 0:
                ax.set_title(f'Direction {d_info["label"]}', fontsize=15)
            if d == 0:
                ax.set_ylabel(f'Node {node_idx}\nDensity', fontsize=12, fontweight='bold')
                
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(alpha=0.2)

    plt.suptitle(f'Local Displacement Distributions at Selected Nodes', fontsize=18, y=1.02)
    plt.tight_layout()
    
    save_file = os.path.join(save_path, "local_node_distributions.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
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


if __name__ == "__main__" :
    # read file from coverage/{mat_model}_{disp_noise}_{load_noise}
    material_model_name = "isihara"
    disp_noise = 0.000
    load_noise = 0.01
    train_load_step_indices = [0, 5, 9]
    validation_load_step_indices = [2, 4, 6, 8]

    # load result 

    true_data_dir = Path("precomputed_vfm")

    analysis_dir = Path("coverage_test") 
    extraction_result_dir = Path("selected_model") 
    case_name = f"{material_model_name}_{disp_noise}_{load_noise}"
    save_path = analysis_dir / case_name
    save_path.mkdir(parents=True, exist_ok=True)

    step = -1
    pred_dir_name = analysis_dir / f"{material_model_name}_{disp_noise}_{load_noise}" 
    gt_samples_dir_name = analysis_dir / f"{material_model_name}_gt_{load_noise}"
    files = os.listdir(pred_dir_name / "piola_load_samples")
    t_files = os.listdir(gt_samples_dir_name)
    true_data = np.load(true_data_dir / f"{case_name}.npz")
    u_true = true_data["u"][step]

    u_samples = [] 
    for f in files :
        data = np.load(pred_dir_name/ "piola_load_samples" / f)
        u_samples.append(data["u_pred"][step])
    u_samples = jnp.array(u_samples)

    u_true_samples = []
    for f in t_files :
        data = np.load(gt_samples_dir_name/ f)
        u_true_samples.append(data["u"][step])
    u_true_samples = jnp.array(u_true_samples)

    err = u_samples - u_true[None, :, :]
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
    plot_disp_field(data["node_coords"], data["cells"], u_true, u_samples.mean(axis=0), u_samples.std(axis=0), node_indices, save_path)
    # plot disp field
    # node_index = 100
    u_pred_samples = u_samples
    node_type = true_data["node_type"]
    plot_node_distributions(u_true_samples, u_pred_samples, node_indices, save_path)
    # for n_idx in node_indices :
    #     plot_comprehensive_analysis(u_true, u_pred_samples, node_type, n_idx, save_path)
    pass