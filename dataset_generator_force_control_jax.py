
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
import argparse
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Falls back to DejaVu if Times isn't found
    "font.size": 16,                # Base font size
    "axes.titlesize": 18,           # Subplot titles
    "axes.labelsize": 14,           # X and Y labels
    "legend.fontsize": 12,          # Legend text
    "xtick.labelsize": 12,          # Axis tick numbers
    "ytick.labelsize": 12,
    "figure.dpi": 600,              # High resolution for the screen and save
    "savefig.dpi": 600,             # Ensures saved file is high quality
    "text.usetex": False            # Set to True only if you have a full LaTeX install
})


def plot_fem_verification(I1_bar_true, I2_bar_true, J_true,
                          I1_bar_pred, I2_bar_pred, J_pred,
                          I1_bar_train, I2_bar_train, J_train,
                          inducing_points, save_path):

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    # Data groups for iteration
    data = [
        (I1_bar_true, I1_bar_pred, r'$\bar{I}_1$', axes1[0]),
        (I2_bar_true, I2_bar_pred, r'$\bar{I}_2$', axes1[1]),
        (J_true, J_pred, r'$J$', axes1[2])
    ]

    for true, pred, name, ax in data:
        # Calculate R2
        r2 = r2_score(true.flatten(), pred.flatten())
        
        # Plot Scatter (Hollow circles)
        ax.scatter(true, pred, alpha=0.5, edgecolors='b', facecolors='none', s=20)
        
        # Identity Line (Perfect parity)
        all_vals = np.concatenate([true.flatten(), pred.flatten()])
        low, high = np.min(all_vals), np.max(all_vals)
        ax.plot([low, high], [low, high], 'k--', alpha=0.75, zorder=0)
        
        # Add R2 Text Box
        ax.text(0.05, 0.92, f'$R^2 = {r2:.4f}$', transform=ax.transAxes, 
                fontsize=14, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_title(f'{name} Accuracy', fontsize = 16)
        ax.set_xlabel('True')
        if ax == axes1[0]:
            ax.set_ylabel('Pred')
        
        ax.grid(False) # Consistent with your previous request
        ax.set_xlim([low, high])
        ax.set_ylim([low, high])
    fig1.suptitle("Predicted Material Accuracy on FEM", fontsize=20)
    fig1.tight_layout()
    
    # Save files
    os.makedirs(save_path, exist_ok=True)
    fig1.savefig(os.path.join(save_path, "fem_deployment_accuracy_r2.png"), dpi=600)
    # --- FIGURE 2: Invariant Space Coverage ---
    # Inducing points indices: 0:I1_bar, 1:I2_bar, 2:J
    # z_i1 = inducing_points[:, 0]
    # z_i2 = inducing_points[:, 1]
    # z_j  = inducing_points[:, 2]

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # I1_bar vs I2_bar
    axes2[0].scatter(I1_bar_pred - 3, I2_bar_pred- 3, edgecolors='red', facecolors='none', alpha=0.3, label='Pred')
    axes2[0].scatter(I1_bar_train- 3, I2_bar_train- 3, edgecolors='black', facecolors='none', alpha=0.5, label='Train')
    # axes2[0].scatter(z_i1- 3, z_i2 - 3, color='blue', marker='x', s=50, label='Inducing')
    axes2[0].set_xlabel(r'$\bar{I}_1 - 3$')
    axes2[0].set_ylabel(r'$\bar{I}_2 - 3$')
    axes2[0].set_title(r'$\bar{I}_1 - 3$ vs $\bar{I}_2 - 3$')
    axes2[0].legend()

    # I1_bar vs J
    axes2[1].scatter(I1_bar_pred- 3, (J_pred-1)**2, edgecolors='red', facecolors='none', alpha=0.3)
    axes2[1].scatter(I1_bar_train- 3, (J_train-1)**2, edgecolors='black', facecolors='none', alpha=0.5)
    # axes2[1].scatter(z_i1- 3, (z_j-1)**2, color='blue', marker='x', s=50)
    axes2[1].set_xlabel(r'$\bar{I}_1 - 3$')
    axes2[1].set_ylabel(r'$(J - 1)^2$')
    axes2[1].set_title(r'$\bar{I}_1 - 3$ vs $(J - 1)^2$')

    # I2_bar vs J
    axes2[2].scatter(I2_bar_pred- 3, (J_pred-1)**2, edgecolors='red', facecolors='none', alpha=0.3)
    axes2[2].scatter(I2_bar_train- 3, (J_train-1)**2, edgecolors='black', facecolors='none', alpha=0.5)
    # axes2[2].scatter(z_i2- 3, (z_j-1)**2, color='blue', marker='x', s=50)
    axes2[2].set_xlabel(r'$\bar{I}_2 - 3$')
    axes2[2].set_ylabel(r'$(J - 1)^2$')
    axes2[2].set_title(r'$\bar{I}_2 - 3$ vs $(J - 1)^2$')
    fig2.suptitle("Training and testing invariant space", fontsize=20)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_path, "invariant_space_coverage.png"), dpi=600)



import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os

def plot_disp_field(node_coords, cells, u_true, u_pred_mean, u_pred_std, save_path):
    # --- Color Definitions ---
    triangulation = tri.Triangulation(node_coords[:, 0], node_coords[:, 1], cells)
    
    # Calculate magnitudes and coordinates
    # Note: Using your transformed parameters for displacement application
    def get_mag(u): return np.linalg.norm(u, axis=1)
    
    coords_true = node_coords + u_true
    coords_pred = node_coords + u_pred_mean
    
    mag_true = get_mag(u_true)
    mag_pred = get_mag(u_pred_mean)
    error = np.linalg.norm(u_true - u_pred_mean, axis=1)
    mag_std = get_mag(u_pred_std) if u_pred_std.ndim > 1 else u_pred_std

    fig, axes = plt.subplots(2, 2, figsize=(8, 12))
    plt.suptitle('Deformed Field: Accuracy & Uncertainty', fontsize=20)

    # 1,1: Deformed domain wrt u_true
    tri_true = tri.Triangulation(coords_true[:, 0], coords_true[:, 1], cells)
    im1 = axes[0, 0].tripcolor(tri_true, mag_true, cmap='Blues')
    axes[0, 0].set_title('True Material')
    fig.colorbar(im1, ax=axes[0, 0])

    # 1,2: Deformed domain wrt u_pred
    tri_pred = tri.Triangulation(coords_pred[:, 0], coords_pred[:, 1], cells)
    im2 = axes[0, 1].tripcolor(tri_pred, mag_pred, cmap='Blues')
    axes[0, 1].set_title('Predicted Material')
    fig.colorbar(im2, ax=axes[0, 1])

    # 2,1: Nodal error on pred deformed domain
    im3 = axes[1, 0].tripcolor(tri_pred, error, cmap='inferno')
    axes[1, 0].set_title('$(||u_{true} - u_{pred}||)$')
    fig.colorbar(im3, ax=axes[1, 0])

    # 2,2: u_std on pred deformed domain
    im4 = axes[1, 1].tripcolor(tri_pred, mag_std, cmap='magma')
    axes[1, 1].set_title('Uncertainty $(\sigma_u)$')
    fig.colorbar(im4, ax=axes[1, 1])

    # Standardize labels
    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.label_outer()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "displacement_analysis.png"), dpi=300, bbox_inches='tight')

def plot_dataset_viz(data, material_model_name, disp_noise_level, load_noise_level, save_path) :
    # --- 1. Setup Dummy Data (Simulating FEM Output) ---
    # Create a simple 2x2 rectangular mesh with 4 nodes and 2 triangular elements
    # Node coordinates (Undeformed mesh_pos)
    mesh_pos = data["mesh_pos"]

    # Element connectivity (cells: indices of nodes forming each triangle)
    cells = data["cells"]

    # Displacement components (ux and uy) at each node
    # This simulates a simple shear/tensile deformation
    percent_noise = 0.000
    ux = np.array(data["u"][-1, :, 0])
    ux[(data["node_type"][:, 1] != 1)] += np.random.normal(0, percent_noise, ux.shape)[(data["node_type"][:, 1] != 1)]
    uy = np.array(data["u"][-1, :, 1])
    uy[(data["node_type"][:, 2] != 1)] += np.random.normal(0, percent_noise, uy.shape)[(data["node_type"][:, 2] != 1)]

    # Combine components into the full displacement vector u
    u = np.column_stack((ux, uy))
    # u[data["node_type"] == 0] = u[data["node_type"] == 0] + np.random.normal(0, 0.0001, u.shape)[data["node_type"] == 0]
    # Calculate the deformed coordinates (world_pos)
    world_pos = mesh_pos[:, :2] + u

    # --- 2. Initialize Triangulation Objects ---
    # We need the x and y coordinates from the undeformed mesh
    x = mesh_pos[:, 0]
    y = mesh_pos[:, 1]

    # Create the Matplotlib Triangulation object
    # This object stores the connectivity (cells) and coordinates (x, y)
    triangulation = tri.Triangulation(x, y, cells)

    # --- 3. Plotting ---

    # Set up the figure with 1 row and 3 columns for the plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Finite Element Visualization (Undeformed vs. Deformed)', fontsize=16)

    # --- Subplot 1: Plotting UX (Horizontal Displacement) ---
    ax1 = axes[0]
    # tripcolor uses the triangulation to color the triangles based on the nodal value
    # `facecolors` uses the average of the three nodal values per triangle for coloring
    tpc1 = ax1.tripcolor(triangulation, ux, cmap='viridis', edgecolors='k', linewidth=0.5)
    fig.colorbar(tpc1, ax=ax1, label='$u_x$ Displacement')
    # ax1.scatter(mesh_pos[data["node_type"] == 5, 0], mesh_pos[node_type == 5, 1])
    ax1.set_title('Color Plot: $u_x$ (Horizontal Displacement)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_aspect('equal')

    # --- Subplot 2: Plotting UY (Vertical Displacement) ---
    ax2 = axes[1]
    tpc2 = ax2.tripcolor(triangulation, uy, cmap='magma', edgecolors='k', linewidth=0.01)
    fig.colorbar(tpc2, ax=ax2, label='$u_y$ Displacement')

    ax2.set_title('Color Plot: $u_y$ (Vertical Displacement)')
    ax2.set_xlabel('X Position')
    ax2.set_aspect('equal')

    # --- Subplot 3: Plotting Deformed Domain ---
    ax3 = axes[2]

    # Plot the outline of the UNDEFORMED mesh for reference (dashed gray)
    ax3.triplot(triangulation, 'r-', alpha=0.5, linewidth=0.5, label='Undeformed Mesh')

    # Plot the DEFORMED mesh. We must manually create a new triangulation
    # object using the deformed coordinates (world_pos) but the SAME connectivity (cells).
    x_def = world_pos[:, 0]
    y_def = world_pos[:, 1]
    tri_def = tri.Triangulation(x_def, y_def, cells)

    # Plot the deformed mesh (solid blue lines)
    ax3.triplot(tri_def, 'b-', linewidth=0.5, label='Deformed Mesh')

    ax3.set_title('Deformed Domain')
    ax3.set_xlabel('X Position')
    ax3.legend()
    ax3.set_aspect('equal')

    # Adjust layout to prevent overlaps
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # create save_path dir if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
   
    plt.savefig(save_path + f"/{material_model_name}_{disp_noise_level}_{load_noise_level}.png", dpi=300, bbox_inches='tight')

# Usage:
# plot_force_fields(node_coords, cells, R_nodes)
def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--model_path', type=str, default="20260411T115941_isihara_0.0_0.01_8_0.975_5_40.0_1_0")
    # parser.add_argument('--validation_load_step_indices', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--n_sample', type=int, default=128)


    return parser.parse_args()


if __name__ == "__main__" :
    args = parse_args()
    material_model_name = "isihara"
    disp_noise = 0.00
    load_noise = 0.00
    target_load = 6.0
    asym_factor = 1.0

    # validation_load_step_indices = args.validation_load_step_indices
    num_steps = 20
    # load result 


    save_path = Path("raw_dataset_jax")
    save_path.mkdir(parents=True, exist_ok=True)
    # get I_obs_all.npy

    true_material_model = get_material(material_model_name)
    true_piola_stress_func = lambda f : true_material_model.P(fto3x3(f))[:2, :2]

    # Define constitutive relationship.
    class HyperElasticity(Problem):
        def __init__(self, material_model_piola_stress, **kwargs) :
            super().__init__(**kwargs)
            self.material_model_piola_stress = material_model_piola_stress # should be function outputing piola stress [2x2 matrix]
        def custom_init(self):
            self.fe = self.fes[0]
        def get_surface_maps(self):
            def surface_map_top(u, x, load):
                return jnp.array([0., -load[0]])
            def surface_map_right(u,x,load) :
                return jnp.array([-load[0], 0.0])
            return [surface_map_right, surface_map_top]
        def set_params(self, params):
            surface_params = params
            self.internal_vars_surfaces = [[surface_params]]
        def get_tensor_map(self):

            def first_PK_stress(u_grad):
                I = jnp.eye(self.dim)
                F = u_grad + I
                P = self.material_model_piola_stress(F)
                return P

            return first_PK_stress
        
    # Specify mesh-related information (first-order hexahedron element).
    mesh_data = jnp.load("mesh/mesh.npz")
    node_coords = mesh_data["node_coords"][:, :2]
    cells = mesh_data["cells"]
    ele_type = 'TRI3'
    cell_type = get_meshio_cell_type(ele_type)
    mesh = Mesh(node_coords, cells)


    # Define boundary locations.
    def left(point):
        return jnp.isclose(point[0], 0., atol=1e-6)
    def bottom(point):
        return jnp.isclose(point[1], 0., atol=1e-6)
    def right(point):
        return jnp.isclose(point[0], 1., atol=1e-6)
    def top(point):
        return jnp.isclose(point[1], 1.0, atol=1e-6)

    zero_dbc = lambda point : 0
    dirichlet_bc_info = [
        [bottom, left] , 
        [1, 0],
        [zero_dbc, zero_dbc]]
    


    # 1. Compute boolean masks for each boundary
    is_left = jax.vmap(left)(node_coords)
    is_bottom = jax.vmap(bottom)(node_coords)
    is_right = jax.vmap(right)(node_coords)
    is_top = jax.vmap(top)(node_coords)

    # 2. Create integer indices (0=Internal, 1=Left, 2=Bottom, 3=Right, 4=Top)
    # We use jnp.where to replicate your original priority: Top > Right > Bottom > Left
    # node_indices = jnp.zeros(node_coords.shape[0], dtype=int)
    # node_indices = jnp.where(is_left, 1, node_indices)
    # node_indices = jnp.where(is_bottom, 2, node_indices)
    # node_indices = jnp.where(is_right, 3, node_indices)
    # node_indices = jnp.where(is_top, 4, node_indices)
    # Check if internal (not on any boundary)
    is_internal = ~(is_left | is_bottom | is_right | is_top)

    # Stack into a (N, 5) matrix
    node_type = jnp.stack([
        is_internal, is_left, is_bottom, is_right, is_top
    ], axis=-1).astype(jnp.float32)
    # # 3. Transform to one-hot encoding
    # # Resulting shape: (num_nodes, 5)
    # node_type_onehot = jax.nn.one_hot(node_indices, num_classes=5)

    # node_type = np.zeros(node_coords.shape[0], dtype=int)
    # node_type[jax.vmap(left)(node_coords)] = 1
    # node_type[jax.vmap(bottom)(node_coords)] = 2
    # node_type[jax.vmap(right)(node_coords)] = 3
    # node_type[jax.vmap(top)(node_coords)] = 4
    
    true_mat_model = get_material(material_model_name)
    psi_true_func = lambda f: true_mat_model.psi(f)
    piola_true_func = lambda f: true_mat_model.P(f)


    # # Create an instance of the problem.
    problem_true = HyperElasticity(mesh = mesh,
                            vec=2,
                            dim=2,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns = [right,top],
                            material_model_piola_stress=true_piola_stress_func)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt", 
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-5,  # Force higher accuracy in the linear solve
        "ksp_atol": 1e-8,
    }
    key = jax.random.PRNGKey(42)
    # asym_factor = 0.95
    # num_load_samples = 32
    # loads_top = jnp.linspace(0.0, 10, 10)
    # target_load = 10.0
    noise_std = load_noise * target_load
    target_load_noisy = target_load + noise_std * jax.random.normal(key)

    # noisy_target_loads = 
    noisy_load_top_base = jnp.linspace(0.0, target_load_noisy, num_steps).reshape(-1,1)
    noisy_load_right_base = noisy_load_top_base * asym_factor
    # Baseline loads shape: (10, 2)
    loads_noisy = jnp.concat([noisy_load_right_base, noisy_load_top_base], axis=1)

    def solve_fem(problem, petsc_options, loads) :
        u_list = []
        u = jnp.zeros_like(problem.mesh[0].points)
        for i, load in enumerate(loads):
            print("load step ", i, "= ", load)
            shape_right = (len(problem.boundary_inds_list[0]), problem.fes[0].num_face_quads, 1)
            shape_top = (len(problem.boundary_inds_list[1]), problem.fes[0].num_face_quads, 1)

            problem.internal_vars_surfaces = [
                [
                    jnp.full(fill_value=load[0], shape=shape_right),
                ],
                [
                    jnp.full(fill_value=load[1], shape=shape_top)
                ]
            ]
            u_= solver(problem, solver_options={'petsc_solver': petsc_options,
                                                    'initial_guess': u})
            u = u_[0]
            u_list.append(u)
        u_array = jnp.stack(u_list, axis=0)  
        return u_array

    u_true = solve_fem(problem_true, petsc_options, loads_noisy)


    data = dict(u = u_true, mesh_pos = node_coords, cells = cells, node_type = node_type)
    plot_dataset_viz(data, material_model_name, disp_noise, load_noise, "dataset_viz_jax")