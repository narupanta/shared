
# Import some useful modules.
import jax
import jax.numpy as jnp
import os

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
import jax.random as jr 
jax.config.update("jax_enable_x64", True)

from core.utils import *
from core.model import SparseHyperelasticityGP, GPParams, GPRawParams
from core.loss_function import force_residual_force_controlled
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

def plot_fem_verification(I1_bar_true, I2_bar_true, J_true,
                          I1_bar_pred, I2_bar_pred, J_pred,
                          I1_bar_train, I2_bar_train, J_train,
                          inducing_points):
    """
    Plots the FEM verification figures.
    inducing_points: assumed to be (M, 3) representing [I1_bar, I2_bar, J]
    """

    # --- FIGURE 1: Accuracy (Pred vs True) ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    # I1_bar
    axes1[0].scatter(I1_bar_true, I1_bar_pred, alpha=0.5, edgecolors='b', facecolors='none')
    lims = [np.min([axes1[0].get_xlim(), axes1[0].get_ylim()]),
            np.max([axes1[0].get_xlim(), axes1[0].get_ylim()])]
    axes1[0].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axes1[0].set_title(r'$\bar{I}_1$ Accuracy')
    axes1[0].set_xlabel('True')
    axes1[0].set_ylabel('Pred')

    # I2_bar
    axes1[1].scatter(I2_bar_true, I2_bar_pred, alpha=0.5, edgecolors='b', facecolors='none')
    lims = [np.min([axes1[1].get_xlim(), axes1[1].get_ylim()]),
            np.max([axes1[1].get_xlim(), axes1[1].get_ylim()])]
    axes1[1].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axes1[1].set_title(r'$\bar{I}_2$ Accuracy')
    axes1[1].set_xlabel('True')

    # J
    axes1[2].scatter(J_true, J_pred, alpha=0.5, edgecolors='b', facecolors='none')
    lims = [np.min([axes1[2].get_xlim(), axes1[2].get_ylim()]),
            np.max([axes1[2].get_xlim(), axes1[2].get_ylim()])]
    axes1[2].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axes1[2].set_title(r'$J$ Accuracy')
    axes1[2].set_xlabel('True')

    plt.tight_layout()
    plt.savefig('fem_accuracy_parity.png')

    # --- FIGURE 2: Invariant Space Coverage ---
    # Inducing points indices: 0:I1_bar, 1:I2_bar, 2:J
    z_i1 = inducing_points[:, 0]
    z_i2 = inducing_points[:, 1]
    z_j  = inducing_points[:, 2]

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # I1_bar vs I2_bar
    axes2[0].scatter(I1_bar_pred - 3, I2_bar_pred- 3, edgecolors='red', facecolors='none', alpha=0.3, label='Pred (Sim)')
    axes2[0].scatter(I1_bar_train- 3, I2_bar_train- 3, edgecolors='black', facecolors='none', alpha=0.5, label='Train')
    axes2[0].scatter(z_i1- 3, z_i2 - 3, color='blue', marker='x', s=50, label='Inducing')
    axes2[0].set_xlabel(r'$\bar{I}_1 - 3$')
    axes2[0].set_ylabel(r'$\bar{I}_2 - 3$')
    axes2[0].set_title(r'$\bar{I}_1 - 3$ vs $\bar{I}_2 - 3$')
    axes2[0].legend()

    # I1_bar vs J
    axes2[1].scatter(I1_bar_pred- 3, (J_pred-1)**2, edgecolors='red', facecolors='none', alpha=0.3)
    axes2[1].scatter(I1_bar_train- 3, (J_train-1)**2, edgecolors='black', facecolors='none', alpha=0.5)
    axes2[1].scatter(z_i1- 3, (z_j-1)**2, color='blue', marker='x', s=50)
    axes2[1].set_xlabel(r'$\bar{I}_1 - 3$')
    axes2[1].set_ylabel(r'$(J - 1)^2$')
    axes2[1].set_title(r'$\bar{I}_1 - 3$ vs $J$')

    # I2_bar vs J
    axes2[2].scatter(I2_bar_pred- 3, (J_pred-1)**2, edgecolors='red', facecolors='none', alpha=0.3)
    axes2[2].scatter(I2_bar_train- 3, (J_train-1)**2, edgecolors='black', facecolors='none', alpha=0.5)
    axes2[2].scatter(z_i2- 3, (z_j-1)**2, color='blue', marker='x', s=50)
    axes2[2].set_xlabel(r'$\bar{I}_2 - 3$')
    axes2[2].set_ylabel(r'$(J - 1)^2$')
    axes2[2].set_title(r'$\bar{I}_2 - 3$ vs $(J - 1)^2$')

    plt.tight_layout()
    plt.savefig('invariant_space_coverage.png')


def plot_force_fields(node_coords, cells, f, filename="force_fields.png"):
    """
    node_coords: (N, 2) array of nodal coordinates
    cells: (C, 3) array of node indices per triangle
    f: (N, 2) internal force or residual vector [fx, fy]
    """
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    fx = f[:, 0]
    fy = f[:, 1]

    # Create the triangulation object
    triangulation = tri.Triangulation(x, y, cells)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Plot f_x Field
    ax0 = axes[0]
    # 'tricontourf' creates the filled color contours
    tpc0 = ax0.tricontourf(triangulation, fx, cmap='RdBu_r', levels=50)
    fig.colorbar(tpc0, ax=ax0, label='$f_x$ (Force/Residual)')
    ax0.set_title('Spatial Field: $f_x$')
    ax0.set_aspect('equal')
    ax0.set_xlabel('$x$')
    ax0.set_ylabel('$y$')

    # 2. Plot f_y Field
    ax1 = axes[1]
    tpc1 = ax1.tricontourf(triangulation, fy, cmap='RdBu_r', levels=50)
    fig.colorbar(tpc1, ax=ax1, label='$f_y$ (Force/Residual)')
    ax1.set_title('Spatial Field: $f_y$')
    ax1.set_aspect('equal')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Figure saved as {filename}")

# Usage:
# plot_force_fields(node_coords, cells, R_nodes)

if __name__ == "__main__" :
    
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
    mesh_data = jnp.load("/home/mmdiscovery/shared/mesh/square_with_holes.npz")
    node_coords = mesh_data["node_coords"]
    cells = mesh_data["cells"]
    ele_type = 'TRI3'
    cell_type = get_meshio_cell_type(ele_type)
    data_dir = os.path.join('data')

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
    top_dbc = lambda point : 0.1
    dirichlet_bc_info = [
        [bottom, bottom] , 
        [0, 1],
        [zero_dbc, zero_dbc]]
    
    node_type = np.zeros(node_coords.shape[0], dtype=int)
    check = jnp.sum(jax.vmap(left)(node_coords))
    node_type[jax.vmap(left)(node_coords)] = 1
    node_type[jax.vmap(bottom)(node_coords)] = 2
    node_type[jax.vmap(right)(node_coords)] = 3
    node_type[jax.vmap(top)(node_coords)] = 4

    true_material_model = get_material("isihara")
    true_piola_stress_func = lambda f : true_material_model.P(fto3x3(f))[:2, :2]
    model_path = "/home/mmdiscovery/shared/selected_model/Isihara/" # Replace with the actual path to your saved model

    with open(os.path.join(model_path, "best_params.npy"), "rb") as f:
        best_params = jnp.load(f, allow_pickle=True).item()

    with open(os.path.join(model_path, "I_z.npy"), "rb") as f:
        I_z = jnp.load(f)
    with open(os.path.join(model_path, "I_obs_all.npy"), "rb") as f:
        I_obs_all = jnp.load(f)
    best_raw_params = GPRawParams(**best_params)

    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = calculate_min_ls(dev_z)
    min_vol = calculate_min_ls(vol_z)

    learned_gp = SparseHyperelasticityGP(best_raw_params, I_z, min_dev, min_vol)

    pred_piola_stress_func = lambda f: learned_gp.piola(fto3x3(f), key = None)[:2, :2]

    # # Create an instance of the problem.
    problem_true = HyperElasticity(mesh = mesh,
                            vec=2,
                            dim=2,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns = [right,top],
                            material_model_piola_stress=true_piola_stress_func)

    problem_pred = HyperElasticity(mesh = mesh,
                            vec=2,
                            dim=2,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns = [right,top],
                            material_model_piola_stress=pred_piola_stress_func)
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt", 
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-5,  # Force higher accuracy in the linear solve
        "ksp_atol": 1e-8,
    }
    asym_factor = 0.0
    loads_top = jnp.linspace(0.0, 12.0, 20)
    loads_right = loads_top * asym_factor
    loads = jnp.stack([loads_right, loads_top], axis=1)
    # Solve the defined problem.

    # u_true = jnp.zeros_like(node_coords)

    def solve_fem(problem, petsc_options, loads) :
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
        return u

    u_true = solve_fem(problem_true, petsc_options, loads)
    def piola(f, key) :
        piola_mean = learned_gp.piola_dist(f).mean
        piola_var = learned_gp.piola_dist(f).var
        return piola_mean + jax.random.normal(key, piola_mean.shape) * piola_var
    def psi(f,key) :

        lambda f: learned_gp.psi(fto3x3(f),key)[:2, :2]

    main_key = jr.PRNGKey(42)
    n_samples = 1
    u_samples = []
    R_nodes_samples = []
    true_R_nodes = force_residual_force_controlled(u_true, loads[-1], -loads[-1], jax.vmap(lambda f: true_material_model.phi(fto3x3(f))),
                                                                                            coords = node_coords,
                                                                                            cells = cells, 
                                                                                            node_type = node_type)
    for i in range(n_samples) :
        main_key, subkey = jr.split(main_key)
        # u_pred = jnp.zeros_like(node_coords)

        pred_piola_stress_func = lambda f: piola(fto3x3(f), key = subkey)[:2, :2]
        
        u_pred = solve_fem(problem_pred, petsc_options, loads)
        u_samples.append(u_pred)
        
        R_nodes = force_residual_force_controlled(u_pred, loads[-1], -loads[-1], jax.vmap(lambda f: learned_gp.psi(f, key = subkey)),
                                                                                            coords = node_coords,
                                                                                            cells = cells, 
                                                                                            node_type = node_type)
        R_nodes_samples.append(R_nodes)
    u_pred = jnp.array(u_samples)[-1, :, :]
    R_nodes_array = jnp.array(R_nodes_samples)[0]

    # --- Color Definitions & Custom Colormap ---
    COLOR_BLACK = '#000000'
    COLOR_GREY  = '#808080'
    COLOR_NAVY  = '#000080'
    COLOR_ROYAL = '#4169E1'

    # Creating a gradient: Black -> Grey -> Navy -> Royal
    custom_cmap = LinearSegmentedColormap.from_list("fem_style", 
        [COLOR_BLACK, COLOR_GREY, COLOR_NAVY, COLOR_ROYAL], N=256)

    # --- 1. Calculations ---
    u_true_sq_mag = u_true[:, 0]**2 + u_true[:, 1]**2
    u_pred_sq_mag = u_pred[:, 0]**2 + u_pred[:, 1]**2
    u_error_mag = np.linalg.norm(u_true - u_pred, axis=1)

    # Deformed nodal positions
    wp_true = node_coords[:, :2] + u_true
    wp_pred = node_coords[:, :2] + u_pred

    # --- SCALE SYNCHRONIZATION ---
    # Calculate GLOBAL limits across both True and Predicted deformed sets
    all_x = np.concatenate([wp_true[:, 0], wp_pred[:, 0]])
    all_y = np.concatenate([wp_true[:, 1], wp_pred[:, 1]])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    # Add a 5% margin padding to the axes
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    x_lims = (x_min - x_margin, x_max + x_margin)
    y_lims = (y_min - y_margin, y_max + y_margin)

    # Create Triangulations
    tri_true = tri.Triangulation(wp_true[:, 0], wp_true[:, 1], cells)
    tri_pred = tri.Triangulation(wp_pred[:, 0], wp_pred[:, 1], cells)

    # --- 2. Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), facecolor='white')
    fig.suptitle('FEM Deformation Analysis (Synchronized Axis Scales)', fontsize=18, fontweight='bold', y=1.05)

    # Plot 1: True Deformed
    ax1 = axes[0]
    tpc1 = ax1.tripcolor(tri_true, u_true_sq_mag, cmap="Blues", shading='gouraud')
    ax1.triplot(tri_true, color=COLOR_GREY, linewidth=0.5, alpha=0.3)
    fig.colorbar(tpc1, ax=ax1, label='$u_x^2 + u_y^2$')
    ax1.set_title('True Deformed Domain', fontsize=14)

    # Plot 2: Predicted Deformed
    ax2 = axes[1]
    tpc2 = ax2.tripcolor(tri_pred, u_pred_sq_mag, cmap="Blues", shading='gouraud')
    ax2.triplot(tri_pred, color=COLOR_GREY, linewidth=0.5, alpha=0.3)
    fig.colorbar(tpc2, ax=ax2, label='$u_x^2 + u_y^2$')
    ax2.set_title('Predicted Deformed Domain', fontsize=14)

    # Plot 3: Absolute Error on Predicted Mesh
    ax3 = axes[2]
    tpc3 = ax3.tripcolor(tri_pred, u_error_mag, cmap='inferno', shading='gouraud')
    ax3.triplot(tri_pred, color=COLOR_GREY, linewidth=0.5, alpha=0.4)
    fig.colorbar(tpc3, ax=ax3, label='Error Magnitude')
    ax3.set_title('Nodal Prediction Error', fontsize=14, fontweight='bold')

    # --- Formatting & Applying Uniform Scale ---
    for i, ax in enumerate(axes):
        ax.set_aspect('equal') # Maintain geometric proportions
        ax.set_xlim(x_lims)    # Synchronized X
        ax.set_ylim(y_lims)    # Synchronized Y
        ax.set_xlabel('X Position')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 0:
            ax.set_ylabel('Y Position')

    plt.tight_layout()
    plt.savefig(os.path.join("output", "test_fem_analysis.png"))

    f_true, _ = deformation_gradient_element(node_coords[cells], u_true[cells])
    f_pred, _ = deformation_gradient_element(node_coords[cells], u_pred[cells])
    invariants_true, _ = jax.vmap(invariants_and_derivatives)(f_true)
    invariants_pred, _ = jax.vmap(invariants_and_derivatives)(f_pred)
    dev_true, vol_true = jax.vmap(transform_input_features)(invariants_true)
    dev_pred, vol_pred = jax.vmap(transform_input_features)(invariants_pred)
    I1_true = dev_true[:, 0]
    I2_true = dev_true[:, 1]
    J_true = vol_true
    I1_pred = dev_pred[:, 0]
    I2_pred = dev_pred[:, 1]
    J_pred = vol_pred
    # I1_pred = dev_true[:, 0]
    # I2_pred = dev_true[:, 1]
    # J_pred = vol_true
    dev_train, vol_train = jax.vmap(transform_input_features)(I_obs_all)
    I1_train = dev_train[:, 0]
    I2_train = dev_train[:, 1]
    J_train = vol_train
    
    inducing = I_z
    plot_fem_verification(I1_true, I2_true, J_true,
                        I1_pred, I2_pred, J_pred,
                        I1_train, I2_train, J_train,
                        inducing)
    #plot I1_bar_pred vs I1_bar_true, I2_bar_pred vs I2_bar_true, J_pred vs J_true

    #plot I1_bar vs I2_bar (train, test), I1_bar vs J (train, test), I2_bar vs J (train, test)
