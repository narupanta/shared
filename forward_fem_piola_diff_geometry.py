
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
from pathlib import Path
from core.utils import *
from core.model import SparseHyperelasticityGP, GPParams, GPRawParams
# from core.loss_function import force_residual_force_controlled
from core.material_models import get_material
from core.datasetclass import BenchmarkDataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def plot_fem_verification(I1_bar_true, I2_bar_true, J_true,
                          I1_bar_mean, I1_bar_upper, I1_bar_lower, 
                          I2_bar_mean, I2_bar_upper, I2_bar_lower,
                          J_mean, J_upper, J_lower):
    
    true_vals = [I1_bar_true, I2_bar_true, J_true]
    means = [I1_bar_mean, I2_bar_mean, J_mean]
    uppers = [I1_bar_upper, I2_bar_upper, J_upper]
    lowers = [I1_bar_lower, I2_bar_lower, J_lower]
    titles = [r'$\bar{I}_1$', r'$\bar{I}_2$', r'$J$']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i in range(3):
        t, m, u, l = true_vals[i], means[i], uppers[i], lowers[i]
        
        # Calculate Metrics
        r2 = r2_score(t, m)
        # Coverage: percentage of true values within [lower, upper]
        coverage = np.mean((t >= l) & (t <= u)) * 100
        
        # Plotting
        # Vertical error bars represent the 95% CI
        axes[i].errorbar(t, m, yerr=[m - l, u - m], fmt='o', 
                         alpha=0.4, ecolor='gray', mfc='none', mec='b', capsize=2)
        
        # Parity Line
        lims = [np.min([t, m]), np.max([t, m])]
        axes[i].plot(lims, lims, 'k--', alpha=0.75, label='Parity')
        
        # Text annotations for R2 and Coverage
        stats_text = f'$R^2: {r2:.4f}$\nCov: {coverage:.1f}%'
        axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        axes[i].set_title(f'{titles[i]} Accuracy & CI')
        axes[i].set_xlabel('True')
        if i == 0: axes[i].set_ylabel('Predicted Mean')

    plt.tight_layout()
    plt.savefig('fem_deployment/fem_accuracy_parity_with_ci.pdf')

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

def plot_disp_field(node_coords, cells, u_true, u_pred_mean, u_pred_std, save_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "text.usetex": False
    })

    # --- Data Preparation ---
    coords_true = node_coords + u_true
    coords_pred = node_coords + u_pred_mean
    
    def get_mag(u): return np.linalg.norm(u, axis=1)
    mag_true = get_mag(u_true)
    mag_pred = get_mag(u_pred_mean)
    error = np.linalg.norm(u_true - u_pred_mean, axis=1)/(mag_true+1e-6) * 100
    mag_std = get_mag(u_pred_std) if u_pred_std.ndim > 1 else u_pred_std

    fig, axes = plt.subplots(2, 2, figsize=(12*1.5, 14*1.5)) # Slightly wider to accommodate colorbars
    # plt.suptitle('Deformed Field: Accuracy & Uncertainty', fontsize=20)


    # --- Helper Function for Colorbars ---
    def add_colorbar(im, ax, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        # cbar = fig.colorbar(im, cax=cax, label=label)
        cbar = fig.colorbar(im, cax=cax)
        
        # 1. Set the main colorbar title (label) and its size
        cbar.set_label(label, size=36, weight='bold') # Adjust '16' as needed
        
        # 2. Set the size of the tick values (the numbers)
        cbar.ax.tick_params(labelsize=36) # Adjust '14' as needed
        cbar.locator = ticker.MaxNLocator(nbins=4)
        cbar.update_ticks()

    # 1,1: True Material
    tri_true = tri.Triangulation(coords_true[:, 0], coords_true[:, 1], cells)
    im1 = axes[0, 0].tripcolor(tri_true, mag_true, cmap='Blues')
    # axes[0, 0].set_title('True Material Model $\|\mathbf{u_{true}}\|$')
    add_colorbar(im1, axes[0, 0], "$\|\mathbf{u_{true}}\|$")

    # 1,2: Predicted Material
    tri_pred = tri.Triangulation(coords_pred[:, 0], coords_pred[:, 1], cells)
    im2 = axes[0, 1].tripcolor(tri_pred, mag_pred, cmap='Blues')
    # axes[0, 1].set_title('Predicted Material Model $\|\mathbf{u_{pred}}\|$')
    add_colorbar(im2, axes[0, 1], "$\|\mathbf{u_{pred}}\|$")

    # 2,1: Nodal error
    im3 = axes[1, 0].tripcolor(tri_pred, error, cmap='inferno')
    # axes[1, 0].set_title(r'$||\mthbf{u_{true}} - \mathbf{u_{pred}}||$')
    add_colorbar(im3, axes[1, 0], r"$\% Error$")

    # 2,2: Uncertainty
    im4 = axes[1, 1].tripcolor(tri_pred, mag_std, cmap='magma')
    # axes[1, 1].set_title(r'Uncertainty ($\sigma_u$)')
    add_colorbar(im4, axes[1, 1], "$\sigma_{\|\mathbf{u_{pred}}\|}$")

    # Standardize labels
    for ax in axes.flat:
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # for ax in axes.flat:
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Remove tick marks and tick labels (values)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Ensure the grid is off so it doesn't show through the transparency
        # ax.grid(False)
        # Optional: if you want to remove the frame/box as well, uncomment the line below
        ax.axis('off') 
        
        # ax.set_aspect('equal')
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # plt.tight_layout(rect=[0, 0, 1, 1])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "fem_diff_geometry.pdf"), bbox_inches='tight')
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
    mesh_data = jnp.load("mesh/square_with_holes.npz")
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
    extraction_result_dir = Path("saved_model")
    case_name = "20260502T215835_isihara_0.0001_0.01_8.0_0.9_5_80.0_1_0"
    mat_model_name = case_name.split("_")[1]
    true_material_model = get_material(mat_model_name)
    true_piola_stress_func = lambda f : true_material_model.P(fto3x3(f))[:2, :2]
    # model_path = f"/home/mmdiscovery/shared/saved_model/{case_name}" # Replace with the actual path to your saved model
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
        "snes_monitor": None,
        "snes_atol": 1e-6,
        "snes_rtol": 1e-6,
        "snes_stol": 1e-6,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    asym_factor = 0.0
    loads_top = jnp.linspace(0.0, 6.0, 10)
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

    main_key = jr.PRNGKey(42)
    n_samples = 256
    u_samples = []
    for i in range(n_samples) :
        # main_key, subkey = jr.split(main_key)
        # u_pred = jnp.zeros_like(node_coords)
        success = False
        tries = 0
        max_tries = 5
        
        while not success and tries < max_tries:
            main_key, subkey = jr.split(main_key)
            
            # 1. Setup the problem with the new subkey
            problem_pred = HyperElasticity(
                mesh=mesh,
                vec=2,
                dim=2,
                ele_type=ele_type,
                dirichlet_bc_info=dirichlet_bc_info,
                location_fns=[right, top],
                material_model_piola_stress=lambda f: model.piola(fto3x3(f), subkey)[:2, :2]
            )
            
            try:
                # 2. Attempt the solve
                print(f"Sample {i}: Attempt {tries + 1}/{max_tries}...")
                u_pred = solve_fem(problem_pred, petsc_options, loads)
                
                # If we reach here, solve_fem succeeded
                success = True 
                
            except Exception as e:
                # 3. Handle failure
                tries += 1
                print(f"Simulation failed on sample {i}, try {tries}: {e}")
                if tries >= max_tries:
                    print(f"Max tries reached for sample {i}. Skipping or raising error.")
                    # Option A: raise e (stops everything)
                    # Option B: break (moves to next 'i' in n_sample)
                    raise e 

        # 4. Save and append only if successful
        if success:
            u_samples.append(u_pred)
            sample_dir = os.path.join("fem_deployment", "samples")
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
                
            save_file = os.path.join(sample_dir, f"u_pred_{i}.npz")
            np.savez_compressed(save_file, u_pred=u_pred, cells=cells, 
                                node_coords=node_coords, node_type=node_type)
    u_pred_samples = jnp.array(u_samples)

    u_pred_mean = jnp.mean(u_pred_samples, axis=0)
    u_pred_std = jnp.std(u_pred_samples, axis=0)

    # plot_disp_field(node_coords, cells, u_true, u_pred_mean, u_pred_std, "fem_deployment")

    f_true, _ = deformation_gradient_element(node_coords[cells], u_true[cells])
    vmapped_def_grad = jax.vmap(deformation_gradient_element, in_axes=(None, 0))
    f_pred_samples, _ = vmapped_def_grad(node_coords[cells], u_pred_samples[:, cells])
    invariants_true, _ = jax.vmap(invariants_and_derivatives)(jax.vmap(fto3x3)(f_true))
    vmapped_inv_samples = jax.vmap(jax.vmap(invariants_and_derivatives))
    invariants_pred_samples, _ = vmapped_inv_samples(jax.vmap(jax.vmap(fto3x3))(f_pred_samples))
    dev_true, vol_true = jax.vmap(transform_input_features)(invariants_true)
    vmapped_feat_samples = jax.vmap(jax.vmap(transform_input_features))
    dev_pred_samples, vol_pred_samples = vmapped_feat_samples(invariants_pred_samples)
    dev_pred_mean = jnp.mean(dev_pred_samples, axis=0)
    vol_pred_mean = jnp.mean(vol_pred_samples, axis=0)
    I1_true = dev_true[:, 0]
    I2_true = dev_true[:, 1]
    J_true = vol_true[:, 0]
    I1_pred_mean = dev_pred_mean[:, 0]
    I2_pred_mean = dev_pred_mean[:, 1]
    J_pred_mean = vol_pred_mean[:, 0]
    I1_pred_lower, I1_pred_upper = np.quantile(dev_pred_samples[:, :, 0], [0.025, 0.975], axis = 0)
    I2_pred_lower, I2_pred_upper = np.quantile(dev_pred_samples[:, :, 1], [0.025, 0.975], axis = 0)
    J_pred_lower, J_pred_upper = np.quantile(vol_pred_samples[:, :, 0], [0.025, 0.975], axis = 0)

    # I1_pred = dev_true[:, 0]
    # I2_pred = dev_true[:, 1]
    # J_pred = vol_true
    dev_train, vol_train = jax.vmap(transform_input_features)(I_obs_all.reshape(-1, 3))
    I1_train = dev_train[:, 0]
    I2_train = dev_train[:, 1]
    J_train = vol_train
    
    inducing = I_z
    plot_fem_verification(I1_true, I2_true, J_true,
                        I1_pred_mean, I1_pred_upper, I1_pred_lower,
                        I2_pred_mean, I2_pred_upper, I2_pred_lower,
                        J_pred_mean, J_pred_upper, J_pred_lower)
    #plot I1_bar_pred vs I1_bar_true, I2_bar_pred vs I2_bar_true, J_pred vs J_true

    #plot I1_bar vs I2_bar (train, test), I1_bar vs J (train, test), I2_bar vs J (train, test)
