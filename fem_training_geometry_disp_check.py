
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
from core.datasetclass import BenchmarkDataset, TractionDataset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

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
    
    # get training data
    material_model = "isihara"
    dataset_name = "isihara_fix"
    dataset = TractionDataset("dataset", dataset_name)
    F_all = []
    reactions = []
    u_all = []
    loads = []
    for loadstep in range(0, len(dataset), 2) :
        # if loadstep != len(dataset) - 1 :
        #     continue
        data = dataset[loadstep]
        coords = data["mesh_pos"][:,:2]
        cells = data["cells"]
        # u = data["u"]
        u_percent_noise = 0.00001
        node_type = data["node_type"]
        ux = data["u"][:, 0]
        # ux[(data["node_type"] != 1)] += np.random.normal(0, percent_noise * 1, ux.shape)[(data["node_type"] != 1)]
        uy = data["u"][:, 1]
        # uy[(data["node_type"] != 2)] += np.random.normal(0, percent_noise * 1, uy.shape)[(data["node_type"] != 2)]

        # Combine components into the full displacement vector u
        u = np.column_stack((ux, uy))
        # u[node_type == 0] = u[node_type == 0] + jax.random.normal(jr.key(0), u.shape)[node_type == 0] * 0.01 * mean_u
        # load_noise = 0.03
        load = data["load"] 
        # load += np.random.normal(0, load_noise * load, load.shape)
        reaction = data["reaction"]
        coord_cells = coords[cells]
        u_cells = u[cells]
        
        F, dNdx = deformation_gradient_element(coord_cells, u_cells)
        loads.append(load)
        u_all.append(u)
        F_all.append(F)
        reactions.append(reaction)
    
    model_path = "/home/mmdiscovery/shared/selected_model/20260209T135936/" # Replace with the actual path to your saved model

    with open(os.path.join(model_path, "best_params.npy"), "rb") as f:
        best_params = jnp.load(f, allow_pickle=True).item()

    with open(os.path.join(model_path, "I_z.npy"), "rb") as f:
        I_z = jnp.load(f)
    with open(os.path.join(model_path, "I_obs_all.npy"), "rb") as f:
        I_obs_all = jnp.load(f)
    u_array = jnp.array(u_all)
    loads = jnp.array(loads)
    load_noise = 0.03
    # check = jnp.mean(loads, axis = 0)
    load_array = loads + np.random.normal(0, load_noise * (jnp.max(loads) + jnp.min(loads)), loads.shape)
    reaction_array = jnp.array(reactions)

    best_raw_params = GPRawParams(**best_params)
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = calculate_min_ls(dev_z)
    min_vol = calculate_min_ls(vol_z)
    learned_gp = SparseHyperelasticityGP(best_raw_params, I_z, min_dev, min_vol) 
    main_key = jr.PRNGKey(42)
    n_samples = 50
    R_nodes_samples = []
    vmapped_force_residual = jax.vmap(force_residual_force_controlled, in_axes=(0, 0, 0, None, None, None, None))
    for i in range(n_samples) :
        main_key, subkey = jr.split(main_key)
        R_nodes = vmapped_force_residual(u_array, load_array, reaction_array, jax.vmap(lambda f: learned_gp.psi(f, subkey)), coords, cells, node_type)
        R_nodes_samples.append(R_nodes)
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    R_nodes_array = jnp.array(R_nodes_samples)
    free_node = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
    free_R_nodes = R_nodes_array[:, :, free_node, :]
    free_nodes_on_dc1 = R_nodes_array[:, :, node_type[:, 1] == 1, 1]
    free_nodes_on_dc2 = R_nodes_array[:, :, node_type[:, 2] == 1, 0]

    free_nodes_flat1 = free_R_nodes.flatten()
    free_nodes_on_dc1 = free_nodes_on_dc1.flatten()
    free_nodes_on_dc2 = free_nodes_on_dc2.flatten()
    free_nodes = jnp.concat([free_nodes_flat1, free_nodes_on_dc1, free_nodes_on_dc2], axis=0)
    # one_free_node = free_R_nodes[:, :, 100, :].flatten()
    # 1. Flatten your free residuals to a 1D array
    # residual_free: (#free_dofs, 1) -> (#free_dofs,)
    data_free = free_nodes
    fix_nodes_dc1 = (node_type[:, 1] == 1)
    fix_nodes_dc2 = (node_type[:, 2] == 1)
    fix_R_nodes_dc1 = R_nodes_array[:, :, fix_nodes_dc1, 0]
    fix_R_nodes_dc2 = R_nodes_array[:, :, fix_nodes_dc2, 1]
    check1 = jnp.sum(fix_R_nodes_dc1, axis = 2)
    check2 = jnp.sum(fix_R_nodes_dc2, axis = 2)
    R_dc1 = load_array[None, :, 0] + jnp.sum(fix_R_nodes_dc1, axis = 2)
    R_dc2 = load_array[None, :, 1] + jnp.sum(fix_R_nodes_dc2, axis = 2)
    # data_reaction = jnp.concat([R_dc1.flatten(), R_dc2.flatten()])
    data_reaction = jnp.concat([R_dc1.flatten(), R_dc2.flatten()])

    # 2. Create the x-range for the theoretical curve
    # Use 4 sigma to cover 99.9% of the distribution
    x_axis = np.linspace(-4 * learned_gp.params.sigma_phys, 4 * learned_gp.params.sigma_phys, 100)
    x_glob_axis = np.linspace(-4 * learned_gp.params.sigma_glob, 4 * learned_gp.params.sigma_glob, 100)

    theoretical_pdf = norm.pdf(x_axis, 0, learned_gp.params.sigma_phys)
    theoretical_glob_pdf = norm.pdf(x_glob_axis, 0, learned_gp.params.sigma_glob)


    lower_bound = -1.96 * learned_gp.params.sigma_phys
    upper_bound = 1.96 * learned_gp.params.sigma_phys

    glob_lower_bound = -1.96 * learned_gp.params.sigma_glob
    glob_upper_bound = 1.96 * learned_gp.params.sigma_glob

    # 3. Plotting
    plt.figure(figsize=(8, 5))

    # Plot empirical density (Density=True scales the area to 1)
    plt.hist(data_free, bins=500, density=True, alpha=0.6, color='skyblue', label='Empirical Residuals')

    # Plot learned Normal distribution
    plt.plot(x_axis, theoretical_pdf, 'r-', lw=2, label=f'Model N(0, σ={learned_gp.params.sigma_phys:.2e})')
    plt.axvline(lower_bound, color='green', linestyle='--', linewidth=2, label='95% Bound ($\pm 1.96\sigma$)')
    plt.axvline(upper_bound, color='green', linestyle='--', linewidth=2)

    # Fill the 95% region
    plt.axvspan(lower_bound, upper_bound, color='green', alpha=0.1)
    print("Plot saved to uq_verification.png")
    within_bounds = np.sum((data_free >= lower_bound) & (data_free <= upper_bound))
    percentage = (within_bounds / len(data_free)) * 100
    print(f"Actual data within 95% interval: {percentage:.2f}%")
    # Add text annotation
    plt.text(0, plt.gca().get_ylim()[1]*0.8, '95% Confidence\nInterval', 
            horizontalalignment='center', color='darkgreen', fontweight='bold')
    plt.text(0, plt.gca().get_ylim()[1] * 0.7, f'Actual: {percentage:.2f}% within bounds', 
            horizontalalignment='center', color='darkgreen', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.title("UQ Verification: Residual Density vs. Learned Noise")
    plt.xlabel("Residual Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(-0.012, 0.012)
    # Since we are using 'Agg' backend, save it
    plt.savefig(os.path.join("output", 'uq_free_verification.png'))


    plt.figure(figsize=(8, 5))

    # Plot empirical density (Density=True scales the area to 1)
    plt.hist(data_reaction, bins=100, density=True, alpha=0.6, color='skyblue', label='Empirical Residuals')

    # Plot learned Normal distribution
    plt.plot(x_glob_axis, theoretical_glob_pdf, 'r-', lw=2, label=f'Model N(0, σ={learned_gp.params.sigma_glob:.2e})')
    plt.axvline(glob_lower_bound, color='green', linestyle='--', linewidth=2, label='95% Bound ($\pm 1.96\sigma$)')
    plt.axvline(glob_upper_bound, color='green', linestyle='--', linewidth=2)

    # Fill the 95% region
    plt.axvspan(glob_lower_bound, glob_upper_bound, color='green', alpha=0.1)
    print("Plot saved to uq_verification.png")
    within_bounds = np.sum((data_reaction >= glob_lower_bound) & (data_reaction <= glob_upper_bound))
    percentage = (within_bounds / len(data_reaction)) * 100
    print(f"Actual data within 95% interval: {percentage:.2f}%")
    # Add text annotation
    plt.text(0, plt.gca().get_ylim()[1]*0.8, '95% Confidence\nInterval', 
            horizontalalignment='center', color='darkgreen', fontweight='bold')
    plt.text(0, plt.gca().get_ylim()[1] * 0.7, f'Actual: {percentage:.2f}% within bounds', 
            horizontalalignment='center', color='darkgreen', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    plt.title("UQ Verification: Residual Density vs. Learned Noise")
    plt.xlabel("Residual Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.xlim(-0.012, 0.012)
    # Since we are using 'Agg' backend, save it

    plt.savefig(os.path.join("output", 'uq_reaction_verification.png'))
    # data = residual_free.flatten()

    # 2. Setup the theoretical PDF for comparison
    # x_range = np.linspace(-0.2, 0.2, 500)
    # pdf_vals = norm.pdf(x_range, 0, learned_gp.params.sigma_phys)

    # plt.figure(figsize=(10, 6))

    # # Plot the red Model curve
    # plt.plot(x_range, pdf_vals, color='red', lw=2, label=f'Model N(0, σ={learned_gp.params.sigma_phys:.2e})', zorder=3)

    # # 3. Create Scatter with Jitter
    # # We use random y-values to spread points vertically so they don't overlap in one line
    # y_jitter = np.zeros_like(data)
    # plt.scatter(data_free, y_jitter, color='skyblue', alpha=0.3, s=8, label='Individual Residuals', zorder=2)

    # # 4. Optional: Add a 'Rug Plot' (marks on the bottom axis)
    # plt.plot(data_free, np.zeros_like(data) - (pdf_vals.max()*0.05), '|', color='navy', alpha=0.2, label='Exact Locations')

    # plt.title("UQ Verification: Residual Scatter vs. Learned Noise")
    # plt.xlabel("Residual Value")
    # plt.ylabel("Probability Density / Jitter Height")
    # plt.axvline(0, color='black', linestyle='--', alpha=0.2)
    # plt.legend()
    # plt.grid(alpha=0.2)
    # plt.xlim(-0.2, 0.2) # Adjust based on your outlier range
    # plt.savefig('residual_scatter_uq.png')