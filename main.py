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

from core.datasetclass import TractionDataset
from core.loss_function import physical_loss, elbo_loss
from core.plotter import plot_loss_analysis, plot_parameters_hist, plot_r2_strain_energy_function, plot_ut_ebt_ps_uc_ebc_ss

# helper: per-element edge-based neumann traction contribution
import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
# from core.plotter import plot_loss_analysis, plot_parameters_hist

@jax.vmap
def invariants_and_derivatives(F):
    f = fto3x3(F)
    C = f.T @ f
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C @ C))
    I3 = jnp.linalg.det(C)
    # derivatives wrt F (2x2)
    dI1_dF = 2*f
    dI2_dF = 2*(I1*f - f @ C)
    dI3_dF = 2*jnp.linalg.det(f)**2 * jnp.linalg.inv(f).T
    dI_dF = jnp.stack([dI1_dF, dI2_dF, dI3_dF])  # (3,2,2)
    return jnp.array([I1, I2, I3]), dI_dF

def farthest_point_sampling(pts, num_samples):
    """
    pts: (N, 3) array of points
    num_samples: 25
    """
    n_pts = pts.shape[0]
    # Initialize: pick the first point in the list as the start
    selected_indices = jnp.zeros(num_samples, dtype=jnp.int32)
    
    # Track the distance from every point to its NEAREST selected point
    # Start with infinity
    dist_to_set = jnp.full((n_pts,), jnp.inf)
    
    def scan_body(dist_to_set, i):
        # The next point is the one farthest from the current set
        idx = jnp.argmax(dist_to_set)
        
        # Calculate distance from the new point to all other points
        new_pt = pts[idx]
        dists = jnp.sum((pts - new_pt)**2, axis=-1) # Squared Euclidean
        
        # Update distances: dist to set is min(old_dist, dist_to_new_point)
        dist_to_set = jnp.minimum(dist_to_set, dists)
        
        return dist_to_set, idx

    # We manually pick the first point to start
    first_idx = 0
    dist_to_set = jnp.sum((pts - pts[first_idx])**2, axis=-1)
    
    # Run the loop for the remaining 24 points
    _, remaining_indices = jax.lax.scan(scan_body, dist_to_set, jnp.arange(1, num_samples))
    
    return jnp.concatenate([jnp.array([first_idx]), remaining_indices])

if __name__ == "__main__" :
    base_save_path = "saved_model"  # change as needed
    os.makedirs(base_save_path, exist_ok=True)

    # Subfolder with datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    save_path = os.path.join(base_save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)
    material_model = "isihara"
    dataset = TractionDataset("dataset",material_model)
    data = dataset[-1]
    coords = data["mesh_pos"][:,:2]
    cells = data["cells"]
    # u = data["u"]
    percent_noise = 0.00005
    node_type = data["node_type"]
    ux = data["u"][:, 0]
    ux[(data["node_type"] != 1)] += np.random.normal(0, percent_noise * 1, ux.shape)[(data["node_type"] != 1)]
    uy = data["u"][:, 1]
    uy[(data["node_type"] != 2)] += np.random.normal(0, percent_noise * 1, uy.shape)[(data["node_type"] != 2)]

    # Combine components into the full displacement vector u
    u = np.column_stack((ux, uy))
    # u[node_type == 0] = u[node_type == 0] + jax.random.normal(jr.key(0), u.shape)[node_type == 0] * 0.01 * mean_u
    load_parameter = data["load_parameter"]

    coord_cells = coords[cells]
    u_cells = u[cells]

    F, dNdx = deformation_gradient_element(coord_cells, u_cells)
    
    I_obs, _ = invariants_and_derivatives(F)
    lscale_init = jnp.std(I_obs, axis = 0)
    x = I_obs[:, 0] * I_obs[:, 2]**(-1/3)
    y = I_obs[:, 1] * I_obs[:, 2]**(-2/3)
    z = I_obs[:, 2]
    I_obs_dev = jnp.stack([x, y, z], axis = -1)
    
    n_ip = 20
    # Z_i1 = jnp.linspace(I_obs[:, 0].min(), I_obs[:, 0].max(), n_ip)
    # Z_i2 = jnp.linspace(I_obs[:, 1].min(), I_obs[:, 1].max(), n_ip)
    # Z_i3 = jnp.linspace(I_obs[:, 2].min(), I_obs[:, 2].max(), n_ip)

    # Z_grid_i1, Z_grid_i2, Z_grid_i3 = jnp.meshgrid(Z_i1, Z_i2, Z_i3)
    # Z_stacked = jnp.stack([Z_grid_i1.flatten(), Z_grid_i2.flatten(), Z_grid_i3.flatten()], axis=-1)

    # Z_stacked = jnp.stack([Z_i1, Z_i2, Z_i3], axis = -1)

    from sklearn.cluster import KMeans

    # Find the 50 most representative points in your invariant data
    # kmeans = KMeans(n_clusters=n_ip, random_state=0).fit(I_obs)
    # Z_stacked = jnp.array(kmeans.cluster_centers_)
    Z_stacked = I_obs[farthest_point_sampling(I_obs, n_ip)]


    params = {
        "lengthscales": lscale_init[:2],
        "log_sigma_scaling": jnp.array(1.0),
        "log_sigma_poly": jnp.array(1.0),
        "offset": jnp.array(1.0),
        "log_growth_constant": jnp.array(1.0),
        "poly_degree": 2.0,
        # "inducing_invariants": Z_stacked,
        "log_inducing_latent_variable_mean": jnp.ones((n_ip,)) * 0.0,
        "log_inducing_latent_variable_var": jnp.ones((n_ip,)),
        "log_sigma_physic": 1.0,
        "p": 1.0,
        "q": 1.0,
        "r": 1.0,
        "s": 1.0,
        "t": 1.0,
        "c": 1.0,
    }

    opt = optax.adam(1e-2)
    opt_state = opt.init(params)
    n_nodes = int(cells.max()) + 1
    # JIT the loss and gradients
    # loss_and_grad = jax.jit(jax.value_and_grad( # type: ignore
    #     lambda p: physical_loss(p, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter), 
    #             has_aux=True
    # ))
    main_key = jr.PRNGKey(42)

    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda p, k: elbo_loss(p, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter, k),
        has_aux=True
    ))

    losses = []
    best_loss = float('inf')
    g_mean_history = []
    lengthscales_history = []
    log_scale_variance_history = []
    best_params = None
    log_file_path = os.path.join(save_path, "optimization_log.txt")
    
    # Open once and clear (or just let the loop handle it)
    with open(log_file_path, "w") as f:
        f.write("Optimization Start\n" + "="*20 + "\n")
    # Lists to store history
    steps_history = []
    loss_components_hist = {"total_loss": [],"log_like": [], "kl": [], "phy": []}

    # Parameter history
    params_hist = {
        "sigma_poly": [], "sigma_scaling": [], "lengthscales": [],
        "offset": [], "degree": [], "sigma_physic": [], "inducing_mean": []
    }
    for step in range(50000):
        main_key, subkey = jr.split(main_key)
        
        (loss, (log_like_loss, kl_loss, phy_loss)), grads = loss_and_grad(params, subkey)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = params

        if step % 50 == 0:
            # Format the log entry
            log_message = (
                f"step {step:04d} | loss={loss:.6f} | "
                f"log_like={log_like_loss:.6f} | kl={kl_loss:.6f} | phy={phy_loss:.6f}\n"
            )
            
            # Cleanly format params for readability
            clean_params = jax.tree_util.tree_map(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x, 
                params
            )
            log_message += f"params: {clean_params}\n"
            log_message += "-"*50 + "\n"

            # Print to console
            print(f"step {step:04d}  loss={loss:.6f}, phy_loss = {phy_loss:.6f}")

            # Append to log file
            with open(log_file_path, "a") as f:
                f.write(log_message)
            
            # Append to history lists
            steps_history.append(step)
            
            # Record Losses
            loss_components_hist["total_loss"].append(float(loss))
            loss_components_hist["log_like"].append(float(log_like_loss))
            loss_components_hist["kl"].append(float(kl_loss))
            loss_components_hist["phy"].append(float(phy_loss))
            
            # Record Parameters (applying exp where necessary)
            params_hist["sigma_poly"].append(np.exp(float(params["log_sigma_poly"])))
            params_hist["sigma_scaling"].append(np.exp(float(params["log_sigma_scaling"])))
            params_hist["lengthscales"].append(np.array(params["lengthscales"]))
            params_hist["offset"].append(float(params["offset"]))
            params_hist["degree"].append(float(params["poly_degree"]))
            params_hist["sigma_physic"].append(np.exp(float(params["log_sigma_physic"])))
            params_hist["inducing_mean"].append(np.array(jnp.exp(params["log_inducing_latent_variable_mean"])))
            # Keep your history lists updated


    # Final Save
    with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
        jnp.save(f, best_params)
    with open(os.path.join(save_path, "z_stacked.npy"), "wb") as f:
        jnp.save(f, Z_stacked)
    # Save the best parameters
    # with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
    #     jnp.save(f, jax.tree_util.tree_map(lambda x: x, best_params)) # type: ignore
    # Path to your existing log
    log_path = save_path + "/optimization_log.txt"# Change to your actual folder
    plot_loss_analysis(loss_components_hist, params_hist, steps_history, save_path)
    plot_parameters_hist(params_hist, steps_history, save_path)

    learned_gp = SparseHyperelasticityGP(best_params, Z_stacked)

    f = jax.vmap(fto3x3)(F)
    psi_pred = jax.vmap(learned_gp.psi)(f)
    psi_dev_pred = psi_pred - jax.vmap(learned_gp._psi_model_vol)(I_obs[:, 2])
    psi_vol_pred = jax.vmap(learned_gp._psi_model_vol)(I_obs[:, 2])

    true_model = get_material(material_model.lower())
    psi_true = true_model.phi(f)
    psi_dev_true = psi_true - 1.5 * (jnp.sqrt(I_obs[:, 2]) - 1)**2
    psi_vol_true = 1.5 * (jnp.sqrt(I_obs[:, 2]) - 1)**2

    plot_r2_strain_energy_function(psi_pred, psi_true, psi_dev_pred, psi_dev_true, psi_vol_pred, psi_vol_true, save_path)
    plot_ut_ebt_ps_uc_ebc_ss(learned_gp, true_model, save_path)