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
from core.model import TensorBasisSVGP, transform_input_features, enforce_softplus_positive
from core.material_models import get_material
import jax
import jax.numpy as jnp
from core.utils import *
import datetime
import os

from core.datasetclass import TractionDataset
from core.loss_function import physical_loss, elbo_loss
from core.plotter import \
    plot_loss_analysis, \
    plot_parameters_hist, \
    plot_r2_strain_energy_function, plot_ut_ebt_ps_uc_ebc_ss, plot_inducing_points, plot_stress_validation

# helper: per-element edge-based neumann traction contribution
import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
# from core.plotter import plot_loss_analysis, plot_parameters_hist

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
    F_all = []
    for loadstep in range(len(dataset)) :

        data = dataset[loadstep]
        coords = data["mesh_pos"][:,:2]
        cells = data["cells"]
        # u = data["u"]
        percent_noise = 0.0
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
        F_all.append(F)
    F_all_stacked = jnp.concat(F_all)
    invariants_and_derivatives_vmap = jax.vmap(invariants_and_derivatives)
    I_obs_all,_ = invariants_and_derivatives_vmap(F_all_stacked)
    I_obs, _ = invariants_and_derivatives_vmap(F)

    I_obs_dev, j = jax.vmap(transform_input_features)(I_obs)

    n_ip = 20

    params = {
        "raw_a1_gp_lengthscales" : jnp.array([1.0, 1.0, 1.0]),
        "raw_a1_gp_sigma_scaling" : 1.0,
        "raw_a1_u_mean" : jnp.zeros((n_ip,)),
        "raw_a1_u_var" : jnp.ones((n_ip,)),
        
        "raw_a2_gp_lengthscales" : jnp.array([1.0, 1.0, 1.0]),
        "raw_a2_gp_sigma_scaling" : 1.0,
        "raw_a2_u_mean" : jnp.zeros((n_ip,)),
        "raw_a2_u_var" : jnp.ones((n_ip,)),

        "raw_a3_gp_lengthscales" : jnp.array([1.0, 1.0, 1.0]),
        "raw_a3_gp_sigma_scaling" : 1.0,
        "raw_a3_u_mean" : jnp.zeros((n_ip,)),
        "raw_a3_u_var" : jnp.ones((n_ip,)),

        "log_sigma_physic": 1.0,
        "log_sigma_glob": 1.0,

    }

    main_key = jr.PRNGKey(42)
    model = TensorBasisSVGP(params, I_obs, n_ip)
    # model_path = "/home/mmdiscovery/shared/saved_model/20260121T185159/" # Replace with the actual path to your saved model
    # with open(os.path.join(model_path, "best_params.npy"), "rb") as f:
    #     load_params = jnp.load(f, allow_pickle=True).item()
    # model.load_params(load_params)
    # params = load_params
    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda p, k: elbo_loss(p, model, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter, k),
        has_aux=True
    ))

    losses = []
    best_loss = float('inf')
    g_mean_history = []
    lengthscales_history = []
    log_scale_variance_history = []
    best_params = None
    log_file_path = os.path.join(save_path, "optimization_log.txt")
    lr_schedule = optax.linear_schedule(
        init_value=1e-3, 
        end_value=1e-2, 
        transition_steps=5000,
        transition_begin=20000
        )
    opt = optax.adam(learning_rate=1e-2)
    opt_state = opt.init(params)
    n_nodes = int(cells.max()) + 1
    # Open once and clear (or just let the loop handle it)
    with open(log_file_path, "w") as f:
        f.write("Optimization Start\n" + "="*20 + "\n")
    # Lists to store history
    steps_history = []
    loss_components_hist = {"total_loss": [],"log_like": [], "kl": [], "phy": []}

    # Parameter history
    params_hist = {"a1_gp_lengthscales": [], "a1_gp_sigma_scaling": [], "a1_u_mean": [], "a1_u_var": [], 
                   "a2_gp_lengthscales": [], "a2_gp_sigma_scaling": [], "a2_u_mean": [], "a2_u_var": [], 
                   "a3_gp_lengthscales": [], "a3_gp_sigma_scaling": [], "a3_u_mean": [], "a3_u_var": [], 
                   "sigma_physic": [], "sigma_glob": []}
    for step in range(20000):
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
            params_hist["a1_gp_lengthscales"].append(jnp.exp(params["raw_a1_gp_lengthscales"]))
            params_hist["a1_gp_sigma_scaling"].append(jnp.exp(params["raw_a1_gp_sigma_scaling"]))
            params_hist["a1_u_mean"].append(params["raw_a1_u_mean"])
            params_hist["a1_u_var"].append(jnp.exp(params["raw_a1_u_var"]))

            params_hist["a2_gp_lengthscales"].append(jnp.exp(params["raw_a2_gp_lengthscales"]))
            params_hist["a2_gp_sigma_scaling"].append(jnp.exp(params["raw_a2_gp_sigma_scaling"]))
            params_hist["a2_u_mean"].append(params["raw_a2_u_mean"])
            params_hist["a2_u_var"].append(jnp.exp(params["raw_a2_u_var"])) 

            params_hist["a3_gp_lengthscales"].append(jnp.exp(params["raw_a3_gp_lengthscales"]))
            params_hist["a3_gp_sigma_scaling"].append(jnp.exp(params["raw_a3_gp_sigma_scaling"]))
            params_hist["a3_u_mean"].append(params["raw_a3_u_mean"])
            params_hist["a3_u_var"].append(jnp.exp(params["raw_a3_u_var"]))

            params_hist["sigma_physic"].append(jnp.exp(params["log_sigma_physic"]))
            params_hist["sigma_glob"].append(jnp.exp(params["log_sigma_glob"]))
        

            # Keep your history lists updated


    # Final Save
    with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
        jnp.save(f, best_params)
    with open(os.path.join(save_path, "invariants_obs.npy"), "wb") as f:
        jnp.save(f, I_obs)
    # Save the best parameters
    # with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
    #     jnp.save(f, jax.tree_util.tree_map(lambda x: x, best_params)) # type: ignore
    # Path to your existing log
    log_path = save_path + "/optimization_log.txt"# Change to your actual folder
    plot_loss_analysis(loss_components_hist, params_hist, steps_history, save_path)
    # plot_parameters_hist(params_hist, steps_history, save_path)

    learned_gp = TensorBasisSVGP(best_params, I_obs, n_ip)
    true_model = get_material(material_model.lower())

    # plot_ut_ebt_ps_uc_ebc_ss(learned_gp, true_model, save_path)
    plot_stress_validation(learned_gp, true_model, save_path)