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
from core.model import SparseHyperelasticityGP, transform_input_features, enforce_softplus_positive
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
        percent_noise = 5e-5
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
    dev_z = I_obs_dev[farthest_point_sampling(I_obs_dev, n_ip)]
    vol_z = j[farthest_point_sampling(j, n_ip)]
    plot_inducing_points(dev_z, vol_z, I_obs_dev, j, save_path)
    # params = {
    #     "log_lengthscales": jnp.log(lscale_init),
    #     "log_sigma_scaling": jnp.array(1.0),
    #     "log_sigma_poly": jnp.array(1.0),
    #     "offset": jnp.array(1.0),
    #     "log_growth_constant": jnp.array(1.0),
    #     "poly_degree": 2.0,
    #     # "inducing_invariants": Z_stacked,
    #     "log_inducing_latent_variable_mean": jnp.ones((n_ip,)) * 0.0,
    #     "log_inducing_latent_variable_var": jnp.ones((n_ip,)),
    #     "log_sigma_physic": 1.0,
    #     "p": 1.0,
    #     "q": 1.0,
    #     "r": 1.0,
    #     "s": 1.0,
    #     "t": 1.0,
    #     "c": 1.0,
    # }

    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=1e-2,
    #     peak_value=1e-2,
    #     warmup_steps=1000,
    #     decay_steps=2000,
    #     end_value=1e-5
    # )
    # JIT the loss and gradients
    # loss_and_grad = jax.jit(jax.value_and_grad( # type: ignore
    #     lambda p: physical_loss(p, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter), 
    #             has_aux=True
    # ))

    params = {
        "raw_dev_gp_lengthscales" : jnp.array([1.0, 1.0]), 
        "raw_vol_gp_lengthscales" : jnp.array([1.0]), 
        "raw_dev_gp_sigma_scaling" : 1.0,
        "raw_vol_gp_sigma_scaling" : 1.0,
        "raw_dev_z" : dev_z,
        "raw_dev_u_mean" : jnp.zeros((n_ip,)),
        "raw_dev_u_var" : jnp.ones((n_ip,)),
        "raw_vol_z" : vol_z,
        "raw_vol_u_mean" : jnp.zeros((n_ip,)),
        "raw_vol_u_var" : jnp.ones((n_ip,)),
        "log_sigma_physic": 1.0,
        "log_sigma_glob": 1.0,
        "raw_c20": 1.0,
        "raw_c02": 1.0,
        "raw_c11": 1.0,
        "raw_c10": 1.0,
        "raw_c01": 1.0,
        "raw_k": 1.0,
        "raw_q": 1.0
    }

    main_key = jr.PRNGKey(42)
    model = SparseHyperelasticityGP(params, I_obs, n_ip)
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
    params_hist = {
        "dev_gp_sigma_scaling": [], "vol_gp_sigma_scaling": [],
        "dev_gp_lengthscales": [], "vol_gp_lengthscales": [], 
        "dev_u_mean": [], "dev_u_var": [], "vol_u_mean": [], "vol_u_var": [], "dev_z": [], "vol_z": [],
        "sigma_physic": [], "c20": [], "c02": [], "c11": [], "c10": [], "c01": [], "k": [], "q": []
    }
    for step in range(100000):
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
            params_hist["dev_gp_sigma_scaling"].append(jnp.exp(params["raw_dev_gp_sigma_scaling"]))
            params_hist["vol_gp_sigma_scaling"].append(jnp.exp(params["raw_vol_gp_sigma_scaling"]))
            params_hist["dev_gp_lengthscales"].append(jnp.exp(params["raw_dev_gp_lengthscales"]))
            params_hist["vol_gp_lengthscales"].append(jnp.exp(params["raw_vol_gp_lengthscales"]))
            params_hist["dev_z"].append(3 + enforce_softplus_positive(params["raw_dev_z"]))
            params_hist["vol_z"].append(enforce_softplus_positive(params["raw_vol_z"]))
            params_hist["dev_u_mean"].append(enforce_softplus_positive(params["raw_dev_u_mean"]))
            params_hist["dev_u_var"].append(enforce_softplus_positive(params["raw_dev_u_var"]))
            params_hist["vol_u_mean"].append(enforce_softplus_positive(params["raw_vol_u_mean"]))
            params_hist["vol_u_var"].append(enforce_softplus_positive(params["raw_vol_u_var"]))
            params_hist["sigma_physic"].append(np.exp(float(params["log_sigma_physic"])))
            params_hist["c20"].append(enforce_softplus_positive(float(params["raw_c20"])))
            params_hist["c02"].append(enforce_softplus_positive(float(params["raw_c02"])))
            params_hist["c11"].append(enforce_softplus_positive(float(params["raw_c11"]))) 
            params_hist["c10"].append(enforce_softplus_positive(float(params["raw_c10"])))
            params_hist["c01"].append(enforce_softplus_positive(float(params["raw_c01"])))
            params_hist["k"].append(enforce_softplus_positive(float(params["raw_k"])))
            params_hist["q"].append(enforce_softplus_positive(float(params["raw_q"])))
            



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

    learned_gp = SparseHyperelasticityGP(best_params, I_obs, n_ip)

    f = jax.vmap(fto3x3)(F)
    psi_pred = jax.vmap(learned_gp.psi, in_axes=(0,None))(f, None)
    dev_features, vol_features = jax.vmap(transform_input_features)(I_obs)
    psi_dev_pred = jax.vmap(lambda f : learned_gp._predict_gp_component(f, "dev").mean)(dev_features).squeeze()
    psi_vol_pred = jax.vmap(lambda f : learned_gp._predict_gp_component(f, "vol").mean)(vol_features).squeeze()

    true_model = get_material(material_model.lower())
    psi_true = true_model.phi(f)
    j = jnp.sqrt(I_obs[:, 2])
    # psi_dev_true = 0.5 * (j **(-2/3) * I_obs[:, 0] - 3) + (j **(-2/3) * I_obs[:, 0] - 3)**2 + (j **(-4/3) * I_obs[:, 1] - 3)
    psi_dev_true = psi_true - 1.5 * (j - 1)**2

    psi_vol_true = 1.5 * (j - 1)**2

    plot_r2_strain_energy_function(psi_pred, psi_true, psi_dev_pred, psi_dev_true, psi_vol_pred, psi_vol_true, save_path)
    plot_ut_ebt_ps_uc_ebc_ss(learned_gp, true_model, save_path)
    plot_stress_validation(learned_gp, true_model, save_path)