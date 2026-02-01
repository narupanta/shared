import jax 
import jax.numpy as jnp
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from core.model import SparseHyperelasticityGP, transform_input_features, enforce_softplus_positive, GPRawParams, GPParams, GPWeights
from core.material_models import get_material
import jax
import jax.numpy as jnp
from core.utils import *
import datetime
import os
from tqdm import tqdm
from core.datasetclass import TractionDataset
from core.loss_function import physical_loss, elbo_loss, total_loss
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
    material_model = "gentthomas"
    dataset_name = "gentthomas"
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
        load_noise = 0.03
        load = data["load"] 
        load += np.random.normal(0, load_noise * load, load.shape)
        reaction = data["reaction"]
        coord_cells = coords[cells]
        u_cells = u[cells]
        
        F, dNdx = deformation_gradient_element(coord_cells, u_cells)
        loads.append(load)
        u_all.append(u)
        F_all.append(F)
        reactions.append(reaction)
    F_all_stacked = jnp.concat(F_all)
    invariants_and_derivatives_vmap = jax.vmap(invariants_and_derivatives)
    I_obs_all,_ = invariants_and_derivatives_vmap(F_all_stacked)

    F_array = jnp.array(F_all)
    reactions_array = jnp.array(reactions)
    u_array = jnp.array(u_all)
    loads = jnp.array(loads)

    I_all_dev, j = jax.vmap(transform_input_features)(I_obs_all)

    n_ip = 10

    from sklearn.cluster import KMeans

    dev_z = I_all_dev[farthest_point_sampling(I_all_dev, n_ip)]
    vol_z = j[farthest_point_sampling(j, n_ip)]
    plot_inducing_points(dev_z, vol_z, I_all_dev, j, save_path)
    I_z = jnp.concat([dev_z, vol_z], axis = -1)

    # params = {
    #     "raw_dev_gp_lengthscales" : jnp.array([1.0, 1.0]), 
    #     "raw_vol_gp_lengthscales" : jnp.array([1.0]), 
    #     "raw_dev_gp_sigma_scaling" : jnp.array(1.0), # Wrapped
    #     "raw_vol_gp_sigma_scaling" : jnp.array(1.0), # Wrapped
    #     "raw_dev_z" : dev_z,
    #     "raw_dev_u_mean" : jnp.zeros((n_ip,)),
    #     "raw_dev_u_var" : jnp.ones((n_ip,)),
    #     "raw_vol_z" : vol_z,
    #     "raw_vol_u_mean" : jnp.zeros((n_ip,)),
    #     "raw_vol_u_var" : jnp.ones((n_ip,)),
    #     "log_sigma_physic": jnp.array(1.0), # Wrapped
    #     "log_sigma_glob": jnp.array(1.0),   # Wrapped
    #     "raw_c20": jnp.array(1.0),          # Wrapped
    #     "raw_c02": jnp.array(1.0),          # Wrapped
    #     "raw_c11": jnp.array(1.0),          # Wrapped
    #     "raw_c10": jnp.array(1.0),          # Wrapped
    #     "raw_c01": jnp.array(1.0),          # Wrapped
    #     "raw_k": jnp.array(1.0),            # Wrapped
    #     "raw_q": jnp.array(1.0)             # Wrapped
    # }
    params = GPRawParams(
        raw_dev_ls=jnp.array([1.0, 1.0]),
        raw_dev_sig=jnp.array(1.0),
        raw_dev_u_mean=jnp.zeros((n_ip,)),
        raw_dev_u_var=jnp.ones((n_ip,)),

        raw_vol_ls=jnp.array([1.0]),
        raw_vol_sig=jnp.array(1.0),
        raw_vol_u_mean=jnp.zeros((n_ip,)),
        raw_vol_u_var=jnp.ones((n_ip,)),

        raw_c01=jnp.array(1.0),
        raw_c02=jnp.array(1.0),
        raw_c10=jnp.array(1.0),
        raw_c11=jnp.array(1.0),
        raw_c20=jnp.array(1.0),
        raw_k=jnp.array(1.0),
        raw_q=jnp.array(1.0),

        log_sigma_phys=jnp.array(1.0),
        log_simga_glob=jnp.array(1.0)
    )
    def calculate_min_ls(z):
        # For a 2D/3D point cloud, a quick way is to use the 
        # average distance to the nearest neighbor.
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(z)
        distances, _ = nbrs.kneighbors(z)
        avg_dist = jnp.mean(distances[:, 1])
        return avg_dist * 0.5 # Minimum allowable lengthscale
    
    min_dev = calculate_min_ls(dev_z)
    min_vol = calculate_min_ls(vol_z)

    main_key = jr.PRNGKey(42)
    model = SparseHyperelasticityGP(params, I_z, min_dev, min_vol)
    model_path = "/home/mmdiscovery/shared/saved_model/20260131T104933/" # Replace with the actual path to your saved model
    with open(os.path.join(model_path, "best_params.npy"), "rb") as f:
        loaded_dict = jnp.load(f, allow_pickle=True).item()
        loaded_params = GPRawParams(**loaded_dict)
    model.params = model.load_params(loaded_params)
    params = loaded_params
    # loss_and_grad = jax.jit(jax.value_and_grad(
    #     lambda p, k: elbo_loss(p, model, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter, k),
    #     has_aux=True
    # ))

    # loss_and_grad = jax.jit(jax.value_and_grad(
    #     lambda p, k: total_loss(p, model, u_array, loads, reactions_array, coords, cells, node_type, k),
    #     has_aux=True
    # ))

    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda p, k: total_loss(p, model, u_array, loads, reactions_array, coords, cells, node_type, k),
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
    # import optax
    # from tqdm import tqdm
    # import jaxopt
    # # trainable parameters
    # # params = jnp.zeros(6) 
    # def loss_fn(p):
    #     loss_val, aux = total_loss(p, model, u_array, loads, reactions_array, coords, cells, node_type, None)
    #     return loss_val

    #     # return total_loss(p, model, u_array, loads, reactions_array, coords, cells, node_type, k)[0]
    # solver = jaxopt.LBFGS(fun=loss_fn, maxiter=500, verbose = True)

    # # 3. Use jax.lax.scan for the inner loop (if doing multiple restarts/steps)
    # # However, for L-BFGS, usually one call to .run() is enough
    # res = solver.run(params)
    # params = res.params
    # print(f"Final loss: {res.state.error}")
    total_step = 50000
    pbar = tqdm(range(total_step), desc="Training Sparse GP", unit="step")
    true_model = get_material(material_model.lower())
    for step in pbar:
        main_key, subkey = jr.split(main_key)
        
        # JAX execution
        (loss, (log_like_loss, kl_loss, phy_loss, phys_loss2)), grads = loss_and_grad(params, subkey)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = params

        if step % 50 == 0:
            # Update the progress bar postfix with current metrics
            # This shows up right next to the time left
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "phy": f"{phy_loss:.4f}",
                "phy2": f"{phys_loss2:.4f}"
            })

            # --- Your existing logging logic ---
            # Note: Using pbar.write() instead of print() prevents 
            # the progress bar from breaking into multiple lines.
            log_message = (
                f"step {step:04d} | loss={loss:.6f} | "
                f"log_like={log_like_loss:.6f} | kl={kl_loss:.6f} | "
                f"phy={phy_loss:.6f} | phy2 ={phys_loss2:.6f}\n"
            )
            cur_params = model.load_params(params)
            clean_params = jax.tree_util.tree_map(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x, 
                cur_params
            )
            log_message += f"params: {clean_params}\n"
            log_message += "-"*50 + "\n"
            # ... (rest of your params_hist recording) ...
            steps_history.append(step)
            loss_components_hist["total_loss"].append(float(loss))
            loss_components_hist["log_like"].append(float(log_like_loss))
            loss_components_hist["kl"].append(float(kl_loss))
            loss_components_hist["phy"].append(float(phy_loss))
            params_hist["dev_gp_sigma_scaling"].append(cur_params.dev_sig)
            params_hist["vol_gp_sigma_scaling"].append(cur_params.vol_sig)
            params_hist["dev_gp_lengthscales"].append(cur_params.dev_ls)
            params_hist["vol_gp_lengthscales"].append(cur_params.vol_ls)
            params_hist["dev_z"].append(model.dev_z)
            params_hist["vol_z"].append(model.vol_z)
            params_hist["dev_u_mean"].append(cur_params.dev_u_mean)
            params_hist["dev_u_var"].append(cur_params.dev_u_var)
            params_hist["vol_u_mean"].append(cur_params.vol_u_mean)
            params_hist["vol_u_var"].append(cur_params.vol_u_var)
            params_hist["sigma_physic"].append(cur_params.sigma_phys)
            params_hist["c20"].append(cur_params.c20)
            params_hist["c02"].append(cur_params.c02)
            params_hist["c11"].append(cur_params.c11)
            params_hist["c10"].append(cur_params.c10)
            params_hist["c01"].append(cur_params.c01)
            params_hist["k"].append(cur_params.k)
            params_hist["q"].append(cur_params.q)
            # Append to log file
            with open(log_file_path, "a") as f:
                f.write(log_message)
        if step % (total_step//10) == 0 and step != 0:
            
            plot_model = SparseHyperelasticityGP(best_params, I_z, min_dev, min_vol)
            plot_model.params = plot_model.load_params(best_params)
            plot_model.gpweight = plot_model.precompute_weights(best_params)
            
            plot_ut_ebt_ps_uc_ebc_ss(plot_model, true_model, save_path, step)
            # plot material_modes_validation_step.png

    # for step in range(20000):
    #     main_key, subkey = jr.split(main_key)
        
    #     (loss, (log_like_loss, kl_loss, phy_loss, phys_loss2)), grads = loss_and_grad(params, subkey)
    
    #     updates, opt_state = opt.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)
        
    #     losses.append(loss)
    #     if loss < best_loss:
    #         best_loss = loss
    #         best_params = params

    #     if step % 50 == 0:
    #         # Format the log entry
    #         log_message = (

    #             f"step {step:04d} | loss={loss:.6f} | "
    #             f"log_like={log_like_loss:.6f} | kl={kl_loss:.6f} | phy={phy_loss:.6f} | phy2 ={phys_loss2:.6f}\n"
    #         )
            
    #         # Cleanly format params for readability
    #         clean_params = jax.tree_util.tree_map(
    #             lambda x: x.tolist() if hasattr(x, 'tolist') else x, 
    #             params
    #         )
    #         log_message += f"params: {clean_params}\n"
    #         log_message += "-"*50 + "\n"

    #         # Print to console
    #         print(f"step {step:04d}  loss={loss:.6f}, phy_loss = {phy_loss:.6f}, phy2 ={phys_loss2:.6f}")

    #         # Append to log file
    #         with open(log_file_path, "a") as f:
    #             f.write(log_message)
            
    #         # Append to history lists
    #         steps_history.append(step)
            
    #         # Record Losses


    # Final Save
    with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
        jnp.save(f, best_params._asdict())
    with open(os.path.join(save_path, "I_z.npy"), "wb") as f:
        jnp.save(f, I_z)
    with open(os.path.join(save_path, "I_obs_all.npy"), "wb") as f:
        jnp.save(f, I_obs_all)
    # Save the best parameters
    # with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
    #     jnp.save(f, jax.tree_util.tree_map(lambda x: x, best_params)) # type: ignore
    # Path to your existing log
    log_path = save_path + "/optimization_log.txt"# Change to your actual folder
    plot_loss_analysis(loss_components_hist, params_hist, steps_history, save_path)
    plot_parameters_hist(params_hist, steps_history, save_path)

    learned_gp = SparseHyperelasticityGP(best_params, I_z, min_dev, min_vol)

    # f = jax.vmap(fto3x3)(F)
    # psi_pred = jax.vmap(learned_gp.psi, in_axes=(0,None))(f, None)
    # dev_features, vol_features = jax.vmap(transform_input_features)(I_obs)
    # # psi_dev_pred = jax.vmap(lambda f : learned_gp._predict_gp_component(f, "dev").mean)(dev_features).squeeze()
    # # psi_vol_pred = jax.vmap(lambda f : learned_gp._predict_gp_component(f, "vol").mean)(vol_features).squeeze()

    true_model = get_material(material_model.lower())
    # psi_true = true_model.phi(f)
    # j = jnp.sqrt(I_obs[:, 2])
    # # psi_dev_true = 0.5 * (j **(-2/3) * I_obs[:, 0] - 3) + (j **(-2/3) * I_obs[:, 0] - 3)**2 + (j **(-4/3) * I_obs[:, 1] - 3)
    # psi_dev_true = psi_true - 1.5 * (j - 1)**2

    # psi_vol_true = 1.5 * (j - 1)**2

    # plot_r2_strain_energy_function(psi_pred, psi_true, psi_dev_pred, psi_dev_true, psi_vol_pred, psi_vol_true, save_path)
    plot_ut_ebt_ps_uc_ebc_ss(learned_gp, true_model, save_path, step)
    plot_stress_validation(learned_gp, true_model, save_path)