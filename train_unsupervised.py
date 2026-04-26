import jax 
import jax.numpy as jnp
from jax import config
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from core.model import SparseHyperelasticityGP, transform_input_features, GPRawParams, GPParams, GPWeights
from core.material_models import get_material
import jax
import jax.numpy as jnp
from core.utils import *
import datetime
import os
from tqdm import tqdm
from core.datasetclass import TractionDataset
from core.training_loop import stochastic_training_loop, deterministic_training_loop 
from core.loss_function import total_stochastic_loss, total_physical_loss, vfm_loss, ell
from core.plotter import \
    plot_loss_analysis, \
    plot_parameters_hist, plot_inducing_points, plot_combined_validation
# helper: per-element edge-based neumann traction contribution
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Isihara Model Dataset and Training Configuration")

    # Dataset & Model Config
    parser.add_argument('--material_model_name', type=str, default="isihara")
    parser.add_argument('--disp_noise', type=float, default=0.000)
    parser.add_argument('--load_noise', type=float, default=0.01)
    parser.add_argument('--target_load_true_top', type=float, default=8.0)
    parser.add_argument('--asym_factor', type=float, default=0.9)

    # Training Config
    parser.add_argument('--number_of_mci_sampling', type=int, default=3)
    parser.add_argument('--n_ip', type=int, default=5)
    parser.add_argument('--beta', type=float, default=50.0)
    
    # Booleans (using 0/1 as integers is often safer in shell scripts)
    parser.add_argument('--is_fixed_reaction_force_noise', type=int, default=0)
    parser.add_argument('--is_include_prior_mean', type=int, default=0)

    # Handling the List [0, 5, 9]
    # 'nargs="+"' allows you to pass multiple space-separated integers
    parser.add_argument('--train_load_steps_indices', type=int, nargs='+', default=[1, 3, 5, 9])
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    return parser.parse_args()

def sigma_fix_to_log_sigma_fix(sigma_fix) :
    return jnp.log(sigma_fix)


def freeze_material_params(grads):
        """Zeros out the gradients for the polynomial material parameters."""
        return grads._replace(
            raw_c01=jnp.zeros_like(grads.raw_c01),
            raw_c02=jnp.zeros_like(grads.raw_c02),
            raw_c10=jnp.zeros_like(grads.raw_c10),
            raw_c11=jnp.zeros_like(grads.raw_c11),
            raw_c20=jnp.zeros_like(grads.raw_c20),
            raw_k=jnp.zeros_like(grads.raw_k),
            raw_q=jnp.zeros_like(grads.raw_q),
            raw_s=jnp.zeros_like(grads.raw_s)
            # Note: If you also want to freeze observation noise, add them here:
            # log_sigma_free_x=jnp.zeros_like(grads.log_sigma_free_x), etc.
        )
def freeze_reaction_force_noise(grads) :
    return grads._replace(
        log_sigma_fix_x=jnp.zeros_like(grads.log_sigma_fix_x),
        log_sigma_fix_y=jnp.zeros_like(grads.log_sigma_fix_y))



if __name__ == "__main__" :
    base_save_path = "saved_model"  # change as needed
    os.makedirs(base_save_path, exist_ok=True)
    # training_mode = "stochastic"
    args = parse_args()

    # Now use args.variable_name instead of hardcoded values
    material_model_name = args.material_model_name

    disp_noise = args.disp_noise
    load_noise = args.load_noise
    target_load_true_top = args.target_load_true_top
    asym_factor = args.asym_factor
    number_of_mci_sampling = args.number_of_mci_sampling
    train_load_steps_indices = args.train_load_steps_indices
    n_ip = args.n_ip
    beta = args.beta
    is_fixed_reaction_force_noise = args.is_fixed_reaction_force_noise
    is_include_prior_mean = args.is_include_prior_mean
    n_iterations = args.n_iterations
    learning_rate = args.learning_rate

    # Subfolder with datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    training_config_str = f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}_{n_ip}_{beta}_{is_fixed_reaction_force_noise}_{is_include_prior_mean}"
    save_path = os.path.join(base_save_path, f"{timestamp}_{training_config_str}")
    os.makedirs(save_path, exist_ok=True)

    # load precomputed dataset
    data_dir = Path("precomputed_vfm") 
    npz_files = list(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in {data_dir}")
    prep_dataset_dir = data_dir / f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}.npz"
    prep_data = jnp.load(prep_dataset_dir)
    f2x2 = prep_data["F"][train_load_steps_indices] 

    # Data use in VFM
    f3x3 = jax.vmap(jax.vmap(fto3x3))(f2x2)
    f_neu_nodes = prep_data["f_neu"][train_load_steps_indices] 
    node_type = prep_data["node_type"]
    dNdX = prep_data["dNdX"]
    dA = prep_data["dA"]
    cells = prep_data["cells"]
    load_noise_std = prep_data["load_noise_std"]
    load_noise_std_steps = prep_data["load_noise_std_steps"][train_load_steps_indices] 

    true_mat_model = get_material(material_model_name)
    psi_true_func = lambda f: true_mat_model.psi(f)
    piola_true_func = lambda f: true_mat_model.P(f)

    I_all,_ = jax.vmap(jax.vmap(invariants_and_derivatives))(f3x3)
    # get all data inside prep_data
    dev, vol = jax.vmap(jax.vmap(transform_input_features))(I_all)
    dev_flat =  dev.reshape(-1, dev.shape[-1]) 
    vol_flat = vol.reshape(-1, vol.shape[-1])
    dev_z = farthest_point_sampling_with_fixed_point(dev_flat, n_ip, jnp.array([3.0, 3.0]))
    vol_z = farthest_point_sampling_with_fixed_point(vol_flat, n_ip, jnp.array([1.0]))
    plot_inducing_points(dev_z, vol_z, dev_flat, vol_flat, save_path)
    I_z = jnp.concat([dev_z, vol_z], axis = -1)

    # Setup random key
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    if is_fixed_reaction_force_noise :
        params = GPRawParams(
            # Lengthscales and signal variances (Normal(0, 1))
            raw_dev_ls=jax.random.normal(k1, (2,)),
            raw_dev_sig=jax.random.normal(k1, ()),
            
            # Inducing point means and variances
            raw_dev_z =jax.random.normal(k2, (n_ip, 2)),
            raw_dev_u_mean=jax.random.normal(k2, (n_ip,)),
            raw_dev_u_var=jax.random.normal(k2, (n_ip,)),

            raw_vol_ls=jax.random.normal(k3, (1,)),
            raw_vol_sig=jax.random.normal(k3, ()),

            raw_vol_z =jax.random.normal(k4, (n_ip,1)),        
            raw_vol_u_mean=jax.random.normal(k4, (n_ip,)),
            raw_vol_u_var=jax.random.normal(k4, (n_ip,)),

            # Scalar coefficients (randomizing around 0.0)
            raw_c01=jax.random.normal(k1, ()),
            raw_c02=jax.random.normal(k2, ()),
            raw_c10=jax.random.normal(k3, ()),
            raw_c11=jax.random.normal(k4, ()),
            raw_c20=jax.random.normal(k1, ()),
            raw_k=jax.random.normal(k2, ()),
            raw_q=jax.random.normal(k3, ()),
            raw_s=jax.random.normal(k4, ()),

            # Noise parameters
            log_sigma_free_x=jax.random.normal(k1, ()),
            log_sigma_free_y=jax.random.normal(k2, ()),
            log_sigma_fix_x=sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 0]),
            log_sigma_fix_y=sigma_fix_to_log_sigma_fix(load_noise_std_steps[:, 1])
            )
    else :
        params = GPRawParams(
            # Lengthscales and signal variances (Normal(0, 1))
            raw_dev_ls=jax.random.normal(k1, (2,)),
            raw_dev_sig=jax.random.normal(k1, ()),
            
            # Inducing point means and variances
            raw_dev_z =jax.random.normal(k2, (n_ip, 2)),
            raw_dev_u_mean=jax.random.normal(k2, (n_ip,)),
            raw_dev_u_var=jax.random.normal(k2, (n_ip,)),

            raw_vol_ls=jax.random.normal(k3, (1,)),
            raw_vol_sig=jax.random.normal(k3, ()),

            raw_vol_z =jax.random.normal(k4, (n_ip,1)),        
            raw_vol_u_mean=jax.random.normal(k4, (n_ip,)),
            raw_vol_u_var=jax.random.normal(k4, (n_ip,)),

            # Scalar coefficients (randomizing around 0.0)
            raw_c01=jax.random.normal(k1, ()),
            raw_c02=jax.random.normal(k2, ()),
            raw_c10=jax.random.normal(k3, ()),
            raw_c11=jax.random.normal(k4, ()),
            raw_c20=jax.random.normal(k1, ()),
            raw_k=jax.random.normal(k2, ()),
            raw_q=jax.random.normal(k3, ()),
            raw_s=jax.random.normal(k4, ()),

            # Noise parameters
            log_sigma_free_x=jax.random.normal(k1, ()),
            log_sigma_free_y=jax.random.normal(k2, ()),
            log_sigma_fix_x=jax.random.normal(k3, (load_noise_std_steps.shape[0])),
            log_sigma_fix_y=jax.random.normal(k4, (load_noise_std_steps.shape[0]))
        )
    
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    main_key = jr.PRNGKey(42)

    model = SparseHyperelasticityGP(params, I_z, min_dev, min_vol, max_dev, max_vol, beta = beta, is_include_prior_mean = is_include_prior_mean)



    # Deterministic training loop - just to get rough idea how prior knowledge would look like
    # if training_mode == "deterministic" or training_mode == "two-stage":
    #     deterministic_training_loop() 
    # if training_mode == "stochastic" or training_mode == "two-stage":
    #     stochastic_training_loop()  
    # use the mat params obtained from the deterministic training to construct prior knowledge and freeze them in stochastic training to let gp handle the uncertainty + correction part.


    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda p, k: total_stochastic_loss(p, model, f3x3, cells, cells.max() + 1, f_neu_nodes, node_type, dNdX, dA, k, number_of_mci_sampling),
        has_aux=True
    ))

    losses = []
    best_loss = float('inf')
    g_mean_history = []
    lengthscales_history = []
    log_scale_variance_history = []
    log_file_path = os.path.join(save_path, "optimization_log.txt")


    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(params)
    n_nodes = int(cells.max()) + 1
    # Open once and clear (or just let the loop handle it)
    with open(log_file_path, "w") as f:
        f.write(f"{train_load_steps_indices}, {material_model_name} \n Optimization Start\n" + "="*20 + "\n")
    # Lists to store history
    steps_history = []
    loss_components_hist = {"total_loss": [],"log_like": [], "kl": [], "phy": []}

    # Parameter history
    params_hist = {
        "dev_gp_sigma_scaling": [], "vol_gp_sigma_scaling": [],
        "dev_gp_lengthscales": [], "vol_gp_lengthscales": [], 
        "dev_u_mean": [], "dev_u_var": [], "vol_u_mean": [], "vol_u_var": [], "dev_z": [], "vol_z": [],
        "sigma_free_x": [], "sigma_free_y": [], "sigma_fix_x": [], "sigma_fix_y": [], "c20": [], "c02": [], "c11": [], "c10": [], "c01": [], "k": [], "q": []
    }



    total_step = n_iterations
    pbar = tqdm(range(total_step), desc="Training Sparse GP", unit="step")


    for step in pbar:
        keys = jr.split(main_key, 2)
        main_key = keys[0]
        subkey = keys[1]
        
        # JAX execution
        (loss, (log_like_loss, kl_loss, free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, phy_loss, phys_loss2)), grads = loss_and_grad(params, subkey)
        # grads = freeze_material_params(grads)
        if is_fixed_reaction_force_noise :
            grads = freeze_reaction_force_noise(grads)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = params
            with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
                jnp.save(f, best_params._asdict())
            with open(os.path.join(save_path, "I_z.npy"), "wb") as f:
                jnp.save(f, I_z)
            with open(os.path.join(save_path, "I_obs_all.npy"), "wb") as f:
                jnp.save(f, I_all)
        if step % 50 == 0:
            # Update the progress bar postfix with current metrics
            # This shows up right next to the time left
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "free_x": f"{free_x_log_likelihood:.4f}",
                "free_y": f"{free_y_log_likelihood:.4f}",
                "fix_x": f"{fix_x_log_likelihood:.4f}",
                "fix_y": f"{fix_y_log_likelihood:.4f}",
                "log_like": f"{log_like_loss:.4f}",
                "kl": f"{kl_loss:.4f}",
                "phy": f"{phy_loss:.4f}",
                "phy2": f"{phys_loss2:.4f}"
            })

            # --- Your existing logging logic ---
            # Note: Using pbar.write() instead of print() prevents 
            # the progress bar from breaking into multiple lines.
            log_message = (
                f"step {step:04d} | loss={loss:.6f} | "
                f"log_like={log_like_loss:.6f} | kl={kl_loss:.6f} | free_x={free_x_log_likelihood:.6f} | "
                f"free_y={free_y_log_likelihood:.6f} | fix_x={fix_x_log_likelihood:.6f} | "
                f"fix_y={fix_y_log_likelihood:.6f} | "
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
            params_hist["dev_z"].append(cur_params.dev_z)
            params_hist["vol_z"].append(cur_params.vol_z)
            params_hist["dev_u_mean"].append(cur_params.dev_u_mean)
            params_hist["dev_u_var"].append(cur_params.dev_u_var)
            params_hist["vol_u_mean"].append(cur_params.vol_u_mean)
            params_hist["vol_u_var"].append(cur_params.vol_u_var)

            params_hist["sigma_free_x"].append(cur_params.sigma_free_x)
            params_hist["sigma_free_y"].append(cur_params.sigma_free_y)
            params_hist["sigma_fix_x"].append(cur_params.sigma_fix_x)
            params_hist["sigma_fix_y"].append(cur_params.sigma_fix_y)

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
        if step % (total_step//5) == 0 and step != 0:
            
            plot_model = SparseHyperelasticityGP(best_params, I_z, min_dev, min_vol, max_dev, max_vol)
            plot_model.params = plot_model.load_params(best_params)
            plot_model.gpweight = plot_model.precompute_weights(best_params)
            plot_combined_validation(plot_model, true_mat_model, save_path, step)


    # Final Save
    with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
        jnp.save(f, best_params._asdict())
    with open(os.path.join(save_path, "I_z.npy"), "wb") as f:
        jnp.save(f, I_z)
    with open(os.path.join(save_path, "I_obs_all.npy"), "wb") as f:
        jnp.save(f, I_all)
    log_path = save_path + "/optimization_log.txt"# Change to your actual folder
    plot_loss_analysis(loss_components_hist, params_hist, steps_history, save_path)
    plot_parameters_hist(params_hist, steps_history, save_path)

    learned_gp = SparseHyperelasticityGP(best_params, I_z, min_dev, min_vol, max_dev, max_vol)
    plot_combined_validation(learned_gp, true_mat_model, save_path, step)

    print(f"{timestamp}_{training_config_str}")


