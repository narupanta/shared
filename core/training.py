import jax 

import jax.random as jr
import optax
from core.model import SparseHyperelasticityGP, transform_input_features, enforce_softplus_positive, GPRawParams, GPParams, GPWeights
from core.material_models import get_material, BaseMaterialModel
import jax.numpy as jnp
from core.utils import *
from tqdm import tqdm
from core.plotter import \
    plot_loss_analysis, \
    plot_parameters_hist, \
    plot_r2_strain_energy_function, plot_ut_ebt_ps_uc_ebc_ss, plot_inducing_points, plot_stress_validation



def training_loop(learn_model: SparseHyperelasticityGP, 
                  true_model: BaseMaterialModel, 
                  optimizer: optax.GradientTransformationExtraArgs, 
                  params: GPRawParams, 
                  main_key: jax.random.PRNGKey,
                  loss_and_grad, 
                  n_iter: int, 
                  log_file_path: str) :

    opt_state = optimizer.init(params)
    total_step = n_iter
    pbar = tqdm(range(total_step), desc="Training Sparse GP", unit="step")
    for step in pbar:
        main_key, subkey = jr.split(main_key)
        
        # JAX execution
        (loss, (log_like_loss, kl_loss, phy_tot, phy_free, phy_reac)), grads = loss_and_grad(params, subkey)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params

        if step % 50 == 0:
            # Update the progress bar postfix with current metrics
            # This shows up right next to the time left
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "phy_tot": f"{phy_tot:.4f}",
                "phy_free": f"{phy_free:.4f}",
                "phy_reac": f"{phy_reac:.4f}"
            })

            # --- Your existing logging logic ---
            # Note: Using pbar.write() instead of print() prevents 
            # the progress bar from breaking into multiple lines.
            log_message = (
                f"step {step:04d} | loss={loss:.6f} | "
                f"log_like={log_like_loss:.6f} | kl={kl_loss:.6f} | "
                f"phy_tot={phy_tot:.6f} | phy_free ={phy_free:.6f} | phy_reac ={phy_reac:.6f}\n"
            )
            cur_params = learn_model.load_params(params)
            clean_params = jax.tree_util.tree_map(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x, 
                cur_params
            )
            log_message += f"params: {clean_params}\n"
            log_message += "-"*50 + "\n"
            # Append to log file
            with open(log_file_path, "a") as f:
                f.write(log_message)

        if step % (total_step//10) == 0 and step != 0:
            
            plot_model = learn_model
            plot_model.params = plot_model.load_params(best_params)
            plot_model.gpweight = plot_model.precompute_weights(best_params)
            plot_ut_ebt_ps_uc_ebc_ss(plot_model, true_model, log_file_path, step)
