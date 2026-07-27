import os
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm
from core.plotter import plot_loss_analysis, plot_parameters_hist, plot_combined_validation, plot_energy_decomposition_validation, plot_training_r2
from core.model import SparseHyperelasticityGP

class HyperelasticGPTrainer:
    def __init__(self, model: SparseHyperelasticityGP, initial_params, loss_fn, opt_state, optimizer, save_path, true_mat_model, I_z, I_all, min_dev, min_vol, max_dev, max_vol, freeze_fn=None):
        self.model = model
        self.params = initial_params
        # JIT compile the loss and gradient function
        self.loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        self.opt_state = opt_state
        self.optimizer = optimizer
        self.save_path = save_path
        self.true_mat_model = true_mat_model
        
        self.I_z = I_z
        self.I_all = I_all
        self.min_dev = min_dev
        self.min_vol = min_vol
        self.max_dev = max_dev
        self.max_vol = max_vol
        self.freeze_fn = freeze_fn

        self.log_file_path = os.path.join(save_path, "optimization_log.txt")
        self.loss_components_hist = {"total_loss": [],"log_like": [], "kl": [], "phy": []}
        self.params_hist = {
            "dev_gp_sigma_scaling": [], "vol_gp_sigma_scaling": [],
            "dev_gp_lengthscales": [], "vol_gp_lengthscales": [], 
            "dev_u_mean": [], "dev_u_var": [], "vol_u_mean": [], "vol_u_var": [], "dev_z": [], "vol_z": [],
            "sigma_free_x": [], "sigma_free_y": [], "sigma_fix_x": [], "sigma_fix_y": [],
            "vol_kappa": []
        }
        self.steps_history = []
        self.best_loss = float('inf')
        self.best_params = initial_params
        
    def train(self, n_iterations, main_key, log_info_str):
        with open(self.log_file_path, "w") as f:
            f.write(f"{log_info_str} \n Optimization Start\n" + "="*20 + "\n")

        pbar = tqdm(range(n_iterations), desc="Training Sparse GP", unit="step")
        
        for step in pbar:
            keys = jr.split(main_key, 2)
            main_key = keys[0]
            subkey = keys[1]
            
            # JAX execution
            (loss, aux), grads = self.loss_and_grad(self.params, subkey)
            log_like_loss, kl_loss, free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, phy_loss, phys_loss2 = aux

            if self.freeze_fn:
                grads = self.freeze_fn(grads)
            
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)
            
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = self.params
                with open(os.path.join(self.save_path, "best_params.npy"), "wb") as f:
                    jnp.save(f, self.best_params._asdict())
                with open(os.path.join(self.save_path, "I_z.npy"), "wb") as f:
                    jnp.save(f, self.I_z)
                with open(os.path.join(self.save_path, "I_obs_all.npy"), "wb") as f:
                    jnp.save(f, self.I_all)

            if step % 50 == 0:
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

                log_message = (
                    f"step {step:04d} | loss={loss:.6f} | "
                    f"log_like={log_like_loss:.6f} | kl={kl_loss:.6f} | free_x={free_x_log_likelihood:.6f} | "
                    f"free_y={free_y_log_likelihood:.6f} | fix_x={fix_x_log_likelihood:.6f} | "
                    f"fix_y={fix_y_log_likelihood:.6f} | "
                    f"phy={phy_loss:.6f} | phy2 ={phys_loss2:.6f}\n"
                )
                cur_params = self.model.load_params(self.params)
                clean_params = jax.tree_util.tree_map(
                    lambda x: x.tolist() if hasattr(x, 'tolist') else x, 
                    cur_params
                )
                log_message += f"params: {clean_params}\n"
                log_message += "-"*50 + "\n"

                self.steps_history.append(step)
                self.loss_components_hist["total_loss"].append(float(loss))
                self.loss_components_hist["log_like"].append(float(log_like_loss))
                self.loss_components_hist["kl"].append(float(kl_loss))
                self.loss_components_hist["phy"].append(float(phy_loss))
                
                self.params_hist["dev_gp_sigma_scaling"].append(cur_params.dev_sig)
                self.params_hist["vol_gp_sigma_scaling"].append(cur_params.vol_sig)
                self.params_hist["dev_gp_lengthscales"].append(cur_params.dev_ls)
                self.params_hist["vol_gp_lengthscales"].append(cur_params.vol_ls)
                self.params_hist["dev_z"].append(cur_params.dev_z)
                self.params_hist["vol_z"].append(cur_params.vol_z)
                self.params_hist["dev_u_mean"].append(cur_params.dev_u_mean)
                self.params_hist["dev_u_var"].append(cur_params.dev_u_var)
                self.params_hist["vol_u_mean"].append(cur_params.vol_u_mean)
                self.params_hist["vol_u_var"].append(cur_params.vol_u_var)
                self.params_hist["sigma_free_x"].append(cur_params.sigma_free_x)
                self.params_hist["sigma_free_y"].append(cur_params.sigma_free_y)
                self.params_hist["sigma_fix_x"].append(cur_params.sigma_fix_x)
                self.params_hist["sigma_fix_y"].append(cur_params.sigma_fix_y)
                self.params_hist["vol_kappa"].append(cur_params.vol_kappa)

                with open(self.log_file_path, "a") as f:
                    f.write(log_message)
                    
            if step % max(1, (n_iterations//5)) == 0 and step != 0:
                plot_model = SparseHyperelasticityGP(self.best_params, self.I_z, self.min_dev, self.min_vol, self.max_dev, self.max_vol)
                plot_combined_validation(plot_model, self.true_mat_model, self.save_path, step)

        # Final plots and validation
        plot_loss_analysis(self.loss_components_hist, self.params_hist, self.steps_history, self.save_path)
        plot_parameters_hist(self.params_hist, self.steps_history, self.save_path)
        learned_gp = SparseHyperelasticityGP(self.best_params, self.I_z, self.min_dev, self.min_vol, self.max_dev, self.max_vol, beta=self.model.beta)
        plot_combined_validation(learned_gp, self.true_mat_model, self.save_path, step)
        
        # New Energy Validation Plots
        plot_energy_decomposition_validation(learned_gp, self.true_mat_model, self.save_path)
        
        # Note: If F_train_full is available, it can be passed. 
        # But wait, trainer doesn't have F_train_full!
        return self.best_params
