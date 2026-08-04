import os
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm
from core.plotter import plot_loss_analysis, plot_parameters_hist, plot_combined_validation, plot_energy_decomposition_validation, plot_training_r2
from core.model import SparseHyperelasticityGP

# Enforce mandatory 64-bit precision standard for hyperelastic computations
jax.config.update("jax_enable_x64", True)


class HyperelasticGPTrainer:
    def __init__(self, model: SparseHyperelasticityGP, initial_params, loss_fn, opt_state, optimizer, save_path, true_mat_model, I_z, I_all, min_dev, min_vol, max_dev, max_vol, freeze_fn=None):
        self.model = model
        self.params = initial_params
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

        # JIT compile the single-step loss and gradient function (legacy compatibility)
        self.loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        
        # JIT compile fused block optimization loop via jax.lax.scan for GPU efficiency
        def step_fn(state, subkey):
            params_curr, opt_state_curr = state
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_curr, subkey)
            if freeze_fn:
                grads = freeze_fn(grads)
            updates, opt_state_new = optimizer.update(grads, opt_state_curr)
            params_new = optax.apply_updates(params_curr, updates)
            return (params_new, opt_state_new), (loss, aux)

        @jax.jit
        def train_block(params_in, opt_state_in, keys_in):
            (params_out, opt_state_out), (losses, aux_out) = jax.lax.scan(step_fn, (params_in, opt_state_in), keys_in)
            return params_out, opt_state_out, losses, aux_out

        self.train_block = train_block

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

    def _record_metrics(self, step, loss, aux, params):
        log_like_loss, kl_loss, free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, phy_loss, phys_loss2 = aux
        
        log_message = (
            f"step {step:04d} | loss={loss:.6f} | "
            f"log_like={log_like_loss:.6f} | kl={kl_loss:.6f} | free_x={free_x_log_likelihood:.6f} | "
            f"free_y={free_y_log_likelihood:.6f} | fix_x={fix_x_log_likelihood:.6f} | "
            f"fix_y={fix_y_log_likelihood:.6f} | "
            f"phy={phy_loss:.6f} | phy2 ={phys_loss2:.6f}\n"
        )
        cur_params = self.model.load_params(params)
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
        
        return {
            "loss": f"{loss:.4f}",
            "free_x": f"{free_x_log_likelihood:.4f}",
            "free_y": f"{free_y_log_likelihood:.4f}",
            "fix_x": f"{fix_x_log_likelihood:.4f}",
            "fix_y": f"{fix_y_log_likelihood:.4f}",
            "log_like": f"{log_like_loss:.4f}",
            "kl": f"{kl_loss:.4f}",
            "phy": f"{phy_loss:.4f}",
            "phy2": f"{phys_loss2:.4f}"
        }
        
    def train(self, n_iterations, main_key, log_info_str, block_size: int = 50):
        with open(self.log_file_path, "w") as f:
            f.write(f"{log_info_str} \n Optimization Start\n" + "="*20 + "\n")

        # Decoupled checkpointing: save static dataset arrays once before optimization loop
        with open(os.path.join(self.save_path, "I_z.npy"), "wb") as f:
            jnp.save(f, self.I_z)
        with open(os.path.join(self.save_path, "I_obs_all.npy"), "wb") as f:
            jnp.save(f, self.I_all)
        with open(os.path.join(self.save_path, "best_params.npy"), "wb") as f:
            jnp.save(f, self.best_params._asdict())

        # Determine step blocks for jax.lax.scan execution
        block_size = min(max(1, block_size), max(1, n_iterations))
        n_blocks = (n_iterations + block_size - 1) // block_size

        pbar = tqdm(range(n_blocks), desc="Training Sparse GP (JIT Blocks)", unit="block")
        
        step_idx = 0
        for _ in pbar:
            cur_block_size = min(block_size, n_iterations - step_idx)
            keys = jr.split(main_key, cur_block_size + 1)
            main_key = keys[0]
            block_keys = keys[1:]
            
            # Execute entire block inside JAX XLA compiled graph without host-device sync
            self.params, self.opt_state, losses, aux_out = self.train_block(self.params, self.opt_state, block_keys)
            
            step_idx += cur_block_size
            
            # Extract final metrics from the block
            loss = float(losses[-1])
            aux_step = tuple(a[-1] for a in aux_out)
            
            # Decoupled parameter disk I/O: save only when best loss is broken at block boundary
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = self.params
                with open(os.path.join(self.save_path, "best_params.npy"), "wb") as f:
                    jnp.save(f, self.best_params._asdict())

            # Record metrics and update progress bar (matching legacy step % 50 == 0 behavior)
            postfix = self._record_metrics(step_idx, loss, aux_step, self.params)
            pbar.set_postfix(postfix)
                    
            # Trigger progress visualizations during training milestones
            if step_idx % max(1, (n_iterations // 5)) == 0 and step_idx != 0:
                plot_model = SparseHyperelasticityGP(self.best_params, self.I_z, self.min_dev, self.min_vol, self.max_dev, self.max_vol)
                plot_combined_validation(plot_model, self.true_mat_model, self.save_path, step_idx)

        # Final plots and validation
        plot_loss_analysis(self.loss_components_hist, self.params_hist, self.steps_history, self.save_path)
        plot_parameters_hist(self.params_hist, self.steps_history, self.save_path)
        learned_gp = SparseHyperelasticityGP(self.best_params, self.I_z, self.min_dev, self.min_vol, self.max_dev, self.max_vol, beta=self.model.beta)
        plot_combined_validation(learned_gp, self.true_mat_model, self.save_path, step_idx)
        
        # New Energy Validation Plots
        plot_energy_decomposition_validation(learned_gp, self.true_mat_model, self.save_path)
        
        return self.best_params
