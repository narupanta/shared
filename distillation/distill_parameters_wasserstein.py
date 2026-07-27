import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import optax
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime
import pickle

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
import seaborn as sns
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flax.linen as nn
from core.distillation import MaskedDense, FlaxMADE, DeepFlaxIAF, Critic



def generate_standard_modes(num_points=32, max_gamma=1.0):
    gamma = jnp.linspace(0.0, max_gamma, num_points, dtype=jnp.float64)
    
    F_all = jnp.zeros((6, num_points, 3, 3), dtype=jnp.float64)
    
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((num_points, 3, 3), dtype=jnp.float64)
        arr = arr.at[:, 0, 0].set(f11)
        arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33)
        arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = F_all.at[0].set(set_F(1 + gamma, 1.0, 1.0))            
    F_all = F_all.at[1].set(set_F(1 + gamma, 1 + gamma, 1.0))    
    F_all = F_all.at[2].set(set_F(1 + gamma, 1/(1 + gamma), 1.0)) 
    F_all = F_all.at[3].set(set_F(1/(1 + gamma), 1.0, 1.0))       
    F_all = F_all.at[4].set(set_F(1/(1 + gamma), 1/(1 + gamma), 1.0)) 
    F_all = F_all.at[5].set(set_F(1.0, 1.0, 1.0, f12=gamma))      
    
    return F_all.reshape(-1, 3, 3)

def sample_dataset_deformations(saved_model_dir, num_points=128):
    import os
    import numpy as np
    import jax
    from core.utils import fto3x3
    import jax.numpy as jnp

    model_folder_name = os.path.basename(os.path.normpath(saved_model_dir))
    parts = model_folder_name.split('_')
    
    # Try to parse dataset path based on standard saved model folder structure
    # e.g. 20260723T152354_nh_0.0001_0.01_1.5_0.95_5_80.0_1
    try:
        ugp_model_name = parts[1]
        disp_noise = parts[2]
        load_noise = parts[3]
        target_load = parts[4]
        asym_factor = parts[5]
    except IndexError:
        raise ValueError(f"Could not parse parameters from model directory name: {model_folder_name}")

    data_dir = "dataset/preprocessed/syn_f"
    prep_dataset_path = os.path.join(data_dir, f"{ugp_model_name}_{disp_noise}_{load_noise}_{target_load}_{asym_factor}.npz")
    
    if not os.path.exists(prep_dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {prep_dataset_path}")

    prep_data = np.load(prep_dataset_path)
    F_train_full_2x2 = prep_data["F"]
    
    F_flat_2x2 = F_train_full_2x2.reshape(-1, 2, 2)
    
    if len(F_flat_2x2) < num_points:
        print(f"Warning: Dataset only has {len(F_flat_2x2)} elements. Sampling with replacement.")
        indices = np.random.choice(len(F_flat_2x2), size=num_points, replace=True)
    else:
        indices = np.random.choice(len(F_flat_2x2), size=num_points, replace=False)
        
    F_sampled_2x2 = F_flat_2x2[indices]
    
    F_sampled_3x3 = jax.vmap(fto3x3)(jnp.array(F_sampled_2x2, dtype=jnp.float64))
    
    print(f"Sampled {num_points} exactly observed deformation gradients from {prep_dataset_path}.")
    return F_sampled_3x3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material_model", type=str, default="ogden", choices=["ogden", "gmr", "isihara"])
    parser.add_argument("--saved_model_dir", type=str, required=True, help="Path to GP saved model")
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--lr_flow", type=float, default=5e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512, help="Number of functional samples per step")
    parser.add_argument("--n_critic", type=int, default=10, help="Critic updates per flow update")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty coefficient")
    parser.add_argument('--max_gamma', type=float, default=8.0)
    parser.add_argument("--resume_dir", type=str, default="", help="Path to resume checkpoint from")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--sample_mode", type=str, default="standard", choices=["standard", "dataset_f"], help="Mode for functional samples.")
    parser.add_argument("--num_interp_samples", type=int, default=128, help="Number of points to sample inside interpolation regime")
    args = parser.parse_args()
    
    if args.sample_mode == "standard":
        print(f"Using 6 standard deformation modes with gamma in [0, {args.max_gamma}]...")
        f3x3_flat = generate_standard_modes(num_points=32, max_gamma=args.max_gamma)
    elif args.sample_mode == "dataset_f":
        print(f"Sampling {args.num_interp_samples} actual F points from the training dataset...")
        f3x3_flat = sample_dataset_deformations(args.saved_model_dir, num_points=args.num_interp_samples)
    
    model_folder_name = os.path.basename(os.path.normpath(args.saved_model_dir))
    parts = model_folder_name.split('_')
    ugp_model_name = parts[1] if len(parts) > 1 else "unknown"
    
    current_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = f"distillation/distilled_models/{current_time}_{ugp_model_name}_{args.material_model}_wasserstein"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file_path = os.path.join(out_dir, "distillation_log.txt")
    with open(log_file_path, "w") as f:
        f.write("Step\tTimestamp\tW-Dist\tFlowLoss\tCriticLoss\n")

    # Load pre-trained GP model parameters
    best_params_dict = np.load(os.path.join(args.saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(args.saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    
    gp_model = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)
    
    print("Evaluating full GP posterior distribution over deformation points...")
    mean_psi = gp_model.psi_gp_mean(f3x3_flat)
    cov_psi = gp_model.psi_joint_cov(f3x3_flat)
    
    scale_factor = jnp.std(mean_psi) + jnp.mean(jnp.sqrt(jnp.abs(jnp.diag(cov_psi)))) + 1.0
    
    # Setup Flow
    if args.material_model == "ogden":
        num_params = 9 # 3 mu, 3 alpha, 3 vol
        param_names = [f"mu_{i+1}" for i in range(3)] + [f"alpha_{i+1}" for i in range(3)] + [f"D_{i+1}" for i in range(3)]
    elif args.material_model == "gmr":
        num_params = 12 # 9 dev, 3 vol
        param_names = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3"]
    elif args.material_model == "isihara":
        num_params = 4
        param_names = ["C10", "C01", "C20", "D1"]
        
    flow_module = DeepFlaxIAF(num_params=num_params)
    critic_module = Critic()
    
    rng_seq = jax.random.PRNGKey(42)
    k0, rng_seq = jax.random.split(rng_seq)
    
    flow_params = flow_module.init(k0, k0, args.batch_size)
    
    k1, rng_seq = jax.random.split(rng_seq)
    dummy_psi = jnp.zeros((1, f3x3_flat.shape[0]))
    critic_params = critic_module.init(k1, dummy_psi)
    
    scheduler_flow = optax.exponential_decay(init_value=args.lr_flow, transition_steps=1, decay_rate=0.9999)
    optimizer_flow = optax.chain(optax.clip_by_global_norm(1.0), optax.rmsprop(scheduler_flow))
    optimizer_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr_critic, b1=0.0, b2=0.9))
    opt_state_flow = optimizer_flow.init(flow_params)
    opt_state_critic = optimizer_critic.init(critic_params)
    
    @jax.jit
    def update_critic(c_params, f_params, o_state_c, k_gp, k_flow, k_interp):
        gp_samples = jax.random.multivariate_normal(k_gp, mean_psi, cov_psi, shape=(args.batch_size,), method='svd')
        
        theta_raw = flow_module.apply(f_params, k_flow, args.batch_size)
        theta = jnp.clip(jax.nn.softplus(theta_raw), a_max=10.0)
        
        def get_model_psi(t):
            if args.material_model == "ogden":
                mu = t[:3]
                alpha = t[3:6]
                vol = t[6:9]
                mat = get_material("ogden", mu_params=mu, alpha_params=alpha, vol_params=vol, jit_P=False)
            elif args.material_model == "gmr":
                dev = t[:9] 
                vol = t[9:12]
                mat = get_material("gmr", dev_params=dev, vol_params=vol, jit_P=False)
            elif args.material_model == "isihara":
                mat = get_material("isihara", c10=t[0], c01=t[1], c20=t[2], d1=t[3], jit_P=False)
            return jax.vmap(mat.psi)(f3x3_flat)
            
        model_samples = jax.vmap(get_model_psi)(theta)
        
        def critic_loss_fn(p):
            c_gp = critic_module.apply(p, gp_samples / scale_factor)
            c_model = critic_module.apply(p, model_samples / scale_factor)
            w_dist = c_gp.mean() - c_model.mean()
            
            epsilon = jax.random.uniform(k_interp, shape=(args.batch_size, 1))
            interp = epsilon * gp_samples + (1 - epsilon) * model_samples
            
            def single_critic(x):
                return critic_module.apply(p, x / scale_factor)[0]
                
            grads = jax.vmap(jax.grad(single_critic))(interp)
            grad_norms = jnp.sqrt(jnp.sum(grads**2, axis=1) + 1e-12)
            grad_penalty = jnp.mean((grad_norms - 1.0)**2)
            
            loss = -w_dist + args.lambda_gp * grad_penalty
            return loss, w_dist
            
        (loss, w_dist), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(c_params)
        updates, o_state_c = optimizer_critic.update(grads, o_state_c, c_params)
        c_params = optax.apply_updates(c_params, updates)
        
        return c_params, o_state_c, loss, w_dist

    @jax.jit
    def update_flow(f_params, c_params, o_state_f, k_flow):
        def flow_loss_fn(p):
            theta_raw = flow_module.apply(p, k_flow, args.batch_size)
            theta = jnp.clip(jax.nn.softplus(theta_raw), a_max=10.0)
            
            def get_model_psi(t):
                if args.material_model == "ogden":
                    mu = t[:3]
                    alpha = t[3:6]
                    vol = t[6:9]
                    mat = get_material("ogden", mu_params=mu, alpha_params=alpha, vol_params=vol, jit_P=False)
                elif args.material_model == "gmr":
                    dev = t[:9] 
                    vol = t[9:12]
                    mat = get_material("gmr", dev_params=dev, vol_params=vol, jit_P=False)
                elif args.material_model == "isihara":
                    mat = get_material("isihara", c10=t[0], c01=t[1], c20=t[2], d1=t[3], jit_P=False)
                return jax.vmap(mat.psi)(f3x3_flat)
                
            model_samples = jax.vmap(get_model_psi)(theta)
            c_model = critic_module.apply(c_params, model_samples / scale_factor)
            return -c_model.mean()
            
        loss, grads = jax.value_and_grad(flow_loss_fn)(f_params)
        updates, o_state_f = optimizer_flow.update(grads, o_state_f, f_params)
        f_params = optax.apply_updates(f_params, updates)
        
        return f_params, o_state_f, loss

    print("Training Flow via Wasserstein distance...")
    
    w_dists = []
    f_losses = []
    c_losses = []
    evolution_samples = {}
    
    start_step = 0
    if args.resume_dir:
        ckpt_path = os.path.join(args.resume_dir, "checkpoint.pkl")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
                flow_params = ckpt['flow_params']
                critic_params = ckpt['critic_params']
                opt_state_flow = ckpt['opt_state_flow']
                opt_state_critic = ckpt['opt_state_critic']
                start_step = ckpt['step']
            print(f"Resumed from checkpoint at step {start_step}")
        else:
            print("Checkpoint not found, starting from scratch.")
    
    for step in range(start_step, args.n_iterations + 1):
        # 1. Train Critic
        k_gp, rng_seq = jax.random.split(rng_seq)
        for _ in range(args.n_critic):
            k_flow, k_interp, rng_seq = jax.random.split(rng_seq, 3)
            critic_params, opt_state_critic, c_loss, w_dist = update_critic(
                critic_params, flow_params, opt_state_critic, k_gp, k_flow, k_interp
            )
            
        # 2. Train Flow
        k_flow, rng_seq = jax.random.split(rng_seq)
        flow_params, opt_state_flow, f_loss = update_flow(
            flow_params, critic_params, opt_state_flow, k_flow
        )
        
        if step % 100 == 0:
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Step {step}, W-Dist: {w_dist:.4f}, Flow Loss: {f_loss:.4f}, Critic Loss: {c_loss:.4f}")
            with open(log_file_path, "a") as f:
                f.write(f"{step}\t{current_time_str}\t{float(w_dist):.4f}\t{float(f_loss):.4f}\t{float(c_loss):.4f}\n")
            w_dists.append((step, float(w_dist)))
            f_losses.append((step, float(f_loss)))
            c_losses.append((step, float(c_loss)))
            
        if step > 0 and step % 1000 == 0:
            k3, rng_seq = jax.random.split(rng_seq)
            raw_samples = flow_module.apply(flow_params, k3, 2000)
            evolution_samples[step] = np.array(jnp.clip(jax.nn.softplus(raw_samples), a_max=10.0))
            
        if step > 0 and step % args.save_interval == 0:
            with open(os.path.join(out_dir, "checkpoint.pkl"), "wb") as f:
                pickle.dump({
                    'flow_params': flow_params,
                    'critic_params': critic_params,
                    'opt_state_flow': opt_state_flow,
                    'opt_state_critic': opt_state_critic,
                    'step': step
                }, f)
            
    print("Sampling final posterior...")
    if args.n_iterations not in evolution_samples:
        k3, rng_seq = jax.random.split(rng_seq)
        raw_samples = flow_module.apply(flow_params, k3, 2000)
        evolution_samples[args.n_iterations] = np.array(jnp.clip(jax.nn.softplus(raw_samples), a_max=10.0))
    
    physical_samples = evolution_samples[args.n_iterations]
    
    samples_path = os.path.join(out_dir, "flow_samples.npy")
    np.save(samples_path, physical_samples)
    
    # Plots
    steps, w_vals = zip(*w_dists)
    _, c_vals = zip(*c_losses)
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, w_vals, color='blue', linewidth=2, label="Wasserstein Distance")
    plt.plot(steps, c_vals, color='red', linewidth=2, linestyle='--', label="Lipschitz Loss")
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title('Wasserstein & Lipschitz Losses')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    loss_plot_path = os.path.join(out_dir, f"loss_w1.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    for step_val, step_samples in evolution_samples.items():
        df = pd.DataFrame(np.array(step_samples), columns=param_names)
        num_params_len = len(param_names)
        cols = 4
        rows = (num_params_len + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten()
        
        means = df.mean()
        for i, col in enumerate(param_names):
            sns.histplot(df[col], ax=axes[i], color='blue', bins=30)
            mean_val = means[col]
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2)
            axes[i].set_title(f"{col}\nMean: {mean_val:.4f}", color='red', fontsize=12)
            
        for j in range(num_params_len, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        if step_val == args.n_iterations:
            plot_path = os.path.join(out_dir, f"distributions_{args.material_model}.png")
        else:
            plot_path = os.path.join(out_dir, f"distributions_{args.material_model}_step{step_val}.png")
        plt.savefig(plot_path)
        plt.close()

if __name__ == "__main__":
    main()
