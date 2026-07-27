import jax
import jax.numpy as jnp

import optax
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import datetime

from core.datasetclass import DatasetFactory
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.utils import fto3x3
import seaborn as sns
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flax.linen as nn
from core.distillation import MaskedDense, FlaxMADE



def sample_maf(made_model, params, key, num_samples, num_params):
    z = jax.random.normal(key, (num_samples, num_params))
    x = jnp.zeros_like(z)
    for i in range(num_params):
        loc, log_scale = made_model.apply(params, x)
        scale = jnp.exp(log_scale)
        x = x.at[:, i].set(loc[:, i] + scale[:, i] * z[:, i])
    return x

def log_prob_maf(made_model, params, x):
    loc, log_scale = made_model.apply(params, x)
    scale = jnp.exp(log_scale)
    z = (x - loc) / scale
    log_p_z = jax.scipy.stats.norm.logpdf(z).sum(axis=-1)
    log_det = -log_scale.sum(axis=-1)
    return log_p_z + log_det

def generate_standard_modes(num_points=100, max_gamma=2.0):
    gamma = jnp.linspace(0.0, max_gamma, num_points)
    
    F_all = jnp.zeros((6, num_points, 3, 3))
    
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((num_points, 3, 3))
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material_model", type=str, default="ogden", choices=["ogden", "gmr"])
    parser.add_argument("--saved_model_dir", type=str, required=True, help="Path to GP saved model")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=16, help="MC samples for ELBO")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for data")
    parser.add_argument("--use_standard_modes", action="store_true", help="Use 6 standard deformation modes instead of dataset")
    args = parser.parse_args()
    
    # 1. Load Data
    if args.use_standard_modes:
        print("Using 6 standard deformation modes...")
        f3x3_flat = generate_standard_modes(num_points=100)
    else:
        data_dir = "dataset/precomputed_vfm" 
        prep_dataset_path = os.path.join(data_dir, args.dataset_name)
        dataset = DatasetFactory.create("dataset/precomputed_vfm", data_path=prep_dataset_path)
        prep_data = dataset.get_data()
        f2x2 = prep_data["F"]
        f3x3 = jax.vmap(jax.vmap(fto3x3))(f2x2)
        f3x3_flat = f3x3.reshape(-1, 3, 3)

    # Extract ugp material model name from directory
    model_folder_name = os.path.basename(os.path.normpath(args.saved_model_dir))
    parts = model_folder_name.split('_')
    ugp_model_name = parts[1] if len(parts) > 1 else "unknown"
    
    mode_name = "standard_modes" if args.use_standard_modes else args.dataset_name.replace(".npz", "")
    current_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    
    out_dir = f"distillation_models/{current_time}_{ugp_model_name}_{args.material_model}_{mode_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file_path = os.path.join(out_dir, "distillation_log.txt")
    with open(log_file_path, "w") as f:
        f.write("Epoch\tTime\tLoss\n")

    # 2. Load GP Model
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
    
    print("Evaluating GP over dataset...")
    def predict_chunk(f_chunk):
        dist = gp_model.psi_dist(f_chunk)
        return dist.mean, dist.var
    
    chunk_size = 5000
    means = []
    vars_ = []
    for i in range(0, f3x3_flat.shape[0], chunk_size):
        m, v = predict_chunk(f3x3_flat[i:i+chunk_size])
        means.append(m)
        vars_.append(v)
    
    mean_psi = jnp.concatenate(means)
    var_psi = jnp.concatenate(vars_)
    
    # For memory efficiency during training, use mini-batches of the dataset
    num_data = f3x3_flat.shape[0]
    
    # 3. Flow Setup
    if args.material_model == "ogden":
        num_params = 9 # 3 mu, 3 alpha, 3 vol
        param_names = [f"mu_{i+1}" for i in range(3)] + [f"alpha_{i+1}" for i in range(3)] + [f"D_{i+1}" for i in range(3)]
    elif args.material_model == "gmr":
        num_params = 12 # 9 dev, 3 vol
        param_names = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3"]
        
    flow_module = FlaxMADE(num_params=num_params, hidden_dims=(64, 64))
    rng_seq = jax.random.PRNGKey(42)
    k0, rng_seq = jax.random.split(rng_seq)
    dummy_x = jnp.zeros((1, num_params))
    flow_params = flow_module.init(k0, dummy_x)
    
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(flow_params)
    
    def get_log_likelihood(theta, f_batch, mean_batch, var_batch):
        if args.material_model == "ogden":
            mu = theta[:3]
            alpha = theta[3:6]
            vol = theta[6:9]
            mat = get_material("ogden", mu_params=mu, alpha_params=alpha, vol_params=vol, jit_P=False)
        else:
            dev = theta[:9] 
            vol = theta[9:12]
            mat = get_material("gmr", dev_params=dev, vol_params=vol, jit_P=False)
            
        pred = jax.vmap(mat.psi)(f_batch)
        ll = -0.5 * jnp.log(2 * jnp.pi * var_batch) - 0.5 * (pred - mean_batch)**2 / var_batch
        # Scale log likelihood to account for batch size vs full dataset size
        return jnp.sum(ll) * (num_data / f_batch.shape[0])
        
    @jax.jit
    def loss_fn(params, key, f_batch, mean_batch, var_batch):
        # sample from MAF
        theta_raw = sample_maf(flow_module, params, key, args.n_samples, num_params)
        log_q_raw = log_prob_maf(flow_module, params, theta_raw)
        
        # Transform to strictly positive parameters
        theta = jax.nn.softplus(theta_raw)
        log_det_jacobian = jax.nn.log_sigmoid(theta_raw).sum(axis=-1)
        log_q = log_q_raw - log_det_jacobian
        
        ll = jax.vmap(lambda t: get_log_likelihood(t, f_batch, mean_batch, var_batch))(theta)
        
        elbo = ll.mean(axis=0) - log_q.mean(axis=0) 
        return -elbo.mean()
        
    @jax.jit
    def update(params, opt_state, key, f_batch, mean_batch, var_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, f_batch, mean_batch, var_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
        
    print("Training Flow...")
    
    losses = []
    evolution_samples = {}
    
    for step in range(args.n_iterations + 1):
        # Random mini-batch
        k, rng_seq = jax.random.split(rng_seq)
        idx = jax.random.randint(k, (args.batch_size,), 0, num_data)
        f_batch = f3x3_flat[idx]
        m_batch = mean_psi[idx]
        v_batch = var_psi[idx]
        
        k2, rng_seq = jax.random.split(rng_seq)
        flow_params, opt_state, loss = update(flow_params, opt_state, k2, f_batch, m_batch, v_batch)
        
        if step % 100 == 0:
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Step {step}, Loss: {loss:.4f}")
            with open(log_file_path, "a") as f:
                f.write(f"{step}\t{current_time_str}\t{loss:.4f}\n")
            losses.append((step, float(loss)))
            
        if step > 0 and step % 5000 == 0:
            k3, rng_seq = jax.random.split(rng_seq)
            raw_samples = sample_maf(flow_module, flow_params, k3, 2000, num_params)
            evolution_samples[step] = np.array(jax.nn.softplus(raw_samples))
            
    print("Sampling final posterior...")
    if args.n_iterations not in evolution_samples:
        k3, rng_seq = jax.random.split(rng_seq)
        raw_samples = sample_maf(flow_module, flow_params, k3, 2000, num_params)
        evolution_samples[args.n_iterations] = np.array(jax.nn.softplus(raw_samples))
    
    physical_samples = evolution_samples[args.n_iterations]
    
    # Save parameters
    samples_path = os.path.join(out_dir, "flow_samples.npy")
    np.save(samples_path, physical_samples)
    
    # --- 1. Plot Loss Convergence ---
    steps, loss_vals = zip(*losses)
    plt.figure(figsize=(8, 5))
    
    # Use symlog since ELBO loss can be negative
    plt.plot(steps, loss_vals, color='black', linewidth=2)
    plt.yscale('symlog', linthresh=10.0) 
    plt.xlabel('Iteration')
    plt.ylabel('ELBO Loss (symlog scale)')
    plt.title('Flow Distillation Loss Convergence')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    loss_plot_path = os.path.join(out_dir, f"loss_convergence_{args.material_model}.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")
    
    # --- 2. Plot Parameter Evolution Separately ---
    for step_val, step_samples in evolution_samples.items():
        df = pd.DataFrame(np.array(step_samples), columns=param_names)
        
        num_params = len(param_names)
        cols = 4
        rows = (num_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten()
        
        means = df.mean()
        for i, col in enumerate(param_names):
            sns.histplot(df[col], ax=axes[i], color='blue', bins=30)
            mean_val = means[col]
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2)
            axes[i].set_title(f"{col}\nMean: {mean_val:.4f}", color='red', fontsize=12)
            
        for j in range(num_params, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        if step_val == args.n_iterations:
            plot_path = os.path.join(out_dir, f"distributions_{args.material_model}.png")
        else:
            plot_path = os.path.join(out_dir, f"distributions_{args.material_model}_step{step_val}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Done! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
