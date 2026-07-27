import os
import glob
import pickle
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

from core.datasetclass import DatasetFactory
from core.dataclass import GPRawParams
from core.model import SparseHyperelasticityGP
from core.material_models import get_material
from core.utils import *

# 1. Find the latest model directory
base_save_path = "extraction/extracted_models"
subdirs = [os.path.join(base_save_path, d) for d in os.listdir(base_save_path) if os.path.isdir(os.path.join(base_save_path, d))]
latest_model_dir = max(subdirs, key=os.path.getctime)
print(f"Using latest model directory: {latest_model_dir}")

# 2. Extract configuration from the directory name
folder_name = os.path.basename(latest_model_dir)
parts = folder_name.split('_')
material_model_name = parts[1]
disp_noise = float(parts[2])
load_noise = float(parts[3])
target_load_true_top = float(parts[4])
asym_factor = float(parts[5])
n_ip = int(parts[6])
beta = float(parts[7])

# 3. Load True Model
true_model = get_material(material_model_name)

# 4. Load Dataset
data_dir = "dataset/preprocessed/syn_f" 
prep_dataset_path = os.path.join(data_dir, f"{material_model_name}_{disp_noise}_{load_noise}_{target_load_true_top}_{asym_factor}.npz")
dataset = DatasetFactory.create("dataset/precomputed_vfm", data_path=prep_dataset_path)
prep_data = dataset.get_data()
F_train_full_2x2 = prep_data["F"]  
F_train_full = jax.vmap(jax.vmap(fto3x3))(F_train_full_2x2)

# 5. Load the learned GP model
best_params_dict = np.load(os.path.join(latest_model_dir, "best_params.npy"), allow_pickle=True).item()
best_params = GPRawParams(**best_params_dict)

I_z = jnp.load(os.path.join(latest_model_dir, "I_z.npy"))
dev_z = I_z[:, :2]
vol_z = I_z[:, 2:]
min_dev = jnp.min(dev_z, axis=0)
min_vol = jnp.min(vol_z, axis=0)
max_dev = jnp.max(dev_z, axis=0)
max_vol = jnp.max(vol_z, axis=0)

learned_gp = SparseHyperelasticityGP(best_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=beta)

# --- FIGURE 1: Energy Decomposition Validation ---
def plot_energy_decomposition_validation(learned_gp, true_model, save_path):
    print("Generating Energy Decomposition Validation Plot...")
    num_points = 50
    num_samples = 32
    gamma = jnp.linspace(0.0, 1.0, num_points, dtype=jnp.float64)
    
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

    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    psi_true_tot = jax.vmap(jax.vmap(true_model.psi))(F_all)
    psi_true_dev = jax.vmap(jax.vmap(true_model.psi_dev))(F_all)
    psi_true_vol = jax.vmap(jax.vmap(true_model.psi_vol))(F_all)

    dist_tot = jax.vmap(learned_gp.psi_dist)(F_all)
    psi_mean_tot, psi_std_tot = dist_tot.mean, jnp.sqrt(dist_tot.var)
    
    dist_dev = jax.vmap(learned_gp.dev_psi_dist)(F_all)
    psi_mean_dev, psi_std_dev = dist_dev.mean, jnp.sqrt(dist_dev.var)
    
    dist_vol = jax.vmap(learned_gp.vol_psi_dist)(F_all)
    psi_mean_vol, psi_std_vol = dist_vol.mean, jnp.sqrt(dist_vol.var)

    keys = jax.random.split(jax.random.PRNGKey(42), num_samples)
    
    psi_samples_dev = []
    psi_samples_vol = []
    psi_samples_tot = []
    for k in keys:
        fn = learned_gp.get_path_dev_vol_psi_fn(k)
        dev_s, vol_s = jax.vmap(jax.vmap(fn))(F_all)
        psi_samples_dev.append(dev_s)
        psi_samples_vol.append(vol_s)
        psi_samples_tot.append(dev_s + vol_s)
    psi_samples_dev = jnp.stack(psi_samples_dev, axis=0) 
    psi_samples_vol = jnp.stack(psi_samples_vol, axis=0)
    psi_samples_tot = jnp.stack(psi_samples_tot, axis=0)

    def calc_metrics(true, mean, std):
        rmse = jnp.sqrt(jnp.mean((true - mean)**2))
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        coverage = jnp.mean((true >= lower) & (true <= upper)) * 100
        return rmse, coverage
    
    fig, axes = plt.subplots(6, 3, figsize=(18, 24))
    fig.suptitle(f"Energy Decomposition Validation", fontsize=20, y=1.01)

    for i, name in enumerate(mode_names):
        configs = [
            (0, "Deviatoric", psi_true_dev[i], psi_mean_dev[i], psi_std_dev[i], psi_samples_dev[:, i, :]),
            (1, "Volumetric", psi_true_vol[i], psi_mean_vol[i], psi_std_vol[i], psi_samples_vol[:, i, :]),
            (2, "Total Energy", psi_true_tot[i], psi_mean_tot[i], psi_std_tot[i], psi_samples_tot[:, i, :])
        ]
        for col, col_name, true_val, mean_val, std_val, samples in configs:
            ax = axes[i, col]
            ax.plot(gamma, true_val, 'k--', lw=1.5, label="True", zorder=5)
            ax.plot(gamma, samples.T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
            ax.plot(gamma, mean_val, color="blue", lw=2, label="GP Mean", zorder=3)
            ax.fill_between(gamma, mean_val - 1.96*std_val, mean_val + 1.96*std_val, color="blue", alpha=0.2, zorder=2)
            
            rmse, coverage = calc_metrics(true_val, mean_val, std_val)
            
            ax.set_title(f"{name}: {col_name}\nRMSE: {rmse:.4f} | Cov: {coverage:.1f}%")
            ax.set_xlabel(r"$\gamma$")
            ax.grid(True, alpha=0.2)
            
            y_min, y_max = jnp.min(true_val), jnp.max(true_val)
            pad = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlim(0, 1)
            if i == 0 and col == 2:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "energy_decomposition.png"), bbox_inches='tight')
    plt.close()
    print("Done generating Energy Decomposition Plot.")

# --- FIGURE 2: R2 Plot on Training Data ---
def plot_training_r2(learned_gp, true_model, F_train_full, save_path):
    print("Generating Training Data R2 Plot...")
    num_steps = F_train_full.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Training Data Energy R2 (GP Mean vs Truth)", fontsize=16)
    
    colors = plt.cm.jet(np.linspace(0, 1, num_steps))
    
    all_true = []
    all_mean = []
    all_std = []
    
    for step in range(num_steps):
        F_step = F_train_full[step]
        true_psi = jax.vmap(true_model.psi)(F_step)
        dist = learned_gp.psi_dist(F_step)
        mean_psi = dist.mean
        std_psi = jnp.sqrt(dist.var)
        
        all_true.append(true_psi)
        all_mean.append(mean_psi)
        all_std.append(std_psi)
        
        ax.errorbar(true_psi, mean_psi, yerr=1.96*std_psi, fmt='o', color=colors[step], 
                    alpha=0.2, markersize=3, label=f"Step {step}" if step % 5 == 0 else "")
                    
    all_true = jnp.concatenate(all_true)
    all_mean = jnp.concatenate(all_mean)
    all_std = jnp.concatenate(all_std)
    
    r2 = r2_score(all_true, all_mean)
    rmse = jnp.sqrt(jnp.mean((all_true - all_mean)**2))
    lower = all_mean - 1.96 * all_std
    upper = all_mean + 1.96 * all_std
    coverage = jnp.mean((all_true >= lower) & (all_true <= upper)) * 100
    
    min_val = min(all_true.min(), all_mean.min())
    max_val = max(all_true.max(), all_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Parity")
    
    ax.text(0.05, 0.95, f"R2: {r2:.4f}\nRMSE: {rmse:.4f}\nCoverage: {coverage:.1f}%", 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
            
    ax.set_xlabel("True Total Energy")
    ax.set_ylabel("Predicted Total Energy")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_r2_energy.png"), bbox_inches='tight')
    plt.close()
    print("Done generating Training Data R2 Plot.")

if __name__ == "__main__":
    plot_energy_decomposition_validation(learned_gp, true_model, latest_model_dir)
    plot_training_r2(learned_gp, true_model, F_train_full, latest_model_dir)
    print("All done!")
