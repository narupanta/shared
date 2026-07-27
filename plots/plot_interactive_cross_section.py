import argparse
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import seaborn as sns
from scipy.stats import norm

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material

def generate_standard_modes(num_points=50, max_gamma=1.0):
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
    return F_all, gamma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, default="extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1")
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="gmr", choices=["ogden", "gmr", "isihara"])
    args = parser.parse_args()
    
    print("Loading models and precomputing distributions... (this may take a few seconds)")
    true_model = get_material("isihara", c10=0.5, c01=1.0, c20=1.0, d1=1.5, jit_P=False)
    
    best_params_dict = np.load(os.path.join(args.saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(args.saved_model_dir, "I_z.npy"))
    
    dev_z, vol_z = I_z[:, :2], I_z[:, 2:]
    min_dev, min_vol = jnp.min(dev_z, axis=0), jnp.min(vol_z, axis=0)
    max_dev, max_vol = jnp.max(dev_z, axis=0), jnp.max(vol_z, axis=0)
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)
    
    samples = np.load(os.path.join(args.distilled_dir, "flow_samples.npy"))
    
    # Precompute for 50 gamma steps
    F_all, gamma_vals = generate_standard_modes(num_points=50, max_gamma=1.0)
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    psi_true = jax.vmap(true_model.psi)(F_all) # (6, 50)
    
    psi_dist_mean = np.array([learned_gp.psi_dist(F_all[mode]).mean for mode in range(6)]) # (6, 50)
    psi_dist_var = np.array([learned_gp.psi_dist(F_all[mode]).var for mode in range(6)]) # (6, 50)
    
    def get_distilled_energy(theta, F_chunk):
        if args.material_model == "ogden":
            mat = get_material("ogden", mu_params=theta[:3], alpha_params=theta[3:6], vol_params=theta[6:9], jit_P=False)
        elif args.material_model == "gmr":
            mat = get_material("gmr", dev_params=theta[:9], vol_params=theta[9:12], jit_P=False)
        elif args.material_model == "isihara":
            mat = get_material("isihara", c10=theta[0], c01=theta[1], c20=theta[2], d1=theta[3], jit_P=False)
        return jax.vmap(mat.psi)(F_chunk)

    # Precompute distilled energies for all samples across all gammas
    # This might take a few seconds
    distilled_energies = []
    for m in range(6):
        # s_psi will be (num_samples, 50)
        s_psi = jax.vmap(lambda t: get_distilled_energy(t, F_all[m]))(samples)
        distilled_energies.append(np.array(s_psi))
        
    print("Precomputation finished! Launching interactive plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(bottom=0.2) # Make room for slider
    axes = axes.flatten()
    
    # Create Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    gamma_slider = Slider(
        ax=ax_slider,
        label='Gamma ($\gamma$)',
        valmin=0.0,
        valmax=1.0,
        valinit=0.5,
        valstep=gamma_vals
    )

    def update(val):
        # Find closest gamma index
        idx = np.argmin(np.abs(gamma_vals - gamma_slider.val))
        g_val = gamma_vals[idx]
        
        fig.suptitle(f"Energy Distribution Cross-Section at $\\gamma={g_val:.2f}$", fontsize=18)
        
        for i, name in enumerate(mode_names):
            ax = axes[i]
            ax.clear()
            
            # Ground Truth (Black Dash Line)
            true_val = psi_true[i, idx]
            ax.axvline(true_val, color='black', linestyle='--', lw=2, label="True Physics")
            
            # GP Distribution (Blue)
            mu_gp = psi_dist_mean[i, idx]
            std_gp = np.sqrt(psi_dist_var[i, idx])
            
            # If standard deviation is extremely small (e.g. at gamma=0), add a tiny epsilon to prevent flatlining
            if std_gp < 1e-6:
                std_gp = 1e-6
                
            x = np.linspace(mu_gp - 4*std_gp, mu_gp + 4*std_gp, 100)
            p = norm.pdf(x, mu_gp, std_gp)
            ax.plot(x, p, color='blue', lw=2, label="GP Posterior")
            ax.fill_between(x, p, alpha=0.2, color='blue')
            
            # Distilled Distribution (Yellow Histogram)
            s_psi = distilled_energies[i][:, idx]
            sns.histplot(s_psi, color='gold', ax=ax, label="Distilled Flow", stat='density', alpha=0.4, bins=30)
            
            ax.set_title(name)
            ax.set_xlabel(r"Strain Energy ($\Psi$)")
            ax.set_ylabel("Density")
            
            # Only add legend to the first plot to avoid clutter
            if i == 0:
                ax.legend(loc='upper right')
                
        fig.canvas.draw_idle()

    # Initial draw
    update(0.5)
    gamma_slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    main()
