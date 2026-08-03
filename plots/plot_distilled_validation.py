import argparse
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.features import IsotropicFeatureExtractor

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
    return F_all, gamma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, default="extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1")
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="isihara", choices=["ogden", "gmr", "isihara"])
    args = parser.parse_args()
    
    saved_model_dir = args.saved_model_dir
    distilled_dir = args.distilled_dir
    
    # 1. Load True Model dynamically from saved_model_dir folder name
    model_folder_name = os.path.basename(os.path.normpath(saved_model_dir))
    parts = model_folder_name.split('_')
    true_model_name = parts[1] if len(parts) > 1 else "isihara"
    true_model = get_material(true_model_name, jit_P=False)
    
    # 2. Load GP Model
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)
    
    # 3. Load Distilled Samples
    samples = np.load(os.path.join(distilled_dir, "flow_samples.npy"))
    num_samples = min(32, samples.shape[0])
    selected_samples = samples[:num_samples]
    
    # 4. Generate Data
    F_all, gamma = generate_standard_modes(num_points=100, max_gamma=1.0)
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    # Calculate exact interpolation / extrapolation transitions for each mode
    extractor = IsotropicFeatureExtractor()
    true_min_dev = min_dev - 1e-4
    true_max_dev = max_dev + 1e-4
    true_min_vol = min_vol - 1e-4
    true_max_vol = max_vol + 1e-4
    
    transitions = []
    for mode in range(len(mode_names)):
        dev_m, vol_m = jax.vmap(extractor.extract)(F_all[mode])
        in_bounds = ((dev_m[:, 0] >= true_min_dev[0]) & (dev_m[:, 0] <= true_max_dev[0]) &
                     (dev_m[:, 1] >= true_min_dev[1]) & (dev_m[:, 1] <= true_max_dev[1]) &
                     (vol_m[:, 0] >= true_min_vol[0]) & (vol_m[:, 0] <= true_max_vol[0]))
        if not jnp.all(in_bounds):
            idx = int(jnp.argmax(~in_bounds))
            trans_g = float(gamma[idx])
            if idx == 0:
                trans_g = float(gamma[1])
        else:
            trans_g = float(gamma.max())
        transitions.append(trans_g)

    # 5. Evaluate True and GP
    psi_true = jax.vmap(true_model.psi)(F_all)
    P_true = jax.vmap(jax.vmap(true_model.P))(F_all)
    
    psi_dist_mean = [learned_gp.psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    psi_dist_var = [learned_gp.psi_dist(F_all[mode]).var for mode in range(len(mode_names))]

    P_dist_mean = [learned_gp.piola_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    P_dist_var = [learned_gp.piola_dist(F_all[mode]).var for mode in range(len(mode_names))]
    
    # 6. Evaluate Distilled Samples
    def get_distilled_energy_stress(theta, F_chunk):
        if args.material_model == "ogden":
            mu = theta[:3]
            alpha = theta[3:6]
            vol = theta[6:9]
            mat = get_material("ogden", mu_params=mu, alpha_params=alpha, vol_params=vol, jit_P=False)
        elif args.material_model == "gmr":
            dev = theta[:11] if len(theta) >= 14 else theta[:9]
            vol = theta[11:14] if len(theta) >= 14 else theta[9:12]
            mat = get_material("gmr", dev_params=dev, vol_params=vol, jit_P=False)
        elif args.material_model == "isihara":
            mat = get_material("isihara", c10=theta[0], c01=theta[1], c20=theta[2], d1=theta[3], jit_P=False)
        return jax.vmap(mat.psi)(F_chunk), jax.vmap(mat.P)(F_chunk)
        
    dist_psi_samples = []
    dist_p_samples = []
    for mode in range(len(mode_names)):
        mode_F = F_all[mode]
        # vmap over samples
        s_psi, s_p = jax.vmap(lambda t: get_distilled_energy_stress(t, mode_F))(selected_samples)
        dist_psi_samples.append(s_psi)
        dist_p_samples.append(s_p)
        
    # 7. Plotting
    fig, axes = plt.subplots(6, 2, figsize=(12, 24))
    fig.suptitle(f"Distilled Flow Validation vs GP vs Ground Truth (Isihara $\\rightarrow$ {args.material_model.upper()})", fontsize=20, y=1.01)

    for i, name in enumerate(mode_names):
        if name == "Pure Shear":
            idx_comp = (1, 1); label_P = r"$P_{22}$"
        elif name == "Simple Shear":
            idx_comp = (0, 1); label_P = r"$P_{12}$"
        else:
            idx_comp = (0, 0); label_P = r"$P_{11}$"

        # Column 0: Energy
        ax_psi = axes[i, 0]
        ax_psi.plot(gamma, psi_true[i], 'k--', lw=2.0, label="True Physics", zorder=5)
        
        ax_psi.plot(gamma, dist_psi_samples[i].T, color="orange", lw=0.8, alpha=0.4, zorder=4)
        # Add a dummy line for legend
        ax_psi.plot([], [], color="orange", lw=2.0, label=f"Distilled Samples ({args.material_model.upper()})")

        ax_psi.plot(gamma, psi_dist_mean[i], color="blue", lw=2, label="GP Mean", zorder=3)
        ax_psi.fill_between(gamma, psi_dist_mean[i] - 1.96*jnp.sqrt(psi_dist_var[i]), 
                           psi_dist_mean[i] + 1.96*jnp.sqrt(psi_dist_var[i]), color="blue", alpha=0.1, zorder=2)
        
        y_min, y_max = jnp.min(psi_true[i]), jnp.max(psi_true[i])
        pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        ax_psi.set_ylim(y_min - pad, y_max + pad)
        ax_psi.set_xlim(0, gamma.max())

        # Column 1: Stress
        ax_p = axes[i, 1]
        p_true_comp = P_true[i, :, idx_comp[0], idx_comp[1]]
        p_mean_comp = P_dist_mean[i][:, idx_comp[0], idx_comp[1]]
        p_std_comp = jnp.sqrt(P_dist_var[i][:, idx_comp[0], idx_comp[1]])
        p_samples_comp = dist_p_samples[i][:, :, idx_comp[0], idx_comp[1]]

        ax_p.plot(gamma, p_true_comp, 'k--', lw=2.0, label="True Physics", zorder=5)
        
        ax_p.plot(gamma, p_samples_comp.T, color="orange", lw=0.8, alpha=0.4, zorder=4)
        ax_p.plot([], [], color="orange", lw=2.0, label=f"Distilled Samples ({args.material_model.upper()})")

        ax_p.plot(gamma, p_mean_comp, color="blue", lw=2, label="GP Mean", zorder=3)
        ax_p.fill_between(gamma, p_mean_comp - 1.96*p_std_comp, 
                         p_mean_comp + 1.96*p_std_comp, color="blue", alpha=0.1, zorder=2)

        y_min_p, y_max_p = jnp.min(p_true_comp), jnp.max(p_true_comp)
        pad_p = (y_max_p - y_min_p) * 0.1 if y_max_p != y_min_p else 1.0
        ax_p.set_ylim(y_min_p - pad_p, y_max_p + pad_p)
        ax_p.set_xlim(0, gamma.max())

        # Formatting
        ax_psi.set_title(f"{name}: Energy ($\\Psi$)")
        ax_p.set_title(f"{name}: Stress ({label_P})")
        
        trans_g = transitions[i]
        max_g = float(gamma.max())
        
        for ax in [ax_psi, ax_p]:
            ax.axvspan(0, min(trans_g, max_g), color='green', alpha=0.12, zorder=1, label="Interpolation Region" if (i == 0 and ax == ax_psi) else "")
            if trans_g < max_g:
                ax.axvspan(trans_g, max_g, color='red', alpha=0.12, zorder=1, label="Extrapolation Region" if (i == 0 and ax == ax_psi) else "")
                ax.axvline(x=trans_g, color='darkred', linestyle=':', lw=1.5, alpha=0.8, zorder=4)
            ax.set_xlabel(r"Stretch Measure ($\gamma$)")
            ax.grid(True, alpha=0.3)
            if i == 0: 
                ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_file = os.path.join(distilled_dir, f"distilled_validation_{args.material_model}.png")
    plt.savefig(save_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Validation plot saved to: {save_file}")

if __name__ == "__main__":
    main()
