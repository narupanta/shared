import numpy as np
import jax
import jax.numpy as jnp
from core.material_models import get_material

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

from core.dataclass import GPRawParams
from core.model import SparseHyperelasticityGP
import os

saved_model_dir = "extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1"
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

distilled_dir = "distillation_models/20260721T093048_isihara_gmr_standard_modes"
samples = np.load(distilled_dir + "/flow_samples.npy")

print(f"Loaded {samples.shape[0]} samples. Sample 0 parameters:")
print(samples[0])

F_all, gamma = generate_standard_modes()

def get_distilled_energy_stress(theta, F_chunk):
    dev = theta[:9]
    vol = theta[9:12]
    mat = get_material("gmr", dev_params=dev, vol_params=vol, jit_P=False)
    return jax.vmap(mat.psi)(F_chunk), jax.vmap(mat.P)(F_chunk)

true_model = get_material("isihara", jit_P=False)

for m in range(6):
    mode_F = F_all[m]
    s_psi, s_p = jax.vmap(lambda t: get_distilled_energy_stress(t, mode_F))(samples[:5])
    print(f"Max Distilled Energy at gamma=2.0 (Sample 0, Mode {m}):", s_psi[0, -1])
    
    true_psi = jax.vmap(true_model.psi)(mode_F)
    print(f"True Energy at gamma=2.0 (Mode {m}):", true_psi[-1])
    
    gp_mean = learned_gp.psi_dist(mode_F).mean
    print(f"GP Mean Energy at gamma=2.0 (Mode {m}):", gp_mean[-1])
