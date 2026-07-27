import os
import numpy as np
import jax.numpy as jnp
from core.model import SparseHyperelasticityGP
from core.material_models import get_material
from core.plotter import plot_combined_validation

# Setup paths
dir_path = "extraction/extracted_models/20260713T134231_gentthomas_0.0001_0.01_2.5_0.9_5_1.0_1"

print("Loading parameters...")
class DotDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No attribute {key}")

p_dict = np.load(f"{dir_path}/best_params.npy", allow_pickle=True).item()
p = DotDict(p_dict)

I_z = np.load(f"{dir_path}/I_z.npy")
I_obs_all = np.load(f"{dir_path}/I_obs_all.npy")
dev = I_obs_all[:, :, :2].reshape(-1, 2)
vol = I_obs_all[:, :, 2:].reshape(-1, 1)

max_dev = np.max(dev, axis=0)
min_dev = np.min(dev, axis=0)
max_vol = np.max(vol, axis=0)
min_vol = np.min(vol, axis=0)

print("Initializing model...")
learned_gp = SparseHyperelasticityGP(
    p, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0
)

true_mat_model = get_material("gentthomas")

print("Generating plot with gamma=1.0...")
plot_combined_validation(learned_gp, true_mat_model, dir_path, 99999)
print("Plot successfully generated!")
