import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import os
import numpy as np

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.plotter import plot_combined_validation

def main():
    saved_model_dir = "extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1"
    
    # 1. Load True Model
    true_model = get_material("isihara", jit_P=False)
    
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
    
    # Note: beta=80.0 is passed in training, but we can just use beta=1.0 for prediction
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=80.0)
    
    # 3. Plot
    save_path = saved_model_dir
    step = 5000  # Arbitrary step number for filename
    print("Running plot_combined_validation...")
    plot_combined_validation(learned_gp, true_model, save_path, step)
    print("Plot generated successfully!")

if __name__ == "__main__":
    main()
