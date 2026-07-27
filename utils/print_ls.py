import jax
import jax.numpy as jnp
import numpy as np
import os
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams

def main():
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
    
    # Actually instantiate GP to get the scaled lengthscales!
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)
    
    print("Dev Lengthscales:", learned_gp.params.dev_ls)
    print("Vol Lengthscales:", learned_gp.params.vol_ls)

if __name__ == "__main__":
    main()
