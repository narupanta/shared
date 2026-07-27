import jax
import jax.numpy as jnp
import os
import numpy as np
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from distill_parameters_wasserstein import generate_standard_modes, FlaxMADE, Critic, sample_maf
from core.material_models import get_material

f3x3_flat = generate_standard_modes(num_points=32, max_gamma=1.0)
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

gp_model = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)

mean_psi = gp_model.psi_gp_mean(f3x3_flat)
cov_psi = gp_model.psi_joint_cov(f3x3_flat)

print(f"mean_psi has nan: {jnp.isnan(mean_psi).any()}")
print(f"cov_psi has nan: {jnp.isnan(cov_psi).any()}")

rng_seq = jax.random.PRNGKey(42)
k_gp, k_flow, rng_seq = jax.random.split(rng_seq, 3)

gp_samples = jax.random.multivariate_normal(k_gp, mean_psi, cov_psi, shape=(32,))
print(f"gp_samples has nan: {jnp.isnan(gp_samples).any()}")

flow_module = FlaxMADE(num_params=12)
dummy_theta = jnp.zeros((1, 12))
flow_params = flow_module.init(k_flow, dummy_theta)

theta_raw = sample_maf(flow_module, flow_params, k_flow, 32, 12)
theta = jax.nn.softplus(theta_raw)
print(f"theta has nan: {jnp.isnan(theta).any()}")

def get_model_psi(t):
    dev = t[:9] 
    vol = t[9:12]
    mat = get_material("gmr", dev_params=dev, vol_params=vol, jit_P=False)
    return jax.vmap(mat.psi)(f3x3_flat)
    
model_samples = jax.vmap(get_model_psi)(theta)
print(f"model_samples has nan: {jnp.isnan(model_samples).any()}")
