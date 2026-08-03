---
name: bridge_distillation_discovery
description: Automate the bridge between JAX unsupervised Gaussian Process posterior predictions and PyTorch uncertainty-aware model discovery for isotropic hyperelasticity. Use this skill when extracting GP stress predictions, computing interpolation transitions, or running symbolic distillation pipelines.
---

# Skill: Bridging JAX GP Distillation & PyTorch Discovery (`bridge_distillation_discovery`)

This skill defines the standard operational workflow for taking a trained JAX Unsupervised Gaussian Process (`SparseHyperelasticityGP`) and distilling its learned constitutive behavior into interpretable, uncertainty-quantified PyTorch models.

---

## 1. Standard Deformation Modes & Interpolation Bounds
To ensure robust model discovery, the trained GP is evaluated across canonical deformation pathways.
- **Reference Script:** `distillation/export_gp_to_pytorch.py`
- **Canonical Modes:**
  1. *Uniaxial Tension:* $\mathbf{F} = \text{diag}(1+\gamma, 1.0, 1.0)$
  2. *Equibiaxial Tension:* $\mathbf{F} = \text{diag}(1+\gamma, 1+\gamma, 1.0)$
  3. *Pure Shear / Planar Tension:* $\mathbf{F} = \text{diag}(1+\gamma, (1+\gamma)^{-1}, 1.0)$
  4. *Uniaxial Compression:* $\mathbf{F} = \text{diag}((1+\gamma)^{-1}, 1.0, 1.0)$
  5. *Equibiaxial Compression:* $\mathbf{F} = \text{diag}((1+\gamma)^{-1}, (1+\gamma)^{-1}, 1.0)$
  6. *Simple Shear:* $\mathbf{F}_{12} = \gamma$, with remaining diag $= 1.0$

### Dynamic Feature Bounding
When preparing dataset arrays, use `IsotropicFeatureExtractor` from `core/features.py` to extract deviatoric (`dev`) and volumetric (`vol`) invariants. Always distinguish between **interpolation** (where the feature vector sits inside the convex hull of training full-field observations) and **extrapolation** regimes using `generate_standard_modes_interp()`.

---

## 2. Posterior GP Evaluation (JAX)
Evaluate the predictive posterior mean stress and uncertainty covariance from the trained GP model (`SparseHyperelasticityGP` in `core/model.py`). 

```python
import jax
import jax.numpy as jnp
from core.model import SparseHyperelasticityGP
from core.features import IsotropicFeatureExtractor

# Ensure 64-bit precision
jax.config.update("jax_enable_x64", True)

# 1. Prepare deformation modes & extract features
F_test_3x3 = ... # Shape: (N, 3, 3)
extractor = IsotropicFeatureExtractor()
dev_features, vol_features = jax.vmap(extractor.extract)(F_test_3x3)

# 2. Predict posterior stress mean and variance from trained GP parameters
# (Use the trained model's predictive routines)
pred_stress_mean, pred_stress_variance = gp_model.predict(dev_features, vol_features)
```

---

## 3. Safe Handover: The NumPy Bridge
> [!WARNING]
> **Never import PyTorch in the same active GPU execution block while JAX maintains control of CUDA tensors.** Doing so can cause CUDA initialization failures and device memory deadlocks.

To safely pass data to PyTorch:
1. Explicitly convert all JAX arrays (`F`, `pred_stress_mean`, `pred_stress_variance`, `invariants`) to host NumPy arrays using `np.array(jax_array)`.
2. Export these structured evaluations to disk as `.npz` or `.npy` files, or invoke PyTorch analysis in a cleanly decoupled python sub-process.

```python
import numpy as np

# Export safely to host memory/disk
np.savez(
    "distilled_gp_data.npz",
    F=np.array(F_test_3x3),
    stress_mean=np.array(pred_stress_mean),
    stress_var=np.array(pred_stress_variance),
    dev_features=np.array(dev_features),
    vol_features=np.array(vol_features)
)
```

---

## 4. Invoking PyTorch Uncertainty Model Discovery
Once intermediate predictions are generated, invoke the appropriate PyTorch distillation script located in `distillation/` to perform regression and symbolic expression search:
- **`distill_uqmodeldisc.py`**: Executes uncertainty-aware model discovery, leveraging GP posterior variances as reciprocal instance weights during discovery to prioritize high-confidence interpolation zones over extrapolation zones.
- **`distill_parameters_wasserstein.py`**: Uses Wasserstein distance / distribution matching to match probabilistic predictions between the GP and candidate constitutive expressions.
- **`distill_parameters_flow.py`**: Performs flow-based parameter estimation.

### Example Execution Command
When running discovery via terminal scripts:
```bash
python distillation/distill_uqmodeldisc.py --data_path distilled_gp_data.npz --epochs 1000
```
Ensure that the UQ model discovery library inside `UQInModelDiscovery` is accessible in `sys.path` or properly compiled as a package dependency.
