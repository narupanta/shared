# material_models_autodiff.py
from abc import ABC, abstractmethod
from typing import Dict, Type, Any
import jax
import jax.numpy as jnp

from .utils import C_func, B_func, I1_func, I2_func, I3_func, J_func

# Registry
_material_registry: Dict[str, Type["BaseMaterialModel"]] = {}


def register_material(name: str):
    def decorator(cls):
        _material_registry[name.lower()] = cls
        return cls
    return decorator


def get_material(name: str, **kwargs) -> "BaseMaterialModel":
    name = name.lower()
    if name not in _material_registry:
        raise ValueError(f"Unknown material '{name}'. Available: {list(_material_registry.keys())}")
    return _material_registry[name](**kwargs)


class BaseMaterialModel(ABC):
    """
    Base class: subclasses implement phi(F) (scalar per sample).
    P(F) is provided by this base class using JAX autograd:
        P_{iJ} = d phi / d F_{iJ}.
    This supports both single F (3,3) and batched F (...,3,3).
    """

    def __init__(self, jit_P: bool = True):
        """
        jit_P: if True, the computed P function will be jitted/vmap'ed for speed.
        """
        self.jit_P = jit_P
        # cached compiled functions (lazy)
        self._per_sample_grad = None
        self._batched_grad = None

    @abstractmethod
    def phi(self, F: jnp.ndarray) -> jnp.ndarray:
        """
        Strain energy for a single sample F (shape (3,3)) or for batched F (...,3,3).
        Must return a scalar (0-d array) per sample (or an array with leading batch dims).
        If you accidentally return an array, this base class will sum it before differentiating.
        """
        raise NotImplementedError

    def _make_grad_fns(self) -> None:
        """
        Create and cache grad functions for per-sample and batched inputs.
        We wrap phi so that the function passed to grad returns a scalar for a single 3x3 F.
        """
        if self._per_sample_grad is not None:
            return  # already created

        # per-sample scalar phi function: ensures a scalar for a single (3,3) input
        def per_sample_phi(F_single: jnp.ndarray) -> jnp.ndarray:
            # ensure phi returns a scalar; if phi returns non-scalar, sum it
            out = self.phi(F_single)
            return jnp.sum(out)

        # gradient of scalar w.r.t. F (returns same shape as F_single)
        per_sample_grad = jax.grad(per_sample_phi)

        # optionally jit the single-sample grad
        if self.jit_P:
            per_sample_grad = jax.jit(per_sample_grad)

        # batched version using vmap across leading flattened batch axis
        def batched_grad(F_batched: jnp.ndarray) -> jnp.ndarray:
            # F_batched shape (..., 3, 3)
            # flatten leading dims to (N,3,3), vmap per_sample_grad, then reshape back
            orig_shape = F_batched.shape
            if F_batched.ndim == 2:
                # single sample passed accidentally; delegate to per-sample
                return per_sample_grad(F_batched)
            leading = orig_shape[:-2]
            flat = F_batched.reshape((-1, orig_shape[-2], orig_shape[-1]))  # (N,3,3)
            # vmap the (possibly jitted) per-sample grad
            vmapped = jax.vmap(per_sample_grad)
            result_flat = vmapped(flat)  # (N,3,3)
            return result_flat.reshape(*leading, orig_shape[-2], orig_shape[-1])

        # optionally jit batched_grad for performance
        if self.jit_P:
            batched_grad = jax.jit(batched_grad)

        # cache
        self._per_sample_grad = per_sample_grad
        self._batched_grad = batched_grad

    def P(self, F: jnp.ndarray) -> jnp.ndarray:
        """
        Compute 1st Piola-Kirchhoff stress P = d(phi)/dF.
        Accepts single F (3,3) or batched F (...,3,3). Returns same leading shape with last two dims (3,3).
        """
        F = jnp.asarray(F)
        self._make_grad_fns()
        # pick correct function based on input rank
        if F.ndim == 2:  # (3,3) single sample
            return self._per_sample_grad(F)
        else:
            return self._batched_grad(F)


# ------------------------------
# Mooney–Rivlin and Neo-Hookean that only implement phi()
# ------------------------------

@register_material("mooney-rivlin")
class MooneyRivlin(BaseMaterialModel):
    def __init__(self, c01 = 1.0, c02 = 1.0, c10 = 1.0, c11 = 1.0, c12 = 1.0, c20 = 1.0, c21 = 1.0, c22 = 1.0, d0 = 1.0, d1 = 1.0, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.dev_params = [c01, c02, c10, c11, c12, c20, c21, c22]
        self.vol_params = [d0, d1]

    def phi(self, F: jnp.ndarray) -> jnp.ndarray:
        if F.shape[-2:] == (2, 2):
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                        [F[1, 0], F[1, 1], 0.],
                        [0.,      0.,     1. ]])
        c = C_func(F)
        I1 = I1_func(c)
        I2 = I2_func(c)
        I3 = I3_func(c)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
        i1_dev = I3_safe**(-1/3) * I1
        i2_dev = I3_safe**(-2/3) * I2

        X = i1_dev - 3.0
        Y = i2_dev - 3.0
        
        # --- Deviatoric Terms (W) ---
        # Assuming dev_params = [c01, c02, c10, c11, c12, c20, c21, c22]
        # Using the standard N=2 Polynomial Model terms (C10, C01, C20, C11, C02)
        dev_terms = (
            # C10 * X
            self.dev_params[2] * X + 
            # C01 * Y
            self.dev_params[0] * Y + 
            # C20 * X**2
            self.dev_params[5] * X**2 + 
            # C11 * X * Y
            self.dev_params[3] * X * Y + 
            # C02 * Y**2
            self.dev_params[1] * Y**2 +

            self.dev_params[4] * X*Y**2 + 

            self.dev_params[6] * X**2 * Y + 

            self.dev_params[7] * X**2 * Y ** 2

            # Add C12, C21, C22 terms here if required by your specific model definition
        )
        
        # --- Volumetric Terms (U) ---
        # Assuming vol_params = [d0, d1] are D2 and D1 parameters (inverse bulk moduli)
        J = jnp.sqrt(I3_safe)
        J_minus_1 = J - 1.0

        # Assuming the volumetric function U(J) = (1/D1)(J-1)^2 + (1/D2)(J-1)^4
        # with D1=d1 and D2=d0 (or vice versa, depending on convention)
        
        # D1 is typically the lower order term (quadratic, hence d1)
        # D2 is typically the higher order term (quartic, hence d0)
        vol_terms = (
            # (1/D1) * (J - 1)**2
            (self.vol_params[0]) * J_minus_1**2 + 
            # (1/D2) * (J - 1)**4
            (self.vol_params[1]) * J_minus_1**4
        )
        
        return dev_terms + vol_terms


@register_material("neohookean")
class NeoHookean(BaseMaterialModel):
    def __init__(self, c1=0.5, c2=1.5, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        # E = 70.e3
        # nu = 0.3
        # mu = E/(2.*(1. + nu))
        # kappa = E/(3.*(1. - 2.*nu))
        self.c1 = c1
        self.c2 = c2

    def phi(self, F: jnp.ndarray) -> jnp.ndarray:
        B = B_func(F)
        I1 = I1_func(B)
        I3 = I3_func(B)
        term1 = self.c1 * (I3**(-1/3) * I1 - 3)
        term2 = self.c2 * (jnp.sqrt(I3) - 1)**2
        return term1 + term2


@register_material("isihara")
class Isihara(BaseMaterialModel):
    def __init__(self, c1=0.5, c2=1.5, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.c1 = c1
        self.c2 = c2

    def phi(self, F: jnp.ndarray) -> jnp.ndarray:
        C = C_func(F)
        I1 = I1_func(C)
        I2 = I2_func(C)
        I3 = I3_func(C)
        I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
        term1 = self.c1 * (I3_safe**(-1/3) * I1 - 3)
        term2 = (I3_safe**(-2/3) * I2 - 3)
        term3 = (I3_safe**(-1/3) * I1 - 3)**2
        term4 = self.c2 * (jnp.sqrt(I3_safe) - 1)**2
        return term1 + term2 + term3 + term4



# ------------------------------
# Example small tests / usage
# ------------------------------
if __name__ == "__main__":
    # create a small deformation gradient (single)
    F_single = jnp.array([[1.1, 0.1, 0.0],
                          [0.0, 1.05, 0.0],
                          [0.0, 0.0, 0.98]])
    mm = get_material("neo-hookean", c1=1.0, c2=1.5)
    P_single = mm.P(F_single)  # shape (3,3)
    print("P_single:", P_single)

    # batched F (N,3,3)
    F_batch = jnp.stack([F_single, F_single * 1.01], axis=0)  # (2,3,3)
    P_batch = mm.P(F_batch)  # (2,3,3)
    print("P_batch shape:", P_batch.shape)
