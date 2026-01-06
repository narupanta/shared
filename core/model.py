# gpjax_version.py
import os
from flax.serialization import to_state_dict, from_state_dict
from core.utils import *
from core.datasetclass import *
import jax.numpy as jnp
import jax.random as jr
import pickle
key = jr.key(123)
    
import jax.numpy as jnp
import jax
# def input_normalize()
@jax.jit
def rbf_kernel(X1: jnp.array, X2: jnp.array, sigma_scaling: float, lengthscales: jnp.array) -> jnp.array:
    """
    Computes the ARD RBF Gram/Covariance matrix.
    X1, X2 are (N, D) and (M, D).
    lengthscales is a vector (D,).
    """
    # 1. Calculate the raw difference tensor: (N, M, D)
    X1 = jnp.atleast_2d(X1)
    X2 = jnp.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    
    # 2. Apply the weighted (inverse lengthscale squared) distance for each dimension
    # This is the core ARD step: (x_d - x'_d)^2 / (l_d)^2
    # The exponent is a sum over the D dimension: (N, M)
    exponent = jnp.sum((diff / lengthscales)**2, axis=-1)
    
    # RBF formula: k(x, x') = sigma^2 * exp(- 0.5 * sum_d (x_d - x'_d)^2 / (l_d)^2)
    K = sigma_scaling**2 * jnp.exp(-0.5 * exponent)
    return K    


def matern52_kernel(X1: jnp.array, X2: jnp.array, sigma_scaling: float, lengthscales: jnp.array) -> jnp.array:
    """
    Computes the ARD Matern 5/2 Gram matrix.
    X1: (N, D), X2: (M, D), lengthscales: (D,)
    """
    # 1. Scaled squared difference: sum_d ( (x_d - x'_d) / l_d )^2
    # This represents the squared Mahalanobis distance
    X1 = jnp.atleast_2d(X1)
    X2 = jnp.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    dist_sq = jnp.sum((diff / lengthscales)**2, axis=-1)
    
    # 2. Add a small epsilon for numerical stability when taking the sqrt
    # dist = sqrt(sum (delta_x / l)^2)
    dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-12))
    
    # 3. Matern 5/2 formula
    sqrt5 = jnp.sqrt(5.0)
    K = sigma_scaling ** 2 * (1.0 + sqrt5 * dist + (5.0/3.0) * dist_sq) * jnp.exp(-sqrt5 * dist)
    
    return K

def polynom_kernel(X1, X2, sigma_poly, offset, degree) :
    dot = X1 @ X2.T
    K_poly = sigma_poly**2 * (dot + offset)**2
    return K_poly
def polynom_mean_func(i_dev, params) :
    i1_dev = i_dev[0]
    i2_dev = i_dev[1] 
    return params["p"] * i1_dev**2 + params["q"] * i2_dev**2 + params["r"] * i1_dev * i2_dev + params["s"] * i1_dev + params["t"] * i2_dev + params["c"]
def discovery_kernel(X1, X2, params):

    K_matern = matern52_kernel(X1, X2, params["sigma_scaling"], params["lengthscales"])
    # K_rbf = rbf_kernel(X1, X2, params["sigma_scaling"], params["lengthscales"])
    # K_poly = polynom_kernel(X1, X2, params["sigma_poly"], params["offset"], params["poly_degree"])

    return K_matern
def _dev_input(invariants) :
    i3 = jnp.maximum(invariants[2], 1e-6)
    i1_dev = i3**(-1/3)*invariants[0]
    i2_dev = i3**(-2/3)*invariants[1]
    return jnp.stack([i1_dev, i2_dev, i3], axis = -1)
class SparseHyperelasticityGP :
    '''get g latent energy which will be joint gaussian with latent energy at testpoint'''
    def __init__(self, params, z) :

        self.inducing_points = {
            "inducing_invariants": z,
            "inducing_invariants_dev": jax.vmap(_dev_input)(z),
            }
        self.normalizer = {"max_inducing_invariants_dev": self.inducing_points["inducing_invariants_dev"].max(axis = 0),
                           "min_inducing_invariants_dev": self.inducing_points["inducing_invariants_dev"].min(axis = 0),}
        self.normalized_inducing_points = {"normed_inducing_invariants_dev": 
                                           (self.inducing_points["inducing_invariants_dev"] - self.normalizer["min_inducing_invariants_dev"]) / 
                                           (self.normalizer["max_inducing_invariants_dev"] - self.normalizer["min_inducing_invariants_dev"])}
        self.params = {
            "lengthscales" : params["lengthscales"], 
            "sigma_scaling" : jnp.exp(params["log_sigma_scaling"]),
            # "sigma_scaling" : 1,  
            "sigma_poly": jnp.exp(params["log_sigma_poly"]), 
            "offset": params["offset"], 
            "growth_constant": jnp.exp(params["log_growth_constant"]), 
            "poly_degree": params["poly_degree"],
            # "inducing_invariants": params["inducing_invariants"],
            # "inducing_invariants_dev": jax.vmap(_dev_input)(params["inducing_invariants"]),
            "inducing_latent_variable_mean": jnp.exp(params["log_inducing_latent_variable_mean"]), 
            "inducing_latent_variable_var": jnp.exp(params["log_inducing_latent_variable_var"]),
            "p": params["p"],
            "q": params["q"],
            "r": params["r"],
            "s": params["s"],
            "t": params["t"],
            "c": params["c"],
            }
    def psi(self, deformation_gradient, key = None) :

        deformation_gradient_ref = jnp.eye(deformation_gradient.shape[-1])
        E =  0.5 * (deformation_gradient.T @ deformation_gradient - jnp.eye(deformation_gradient.shape[-1]))
        psi_model = lambda f : self.psi_model(f, key)
        H = jax.grad(psi_model)(deformation_gradient_ref)
        check = H.T == H
        stress_correction = jnp.sum(H * E)
        psi = psi_model(deformation_gradient) - psi_model(deformation_gradient_ref) - stress_correction
        # jax.debug.breakpoint()
        return psi
    def psi_model(self, inputs, key = None) :
        if inputs.shape == (3, 3) :
            i_star, _ = invariants_and_derivatives(inputs)
            return self._psi_model(i_star, key).squeeze()
        else :
            i_star = inputs
            return self._psi_model(i_star, key).squeeze()
    def _psi_model(self, i_star, key = None) :
        i_star = _dev_input(i_star)
        i_star12 = i_star[:2]
        i_star3 = i_star[2]
        return self._psi_model_dev(i_star12, key) + self._psi_model_vol(i_star3)
    
    def _psi_model_dev(self, i_dev, key = None) :
        # inducing_invariants_dev = self.params["inducing_invariants_dev"][:, :2]
        # inducing_invariants_dev = self.inducing_points["inducing_invariants_dev"][:, :2]
        inducing_invariants_dev = self.normalized_inducing_points["normed_inducing_invariants_dev"][:, :2]
        i_dev = (i_dev - self.normalizer["min_inducing_invariants_dev"][:2]) / (self.normalizer["max_inducing_invariants_dev"][:2] - self.normalizer["min_inducing_invariants_dev"][:2])
        Kzz = discovery_kernel(inducing_invariants_dev, inducing_invariants_dev, self.params)
        Kzz += 1e-6 * jnp.eye(inducing_invariants_dev.shape[0])
        Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(inducing_invariants_dev.shape[0]))
        Kiz = discovery_kernel(i_dev, inducing_invariants_dev, self.params)
        Kii = discovery_kernel(i_dev, i_dev, self.params)
        mu_latent_variable = self.params["inducing_latent_variable_mean"]
        S_uu = jnp.diag(self.params["inducing_latent_variable_var"])

        def mean(Kiz, Kzz_inv, inducing_latent_variable) :
            mu_g = Kiz @ Kzz_inv @ (inducing_latent_variable)
            return mu_g
        def var(Kii, Kiz, Kzz_inv, S_uu) :

            var_standard = Kii - Kiz @ Kzz_inv @ Kiz.T
            var_learned = Kiz @ Kzz_inv @ S_uu @ Kzz_inv @ Kiz.T
            
            var_g = var_standard + var_learned
            return var_g
        
        mu_g = mean(Kiz, Kzz_inv, mu_latent_variable)
        var_g = var(Kii, Kiz, Kzz_inv, S_uu)
        if key is not None:
            # TRAINING MODE: Sample from the GP posterior
            std_g = jnp.sqrt(jnp.maximum(var_g, 1e-9))
            eps = jax.random.normal(key, mu_g.shape)
            g_sample = mu_g + std_g * eps
            # g = jnp.log(1 + jnp.exp(g_sample))
            g = g_sample
        else:
            # INFERENCE/PLOTTING MODE: Use the mean (or a better approximation)
            # g = jnp.log(1 + jnp.exp(mu_g))
            g = mu_g
        return g

    def _psi_model_vol(self, i3) :
        return self.params["growth_constant"] * (jnp.sqrt(i3) - 1) ** 2
    
    def piola_stress(self, deformation_gradient, key = None) :
        piola = jax.grad(lambda x : self.psi(x, key))(deformation_gradient)
        return piola
