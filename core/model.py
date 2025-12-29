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

@jax.jit
def rbf_kernel(X1: jnp.array, X2: jnp.array, variance: float, lengthscales: jnp.array) -> jnp.array:
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
    K = variance * jnp.exp(-0.5 * exponent)
    return K    


def matern52_kernel(X1: jnp.array, X2: jnp.array, variance: float, lengthscales: jnp.array) -> jnp.array:
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
    K = variance * (1.0 + sqrt5 * dist + (5.0/3.0) * dist_sq) * jnp.exp(-sqrt5 * dist)
    
    return K

def polynom_kernel(X1, X2, sigma_poly, offset, degree) :
    dot = X1 @ X2.T
    K_poly = sigma_poly * (dot + offset)**degree
    return K_poly

def discovery_kernel(X1, X2, params):

    K_matern = matern52_kernel(X1, X2, jnp.exp(params["log_scale_variance"]), params["lengthscales"])
    K_poly = polynom_kernel(X1, X2, jnp.exp(params["log_sigma_poly"]), jnp.exp(params["log_offset"]), params["poly_degree"])

    return K_poly + K_matern

class SparseHyperelasticityGP :
    '''get g latent energy which will be joint gaussian with latent energy at testpoint'''
    def __init__(self, lengthscales, scaling_variance, sigma_poly, offset, growth_constant, poly_degree, inducing_latent_variable, inducing_invariants) :
        self.inducing_invariants = inducing_invariants
        self.params = {"lengthscales" : lengthscales, 
                       "scaling_variance" : scaling_variance, 
                       "inducing_latent_variable": inducing_latent_variable, 
                       "sigma_poly": sigma_poly, 
                       "offset": offset, 
                       "growth_constant": growth_constant, 
                       "poly_degree": poly_degree}

    def psi_dev_gp(self, i_star):
        '''
        __call__ : evaluate mean Physical Strain Energy Function at test point i_star

        :param self: Description
        :param i_star: Description
        '''
        j_star = jnp.sqrt(i_star[2] + 1e-9)
        i_star_dev = jnp.stack([j_star**(-2/3)*i_star[0], j_star**(-4/3)*i_star[1]], axis = -1)
        j_inducing = jnp.sqrt(self.inducing_points[:, 2] + 1e-9)
        inducing_points_dev = jnp.stack([j_inducing**(-2/3)*self.inducing_points[:, 0], j_inducing**(-4/3)*self.inducing_points[:, 1]], axis = -1)
        # inducing_points_dev = self.inducing_points
        Kzz = discovery_kernel(inducing_points_dev, inducing_points_dev, self.params)
        # Kzz = matern52_kernel(inducing_points_dev, inducing_points_dev, scale_variance, self.params["lengthscales"])
        Kzz += 1e-6 * jnp.eye(inducing_points_dev.shape[0])
        # Kiz = matern52_kernel(i_star_dev, inducing_points_dev, scale_variance, self.params["lengthscales"])
        Kiz = discovery_kernel(i_star_dev, inducing_points_dev, self.params)
        Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(inducing_points_dev.shape[0]))
        latent_var_mean = Kiz @ Kzz_inv @ (self.params["learnable_latent_energy"])
        return (jnp.log(1 + jnp.exp(latent_var_mean))).squeeze() 
    def psi_vol_gp(self, i_star):
        j_star = jnp.sqrt(i_star[2] + 1e-9)
        return jnp.exp(self.params["log_growth_constant"]) * (j_star - 1)**2
    def psi_gp_f(self, f) :
        i_star, _ = invariants_and_derivatives(f)
        return self.psi_dev_gp(i_star) + self.psi_vol_gp(i_star)
    def psi_dev_std(self, deformation_gradient, params):
        i_star, di_df = invariants_and_derivatives(deformation_gradient)
        j_star = jnp.sqrt(i_star[2] + 1e-9)
        i_star_dev = jnp.stack([j_star**(-2/3)*i_star[0], j_star**(-4/3)*i_star[1]], axis = -1)
        # 1. Setup Matrices
        j_inducing = jnp.sqrt(self.inducing_points[:, 2] + 1e-9)
        inducing_points_dev = jnp.stack([j_inducing**(-2/3)*self.inducing_points[:, 0], j_inducing**(-4/3)*self.inducing_points[:, 1]], axis = -1)
        Kzz = discovery_kernel(inducing_points_dev, inducing_points_dev, self.params)
        # Kzz = matern52_kernel(inducing_points_dev, inducing_points_dev, scale_variance, self.params["lengthscales"])
        Kzz += 1e-6 * jnp.eye(inducing_points_dev.shape[0])
        # Kiz = matern52_kernel(i_star_dev, inducing_points_dev, scale_variance, self.params["lengthscales"])
        Kiz = discovery_kernel(i_star_dev, inducing_points_dev, self.params)
        Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(inducing_points_dev.shape[0]))
        L_K = jnp.linalg.cholesky(Kzz)
        
        # 3. Compute Predictive Variance
        # Part A: Solve L_K * v = Kzx  => v = L_K^-1 * Kzx
        v = jax.lax.linalg.triangular_solve(L_K, Kiz, lower=True)
        
        # Part B: Standard GP Variance (Prior - Info gain)
        # diag(Kxx - v.T @ v)
        kxx = discovery_kernel(i_star_dev, i_star_dev, self.params)
        
        var_standard = kxx - Kiz @ Kzz_inv @ Kiz.T
        
        # Part C: Learned Uncertainty contribution
        # Solve L_K * w = L_S  (if you have full matrix L_S)
        # This represents Kzz^-1 * S_uu * Kzz^-1 term
        L_S = jnp.exp(params['g_log_var']) 
        # L_S = params['L_S'] # (M, M)
        S_uu = jnp.diag(L_S)
        
        # Compute Kxz * Kzz^-1 * S_uu * Kzz^-1 * Kzx
        # Which is (v.T @ L_K^-1) @ S_uu @ (L_K^-T @ v)
        # Or more simply:
        # tmp = jax.lax.linalg.triangular_solve(L_K.T, v, lower=False)
        var_learned = Kiz @ Kzz_inv @ S_uu @ Kzz_inv @ Kiz.T
        
        g_var = var_standard + var_learned
        Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(inducing_points_dev.shape[0]))
        latent_var_mean = Kiz @ Kzz_inv @ self.params["learnable_latent_energy"]
        psi_dev_var = ((jnp.exp(latent_var_mean)/(1 + jnp.exp(latent_var_mean)))**2 * g_var).squeeze()
        # jax.debug.breakpoint()
        return psi_dev_var
    
    def psi(self, deformation_gradient) :

        deformation_gradient_ref = jnp.eye(deformation_gradient.shape[-1])
        E =  0.5 * (deformation_gradient.T @ deformation_gradient - jnp.eye(deformation_gradient.shape[-1]))

        H = jax.grad(self.psi_gp_f)(deformation_gradient_ref)
        stress_correction = jnp.sum(H * E)
        psi = self.psi_gp_f(deformation_gradient) - self.psi_gp_f(deformation_gradient_ref) - stress_correction
        return psi
    
    def piola_stress(self, deformation_gradient) :
        piola = jax.grad(self.psi)(deformation_gradient)
        return piola
