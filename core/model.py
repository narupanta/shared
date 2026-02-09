
from core.utils import *
from core.datasetclass import *
import jax.numpy as jnp
from typing import NamedTuple
import jax



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
def rbf_s(f1, f2, sigma_scaling, lengthscales) -> jnp.array:
    """
    Computes the ARD RBF Gram/Covariance matrix.
    X1, X2 are (N, D) and (M, D).
    lengthscales is a vector (D,).
    """
    # 1. Calculate the raw
    r = (f1 - f2)
    exponent = jnp.sum((r / lengthscales)**2, axis=-1)
    ret = sigma_scaling * jnp.exp(-0.5 * exponent)
    return ret

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
def discovery_kernel(X1, X2, scaling, lengthscales):

    # K_matern = matern52_kernel(X1, X2, scaling, lengthscales)
    K_rbf = rbf_kernel(X1, X2, scaling , lengthscales)
    # K_poly = polynom_kernel(X1, X2, params["sigma_poly"], params["offset"], params["poly_degree"])

    return K_rbf

def transform_input_features(invariants) :
    i3 = jnp.maximum(invariants[2], 1e-6)
    j = jnp.sqrt(i3)
    i1_dev = i3**(-1/3)*invariants[0]
    i2_dev = i3**(-2/3)*invariants[1]
    dev_feature = jnp.stack([i1_dev, i2_dev], axis = -1)
    # vol_feature = jnp.stack([j, -2 * j], axis = -1)
    vol_feature = jnp.array([j])

    return dev_feature, vol_feature
class GaussianDistribution :
    def __init__(self, mean, var) :
        self.mean = mean
        self.var = var
    def sample(self, key) :
        eps = jax.random.normal(key, self.mean.shape)
        return self.mean + jnp.sqrt(self.var) * eps


def enforce_softplus_positive(variable) :
    return jnp.log(1 + jnp.exp(variable))

class EnergyDist(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray

class StressDist(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray

from typing import NamedTuple

class GPRawParams(NamedTuple):
    raw_dev_ls: jnp.ndarray
    raw_dev_sig: jnp.ndarray
    raw_dev_u_mean: jnp.ndarray
    raw_dev_u_var: jnp.ndarray

    raw_vol_ls: jnp.ndarray
    raw_vol_sig: jnp.ndarray
    raw_vol_u_mean: jnp.ndarray
    raw_vol_u_var: jnp.ndarray

    raw_c01: jnp.ndarray
    raw_c02: jnp.ndarray
    raw_c10: jnp.ndarray
    raw_c11: jnp.ndarray
    raw_c20: jnp.ndarray
    raw_k: jnp.ndarray
    raw_q: jnp.ndarray
    raw_s: jnp.ndarray


    log_sigma_phys: jnp.ndarray
    log_simga_glob: jnp.ndarray
    
class GPParams(NamedTuple) :
    dev_ls: jnp.ndarray
    dev_sig: jnp.ndarray
    dev_u_mean: jnp.ndarray
    dev_u_var: jnp.ndarray

    vol_ls: jnp.ndarray
    vol_sig: jnp.ndarray
    vol_u_mean: jnp.ndarray
    vol_u_var: jnp.ndarray

    c01: jnp.ndarray
    c02: jnp.ndarray
    c10: jnp.ndarray
    c11: jnp.ndarray
    c20: jnp.ndarray
    k: jnp.ndarray
    q: jnp.ndarray
    s: jnp.ndarray

    sigma_phys: jnp.ndarray
    sigma_glob: jnp.ndarray

class GPWeights(NamedTuple) :
    dev_Kzz: jnp.ndarray
    dev_v: jnp.ndarray
    dev_trace_term: jnp.ndarray
    dev_mahalanobis_term: jnp.ndarray
    dev_M_mat: jnp.ndarray
    dev_Kzz_inv: jnp.ndarray
    dev_logterm: jnp.ndarray

    vol_Kzz: jnp.ndarray
    vol_v: jnp.ndarray
    vol_trace_term: jnp.ndarray
    vol_mahalanobis_term: jnp.ndarray
    vol_M_mat: jnp.ndarray
    vol_Kzz_inv: jnp.ndarray
    vol_logterm: jnp.ndarray




class SparseHyperelasticityGP :
    '''get g latent energy which will be joint gaussian with latent energy at testpoint'''
    def __init__(self, raw_params: GPRawParams, I_z: jnp.ndarray, min_dev, min_vol):
        self.dev_z = I_z[:, :2]
        self.vol_z = I_z[:, 2:]
        self.min_dev = min_dev
        self.min_vol = min_vol
        # 1. Transform raw optimization variables into physical parameters
        self.params = self.load_params(raw_params)
        self.vmapped_rbf_s = jax.jit(jax.vmap(rbf_s, in_axes=(None, 0, None, None)))
        self.tmapped_rbf_s = jax.vmap(jax.vmap(rbf_s, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))
        # 2. Precompute the Cholesky-backed weights for the current iteration
        self.gpweight = self.precompute_weights(raw_params)

    def load_params(self, p: GPRawParams) -> GPParams:
        """Applies physical constraints and transformations once per iteration."""
        return GPParams(
            dev_ls=jax.nn.softplus(p.raw_dev_ls) + self.min_dev, # Added jitter for stability
            dev_sig=jnp.exp(p.raw_dev_sig),
            dev_u_mean=jax.nn.softplus(p.raw_dev_u_mean),
            dev_u_var=jax.nn.softplus(p.raw_dev_u_var) + 1e-6,

            vol_ls=jax.nn.softplus(p.raw_vol_ls) + self.min_vol,
            vol_sig=jnp.exp(p.raw_vol_sig),
            vol_u_mean=jax.nn.softplus(p.raw_vol_u_mean),
            vol_u_var=jax.nn.softplus(p.raw_vol_u_var) + 1e-6,

            # Hyperelastic constants (Mooney-Rivlin)
            **{k: jax.nn.softplus(getattr(p, f"raw_{k}")) for k in ['c01', 'c02', 'c10', 'c11', 'c20', 'k', 'q', "s"]},
            
            sigma_phys=jnp.exp(p.log_sigma_phys) + 1e-4,
            sigma_glob=jnp.exp(p.log_simga_glob) + 1e-4
        )
    

    def _compute_component_weights(self, z, u_mean, u_var, ls, sig, trend_fn):
        """Generic solver for any GP component (Dev or Vol)."""
        Kzz = self.tmapped_rbf_s(z,z, sig, ls) + 1e-6 * jnp.eye(z.shape[0])
        K_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))
        v = u_mean - jax.vmap(trend_fn)(z)
        M_mat = K_inv @ (Kzz - jnp.diag(u_var)) @ K_inv.T
        trace_term = jnp.trace(K_inv @ jnp.diag(u_var))
        mahalanobis_term = v.T @ K_inv @ v
        log_term = jnp.log(jnp.linalg.det(Kzz)) - jnp.log(jnp.linalg.det(jnp.diag(u_var)))
        
        return Kzz, K_inv, v, trace_term, mahalanobis_term, M_mat, log_term

    def precompute_weights(self, params: GPParams) -> GPWeights:
        params = self.load_params(params)
        
        d_k, d_k_inv, d_v, d_tr, d_ma, d_m, d_lt = self._compute_component_weights(
            self.dev_z, params.dev_u_mean, params.dev_u_var, params.dev_ls, params.dev_sig, self.dev_mean_func
        )
        
        v_k, v_k_inv, v_v, v_tr, v_ma, v_m, v_lt = self._compute_component_weights(
            self.vol_z, params.vol_u_mean, params.vol_u_var, params.vol_ls, params.vol_sig, self.vol_mean_func
        )

        return GPWeights(dev_Kzz= d_k, dev_Kzz_inv= d_k_inv, dev_v= d_v, dev_trace_term= d_tr, dev_mahalanobis_term= d_ma, dev_M_mat= d_m, dev_logterm= d_lt,
                        vol_Kzz= v_k, vol_Kzz_inv= v_k_inv, vol_v= v_v, vol_trace_term= v_tr, vol_mahalanobis_term= v_ma, vol_M_mat= v_m, vol_logterm= v_lt)
    
    def dev_mean_func(self, d) :
        params = self.params
        i1_bar_3 = d[0] - 3
        i2_bar_3 = d[1] - 3
        return params.c01 * i1_bar_3 + params.c10 * i2_bar_3 + params.c11 * i1_bar_3 * i2_bar_3 + params.c02 * i1_bar_3**2 + params.c20 * i2_bar_3**2
    def vol_mean_func(self, v) :
        params = self.params
        return params.k * (v[0] - 1) ** 2 + params.s * (v[0] - 1) ** 4 + params.q * jnp.log(v[0])**2
    def dev_gp_mean(self, d) :
        mean_prior = self.dev_mean_func(d)
        dev_v = self.gpweight.dev_v
        k_dz = self.vmapped_rbf_s(d, self.dev_z, self.params.dev_sig, self.params.dev_ls)
        gp_term = k_dz @ self.gpweight.dev_Kzz_inv @ dev_v
        return mean_prior + gp_term
    
    def vol_gp_mean(self, v) :
        mean_prior = self.vol_mean_func(v)
        vol_v = self.gpweight.vol_v
        k_vz = self.vmapped_rbf_s(v, self.vol_z, self.params.vol_sig, self.params.vol_ls)
        gp_term = k_vz @ self.gpweight.vol_Kzz_inv @ vol_v
        return mean_prior + gp_term
    
    def psi_gp_mean(self, f) :
        invariants, _ = invariants_and_derivatives(f)
        dev, vol = transform_input_features(invariants)
        gp_mean = self.dev_gp_mean(dev) + self.vol_gp_mean(vol)
        return gp_mean
    def psi_gp_var_diagonal(self, f):
        """
        Computes only the energy variance at a single point f.
        This is O(M^2) and much faster for jax.grad than the full covariance.
        """
        invariants, _ = invariants_and_derivatives(f)
        dev, vol = transform_input_features(invariants)

        # --- 1. Deviatoric Variance ---
        # Kernel vector k(f, Z)
        k_dz = self.vmapped_rbf_s(dev, self.dev_z, self.params.dev_sig, self.params.dev_ls)
        var_dev = self.params.dev_sig - k_dz @ self.gpweight.dev_M_mat @ k_dz.T


        # --- 2. Volumetric Variance ---
        k_vz = self.vmapped_rbf_s(vol, self.vol_z, self.params.vol_sig, self.params.vol_ls)
        var_vol = self.params.vol_sig - k_vz @ self.gpweight.vol_M_mat @ k_vz.T

        return var_dev + var_vol
    
    def psi_gp_cov(self, f1, f2) :
        invariants1, _ = invariants_and_derivatives(f1)
        invariants2, _ = invariants_and_derivatives(f2)
        dev1, vol1 = transform_input_features(invariants1)
        dev2, vol2 = transform_input_features(invariants2)
        k_d1z = self.vmapped_rbf_s(dev1, self.dev_z, self.params.dev_sig, self.params.dev_ls)
        k_d2z = self.vmapped_rbf_s(dev2, self.dev_z, self.params.dev_sig, self.params.dev_ls)
        k_d1d2 = rbf_s(dev1, dev2, self.params.dev_sig, self.params.dev_ls)

        cov_mat_dev = k_d1d2 - k_d1z @ self.gpweight.dev_M_mat @ k_d2z.T

        k_v1z = self.vmapped_rbf_s(vol1, self.vol_z, self.params.vol_sig, self.params.vol_ls)
        k_v2z = self.vmapped_rbf_s(vol2, self.vol_z, self.params.vol_sig, self.params.vol_ls)
        k_v1v2 = rbf_s(vol1, vol2, self.params.vol_sig, self.params.vol_ls)
        cov_mat_vol = k_v1v2 - k_v1z @ self.gpweight.vol_M_mat @ k_v2z.T

        return cov_mat_dev + cov_mat_vol
    
    def psi_gp_dist(self, f) :
        return EnergyDist(self.psi_gp_mean(f), self.psi_gp_var_diagonal(f))
    def piola_gp_cov(self, f1, f2) :
        stress_cov_tensor = jax.jacfwd(jax.jacrev(self.psi_gp_cov, argnums=1), argnums=0)(f1, f2)
        piola_var = jnp.einsum('ijij->ij', stress_cov_tensor)
        return piola_var
    def piola_gp_dist(self, f):
        piola_mean = jax.grad(lambda tensor: self.psi_gp_mean(tensor))(f)
        piola_var = self.piola_gp_cov(f, f)
        return StressDist(piola_mean, piola_var)
    
    def psi_dist(self, f):
        no_f = jnp.eye(f.shape[-1])
        E = 0.5 * (f.T @ f - no_f)

        psi_mean = self.psi_gp_mean(f) - self.psi_gp_mean(no_f) - jnp.sum(self.piola_gp_dist(no_f).mean * E)
        free_stress_cov_tensor = jax.jacfwd(jax.jacrev(self.psi_gp_cov, argnums=1), argnums=0)(no_f, no_f)
        var_grad_term = jnp.einsum('ij,ijkl,kl', E, free_stress_cov_tensor, E)
        _, cov_f_gradE = jax.jvp(lambda f_prime: self.psi_gp_cov(f, f_prime), (no_f,), (E,))
        _, cov_0_gradE = jax.jvp(lambda f_prime: self.psi_gp_cov(no_f, f_prime), (no_f,), (E,))
        term_cov = cov_f_gradE - cov_0_gradE
        psi_var = self.psi_gp_cov(f, f) + self.psi_gp_cov(no_f, no_f) - 2 * self.psi_gp_cov(f, no_f) + var_grad_term - 2 * term_cov
        psi_safe_var = jnp.maximum(psi_var, 1e-9)
        return EnergyDist(psi_mean, psi_safe_var)
    
    def piola_dist(self, f) :
        no_f = jnp.eye(f.shape[-1])
        piola_gp_dist = self.piola_gp_dist(f)
        piola_ref_dist = self.piola_gp_dist(no_f)
        
        piola_mean = piola_gp_dist.mean - f @ piola_ref_dist.mean
        piola_var = piola_gp_dist.var + f @ piola_ref_dist.var @ f.T - 2 * f @ self.piola_gp_cov(f, no_f)
        piola_safe_var = jnp.maximum(piola_var, 1e-9)
        return StressDist(piola_mean, piola_safe_var)

    def psi(self, f, key = None) :
        dist = self.psi_dist(f)
        if key is None :
            return dist.mean
        else :
            return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)
        
    def piola(self, f, key = None) :
        return jax.grad(lambda f : self.psi(f, key))(f)

    def kl_divergance(self):
        def component_kl(ma_term, log_term, trace_term, M):
            return 0.5 * (log_term - M + trace_term + ma_term)
        dev_kl = component_kl(self.gpweight.dev_mahalanobis_term, self.gpweight.dev_logterm, self.gpweight.dev_trace_term, self.dev_z.shape[0])
        vol_kl = component_kl(self.gpweight.vol_mahalanobis_term, self.gpweight.vol_logterm, self.gpweight.vol_trace_term, self.vol_z.shape[0])
        return dev_kl + vol_kl

