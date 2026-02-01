
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
            **{k: jax.nn.softplus(getattr(p, f"raw_{k}")) for k in ['c01', 'c02', 'c10', 'c11', 'c20', 'k', 'q']},
            
            sigma_phys=jnp.exp(p.log_sigma_phys) + 1e-4,
            sigma_glob=jnp.exp(p.log_simga_glob) + 1e-4
        )
    

    def _compute_component_weights(self, z, u_mean, u_var, ls, sig, trend_fn):
        """Generic solver for any GP component (Dev or Vol)."""
        Kzz = self.tmapped_rbf_s(z,z, sig, ls) + 1e-6 * jnp.eye(z.shape[0])
        K_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))

        # This matrix captures the entire variational uncertainty structure
        # M = K_inv @ diag(u_var) @ K_inv
        # Since u_var is diagonal, this is just scaling columns then multiplying
        # mean posterior
        v = u_mean - jax.vmap(trend_fn)(z)
        # var posterior
        M_mat = K_inv @ (Kzz - jnp.diag(u_var)) @ K_inv.T
        # for kl term
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
        return params.k * (v[0] - 1) ** 2 + params.q * jnp.log(v[0])**2
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
        # vampped_transform = jax.vmap(transform_input_features)
        dev1, vol1 = transform_input_features(invariants1)
        dev2, vol2 = transform_input_features(invariants2)

        # --- Deviatoric Covariance ---
        # vmapped_rbf_s = jax.vmap(rbf_s, in_axes=(None, 0, None, None))
        # 1. Compute kernel vectors between test points and inducing points
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

        # 3. Extract Variance: Diagonal of the 4th order tensor
        # Var(P_ij) corresponds to Cov(P_ij, P_ij)
        piola_var = jnp.einsum('ijij->ij', stress_cov_tensor)
        return piola_var
    def piola_gp_dist(self, f):
        # 1. Compute Mean: Gradient of the Mean Energy
        # We use a lambda to ensure jax.grad only sees the scalar mean output
        piola_mean = jax.grad(lambda tensor: self.psi_gp_mean(tensor))(f)
        piola_var = self.piola_gp_cov(f, f)

        return StressDist(piola_mean, piola_var)
    
    def psi_dist(self, f):
        no_f = jnp.eye(f.shape[-1])
        E = 0.5 * (f.T @ f - no_f)

        psi_mean = self.psi_gp_mean(f) - self.psi_gp_mean(no_f) - jnp.sum(self.piola_gp_dist(no_f).mean * E)
        psi_var = self.psi_gp_cov(f, f) + self.psi_gp_cov(no_f, no_f) - 2 * self.psi_gp_cov(f, no_f) 
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
        # dist = self.piola_dist(f)
        # if key is None :
        #     return dist.mean
        # else :
        #     return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)
        return jax.grad(lambda f : self.psi(f, key))(f)
    

    def psi_quadrature(self, f_tensor, x_node):

        dist = self.psi_dist(f_tensor)
        
        mean = dist.mean
        var = dist.var
        
        std = jnp.maximum(jnp.sqrt(var), 1e-9)
        
        psi = mean + jnp.sqrt(2) * std * x_node
        
        return psi


    
    def kl_divergance(self):
        def component_kl(ma_term, log_term, trace_term, M):
            return 0.5 * (log_term - M + trace_term + ma_term)
        dev_kl = component_kl(self.gpweight.dev_mahalanobis_term, self.gpweight.dev_logterm, self.gpweight.dev_trace_term, self.dev_z.shape[0])
        vol_kl = component_kl(self.gpweight.vol_mahalanobis_term, self.gpweight.vol_logterm, self.gpweight.vol_trace_term, self.vol_z.shape[0])
        return dev_kl + vol_kl
#     def _predict_gp_component(self, features, part: str):
#         """Refactored component predictor (Dev or Vol)"""
#         z = self.inducing_points[f"{part}_z"]
#         ls = self.params[f"{part}_gp_lengthscales"]
#         sig = self.params[f"{part}_gp_sigma_scaling"]
#         m_u = self.params[f"{part}_u_mean"]
#         S_uu = jnp.diag(self.params[f"{part}_u_var"])
        
#         Kiz = discovery_kernel(features, z, sig, ls)
#         Kii = discovery_kernel(features, features, sig, ls)
#         Kzz = self.precomputed_weights[f"{part}_Kzz"]
#         Kzz_inv = self.precomputed_weights[f"{part}_Kzz_inv"]

#         mu_trend_star = self._trend_fn(features, part) 
#         # Trend at the inducing points (Z)
#         mu_trend_z = jax.vmap(self._trend_fn, in_axes=(0, None))(z, part) 
        
#         # 4. MEAN CALCULATION (Prior Mean + GP Correction)
#         # mu = trend(x*) + Kiz @ Kzz_inv @ (u_mean - trend(z))
#         # with prior update
#         residual_at_z = m_u - mu_trend_z
#         gp_correction = Kiz @ Kzz_inv @ residual_at_z
#         mean = mu_trend_star + gp_correction
#         # not update prior #
#         # gp_mean = Kiz @ Kzz_inv @ (m_u)
#         # mu_trend_z = jax.vmap(self._trend_fn, in_axes=(0, None))(z, part) 
    
#         # mu_trend_star = self._trend_fn(features, part) 
#         # mean = mu_trend_star + gp_mean
        
#         # 5. VARIANCE CALCULATION
#         # v_standard = Kii - Kiz @ Kzz_inv @ Kiz.T (Epistemic uncertainty of GP)
#         # v_learned = Kiz @ Kzz_inv @ S_uu @ Kzz_inv @ Kiz.T (Uncertainty of inducing variables)
#         v_standard = Kii - Kiz @ Kzz_inv @ Kiz.T
#         v_learned = Kiz @ Kzz_inv @ S_uu @ Kzz_inv @ Kiz.T
        
#         # Total variance is the sum of standard GP variance and learned uncertainty
#         var = v_standard + v_learned
        
#         return EnergyDist(mean.squeeze(), var.squeeze())

#     def piola(self, F, key) :
#         return jax.grad(lambda f : self.psi(f, key))(F)
#         # dist = self.psi(F)
#         # if key is None :
#         #     return dist.mean
#         # else :
#         #     return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)
        
#     def piola_dist(self, deformation_gradient) :
#         piola_gp_dist = self._piola_gp_dist(deformation_gradient)
#         piola_ref_dist = self._piola_gp_dist(jnp.eye(deformation_gradient.shape[-1])) 
#         # var4th = self._get_piola_gp_cov(deformation_gradient, jnp.eye(deformation_gradient.shape[-1]))
#         # var2nd = jnp.einsum('ijij->ij', var4th)
#         var = piola_gp_dist.var + deformation_gradient @ piola_ref_dist.var @ deformation_gradient.T
#         return StressDist(piola_gp_dist.mean - deformation_gradient @ piola_ref_dist.mean, var)
#     def _get_piola_gp_cov(self, f1, f2) :
#         stress_cov_tensor = jax.jacfwd(jax.jacrev(self._get_psi_gp_cov, argnums=1), argnums=0)(
#                 f1, f2
#             )
#         return stress_cov_tensor
#     def _piola_gp_dist(self, deformation_gradient) :
#         piola_gp_mean = jax.grad(lambda f : self._internal_energy_dist(f).mean)(deformation_gradient)
#         stress_cov_tensor = jax.jacfwd(jax.jacrev(self._get_psi_gp_cov, argnums=1), argnums=0)(
#                 deformation_gradient, deformation_gradient
#             )
            
#             # Extract the variance for each component P_ij
#             # Cov(P_ij, P_ij) is the diagonal of the 4th order tensor
#         piola_var = jnp.einsum('ijij->ij', stress_cov_tensor)
#         return StressDist(piola_gp_mean, piola_var)
    
#     def psi_dist(self, F):
#         ref_f = jnp.eye(F.shape[-1])
#         E = 0.5 * (F.T @ F - ref_f)
        
#         # 1. Means (These are fine as they are)
#         mu_f = self._internal_energy_dist(F).mean
#         mu_ref = self._internal_energy_dist(ref_f).mean
#         p_ref = self._piola_gp_dist(ref_f).mean
#         norm_mean = mu_f
#         base_var = self._internal_energy_dist(F).var
#         # total_var = base_var
#         norm_mean = mu_f - mu_ref - jnp.sum(p_ref * E)

#         # # 2. Correct Variance using the full Covariance Formula
#         # # Var(Psi_norm) = Var(F) + Var(Ref) - 2*Cov(F, Ref) + Var(Stress_Correction)
#         # # Plus cross-terms if you want to be perfectly rigorous.
        
#         var_f = self._get_psi_gp_cov(F, F)
#         var_ref = self._get_psi_gp_cov(ref_f, ref_f)
#         cov_f_ref = self._get_psi_gp_cov(F, ref_f)
        
#         # # Standard GP variance for (Psi(F) - Psi(I))
#         base_var = var_f + var_ref - 2 * cov_f_ref
        
#         # # Add stress correction variance (Simplified approximation)
#         # # In a rigorous setup, you'd also need Cov(Psi, Grad_Psi)
#         piola_ref_var = self._piola_gp_dist(ref_f).var
#         psi_stress_ref_var = jnp.sum(piola_ref_var * (E @ E)) # Corrected E scaling
#         total_var = base_var + psi_stress_ref_var
#         total_var = jnp.maximum(total_var, 1e-9)
#         return EnergyDist(norm_mean, total_var)
    
#     def psi(self, F, key = None):
#         dist = self.psi_dist(F)
    
#         if key is None :
#             return dist.mean
#         else :
#             return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)

#     def _internal_energy(self, F, key = None) :
#         dist = self._internal_energy_dist(F)
#         if key is None :
#             return dist.mean
#         else :
#             return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)


#     def _internal_energy_dist(self, F):
#         i_star, _ = invariants_and_derivatives(F)
#         dev_f, vol_f = transform_input_features(i_star)
        
#         dist_dev = self._predict_gp_component(dev_f, "dev")
#         dist_vol = self._predict_gp_component(vol_f, "vol")
        
#         # Add trends
#         total_mean = dist_dev.mean + dist_vol.mean
# #
#         return EnergyDist(mean = total_mean, var = dist_vol.var + dist_dev.var)

#     def _trend_fn(self, features, part):
#         if part == "dev":
#             return self._dev_mean(features)
#         elif part == "vol":
#             return self._vol_mean(features)

#     def _dev_mean(self, dev_f):
#         p = self.params
#         dev_t =  1.00685 * (dev_f[0] - 3)**2 + 0.47095 * (dev_f[0] - 3) + 1.0657 * (dev_f[1] - 3)
# # 0.47095, 1.0657 , 1.00685, 0.0185 , 0.0069 , 1.51745
#         # dev_t = p["c20"] * (dev_f[0] - 3)**2 + p["c02"]*(dev_f[1]-3)**2 + p["c11"]*(dev_f[0] - 3)*(dev_f[1] - 3) + p["c10"] * (dev_f[0] - 3) + p["c01"] * (dev_f[1] - 3)
#         return dev_t

#     def _vol_mean(self, vol_f):
#         p = self.params
#         vol_t = 1.51745*(vol_f[0] - 1)**2

#         # vol_t = p["k"]*(vol_f[0] - 1)**2 + p["q"] *jnp.log(vol_f[0])**2
#         return vol_t 


#     def _get_psi_gp_cov(self, F1, F2):
#         # 1. Compute Invariants for both inputs
#         # i_star should return (I1, I2, J)
#         i1_star, _ = invariants_and_derivatives(F1)
#         i2_star, _ = invariants_and_derivatives(F2)
#         dev1, vol1 = transform_input_features(i1_star)
#         dev2, vol2 = transform_input_features(i2_star)

#         def sparse_cov(x1, x2, z, sigma, ls, S_uu, part = "dev"):
#             # k_x1z: (1, M), k_zx2: (M, 1), k_x1x2: (1, 1)
#             Kzz = discovery_kernel(z, z, sigma, ls) + 1e-6 * jnp.eye(z.shape[0])
#             Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))
#             Kzz_inv = self.precomputed_weights[f"{part}_Kzz_inv"]

#             k_x1z = discovery_kernel(x1[None, :], z, sigma, ls)
#             k_zx2 = discovery_kernel(z, x2[None, :], sigma, ls)
#             k_x1x2 = discovery_kernel(x1[None, :], x2[None, :], sigma, ls)
#             # term1: Kiz @ Kzz_inv @ Kzi (Standard GP variance reduction)
#             term1 = k_x1z @ Kzz_inv @ k_zx2
#             # term2: Kiz @ (Kzz_inv @ S @ Kzz_inv) @ Kzi (Variational uncertainty)
#             term2 = k_x1z @ Kzz_inv @ S_uu @ Kzz_inv @ k_zx2
            
#             return (k_x1x2 - term1 + term2).reshape()
#         # Deviatoric part

#         cov_dev = sparse_cov(
#             dev1, dev2, self.inducing_points["dev_z"], 
#             self.params["dev_gp_sigma_scaling"], self.params["dev_gp_lengthscales"],
#             jnp.diag(self.params["dev_u_var"]), "dev"
#         )

#         # Volumetric part
#         cov_vol = sparse_cov(
#             vol1, vol2, self.inducing_points["vol_z"], 
#             self.params["vol_gp_sigma_scaling"], self.params["vol_gp_lengthscales"],
#             jnp.diag(self.params["vol_u_var"]), "vol"
#         )

#         return cov_dev + cov_vol

    # def load_params(self, params) :
    #     check1 = jnp.std(self.inducing_points["dev_z"], axis = 0)
    #     check2 = jnp.std(self.inducing_points["vol_z"], axis = 0)
    #     # params = {
    #     #     "dev_gp_lengthscales" : jnp.std(self.inducing_points["dev_z"], axis = 0) + enforce_softplus_positive(params["raw_dev_gp_lengthscales"]), 
    #     #     "vol_gp_lengthscales" : jnp.std(self.inducing_points["vol_z"], axis = 0) + enforce_softplus_positive(params["raw_vol_gp_lengthscales"]), 
    #     #     "dev_gp_sigma_scaling" : jnp.exp(params["raw_dev_gp_sigma_scaling"]),
    #     #     "vol_gp_sigma_scaling" : jnp.exp(params["raw_vol_gp_sigma_scaling"]),
    #     #     "dev_u_mean" : enforce_softplus_positive(params["raw_dev_u_mean"]),
    #     #     "dev_u_var" : enforce_softplus_positive(params["raw_dev_u_var"]),
    #     #     "vol_u_mean" : enforce_softplus_positive(params["raw_vol_u_mean"]),
    #     #     "vol_u_var" : enforce_softplus_positive(params["raw_vol_u_var"]),
    #     #     "c20": enforce_softplus_positive(params["raw_c20"]),
    #     #     "c02": enforce_softplus_positive(params["raw_c02"]),
    #     #     "c11": enforce_softplus_positive(params["raw_c11"]),
    #     #     "c10": enforce_softplus_positive(params["raw_c10"]),
    #     #     "c01": enforce_softplus_positive(params["raw_c01"]),
    #     #     "k": enforce_softplus_positive(params["raw_k"]),
    #     #     "q": enforce_softplus_positive(params["raw_q"]),
    #     # }
    #     # not update prior
    #     # params = {
    #     #     "dev_gp_lengthscales" : jnp.std(self.inducing_points["dev_z"], axis = 0) + enforce_softplus_positive(params["raw_dev_gp_lengthscales"]), 
    #     #     "vol_gp_lengthscales" : jnp.std(self.inducing_points["vol_z"], axis = 0) + enforce_softplus_positive(params["raw_vol_gp_lengthscales"]), 
    #     #     "dev_gp_sigma_scaling" : jnp.exp(params["raw_dev_gp_sigma_scaling"]),
    #     #     "vol_gp_sigma_scaling" : jnp.exp(params["raw_vol_gp_sigma_scaling"]),
    #     #     "dev_u_mean" : params["raw_dev_u_mean"],
    #     #     "dev_u_var" : enforce_softplus_positive(params["raw_dev_u_var"]),
    #     #     "vol_u_mean" : params["raw_vol_u_mean"],
    #     #     "vol_u_var" : enforce_softplus_positive(params["raw_vol_u_var"]),
    #     #     "c20": enforce_softplus_positive(params["raw_c20"]),
    #     #     "c02": enforce_softplus_positive(params["raw_c02"]),
    #     #     "c11": enforce_softplus_positive(params["raw_c11"]),
    #     #     "c10": enforce_softplus_positive(params["raw_c10"]),
    #     #     "c01": enforce_softplus_positive(params["raw_c01"]),
    #     #     "k": enforce_softplus_positive(params["raw_k"]),
    #     #     "q": enforce_softplus_positive(params["raw_q"]),
    #     # }
    #     return params




class TensorBasisSVGP :
    def __init__(self, params, I_obs, n_ip) :
        self.params = self.load_params(params)
        self.inducing_points = {"z" : I_obs[farthest_point_sampling(I_obs, n_ip)]}
        self.precomputed_weights = {}
        self.precompute_weights()
    
    def load_params(self, params) :
        params = {
            "a1_gp_lengthscales" : jnp.exp(params["raw_a1_gp_lengthscales"]), 
            "a1_gp_sigma_scaling" : jnp.exp(params["raw_a1_gp_sigma_scaling"]),
            "a1_u_mean" : params["raw_a1_u_mean"],
            "a1_u_var" : enforce_softplus_positive(params["raw_a1_u_var"]),

            "a2_gp_lengthscales" : jnp.exp(params["raw_a2_gp_lengthscales"]),
            "a2_gp_sigma_scaling" : jnp.exp(params["raw_a2_gp_sigma_scaling"]),
            "a2_u_mean" : params["raw_a2_u_mean"],
            "a2_u_var" : enforce_softplus_positive(params["raw_a2_u_var"]),

            "a3_gp_lengthscales" : jnp.exp(params["raw_a3_gp_lengthscales"]), 
            "a3_gp_sigma_scaling" : jnp.exp(params["raw_a3_gp_sigma_scaling"]),
            "a3_u_mean" : params["raw_a3_u_mean"],
            "a3_u_var" : enforce_softplus_positive(params["raw_a3_u_var"])
        }
        return params
    
    def precompute_weights(self) :
        for alpha in ["a1", "a2", "a3"] :
            z = self.inducing_points["z"]
            ls = self.params[f"{alpha}_gp_lengthscales"]
            sig = self.params[f"{alpha}_gp_sigma_scaling"]
            Kzz = discovery_kernel(z, z, sig, ls) + 1e-6 * jnp.eye(z.shape[0])

            self.precomputed_weights[f"{alpha}_Kzz"] = Kzz
            self.precomputed_weights[f"{alpha}_Kzz_inv"] = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))

    def _gp_component(self, input, alpha) :
        # Kzz = self.precomputed_weights[f"{alpha}_Kzz"]
        Kzz_inv = self.precomputed_weights[f"{alpha}_Kzz_inv"]
        z = self.inducing_points["z"]
        u_mean = self.params[f"{alpha}_u_mean"]
        u_var = self.params[f"{alpha}_u_var"]
        S_uu = jnp.diag(u_var)


        kiz = discovery_kernel(input, z, self.params[f"{alpha}_gp_sigma_scaling"], self.params[f"{alpha}_gp_lengthscales"])
        kii = discovery_kernel(input, input, self.params[f"{alpha}_gp_sigma_scaling"], self.params[f"{alpha}_gp_lengthscales"])
        
        gp_mean = kiz @  Kzz_inv @ u_mean 
        # gp_var = jnp.maximum(kii - kiz @ Kzz_inv @ (self.precomputed_weights[f"{alpha}_Kzz"] - S_uu) @ Kzz_inv @ kiz.T, 1e-9)
        gp_var = 0
        return EnergyDist(gp_mean, gp_var)
    
    def kl_divergance(self) :
        params = self.params

        def kl(u_mean, u_var, z, part) :
            Kzz = self.precomputed_weights[f"{part}_Kzz"]

            Kzz_inv = self.precomputed_weights[f"{part}_Kzz_inv"]

            trace_term = jnp.trace(Kzz_inv @ jnp.diag(u_var))
            mahalanobis_term = (u_mean) @ Kzz_inv @ (u_mean).T
            
            log_det_K = jnp.log(jnp.linalg.det(Kzz))
            log_det_S = jnp.log(jnp.linalg.det(jnp.diag(u_var)))
            
            M = z.shape[0]
            return 0.5 * (trace_term + mahalanobis_term - M + log_det_K - log_det_S)

        total_kl = 0.0
        for alpha in ["a1", "a2", "a3"] :
            z = self.inducing_points["z"]
            kl_ = kl(params[f"{alpha}_u_mean"], params[f"{alpha}_u_var"], z, alpha)
            total_kl += kl_
        return total_kl
    
    def piola_gp_dist(self, f) :
        invariants, _ = invariants_and_derivatives(f)
        c = f.T @ f
        a1 = self._gp_component(invariants, "a1")
        a2 = self._gp_component(invariants, "a2")
        a3 = self._gp_component(invariants, "a3")
        
        fT_inv = jnp.linalg.inv(f.T)

        piola_gp_mean = (a1.mean * jnp.eye(f.shape[-1]) + a2.mean * c + a3.mean * c **2) @ fT_inv
        piola_gp_var = (a1.var * jnp.eye(f.shape[-1]) + a2.var * c + a3.var * c **2) @ fT_inv

        return StressDist(piola_gp_mean, piola_gp_var)
    def piola_dist(self, f) :
        ref_f = jnp.eye(f.shape[-1])
        piola_gp_dist = self.piola_gp_dist(f)
        piola_ref_dist = self.piola_gp_dist(ref_f)
        piola_mean = piola_gp_dist.mean - piola_ref_dist.mean
        piola_var = piola_gp_dist.var + piola_ref_dist.var
        return StressDist(piola_mean, piola_var)
    def piola(self, f, key) :
        dist = self.piola_dist(f)
        if key is None :
            return dist.mean
        else :
            return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)
