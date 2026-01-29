# gpjax_version.py
import os
# from flax.serialization import to_state_dict, from_state_dict
from core.utils import *
from core.datasetclass import *
import jax.numpy as jnp
import jax.random as jr
import pickle
key = jr.key(123)
from typing import NamedTuple
import jax.numpy as jnp
import jax
# def input_normalize()

def farthest_point_sampling(pts, num_samples):
    """
    pts: (N, 3) array of points
    num_samples: 25
    """
    n_pts = pts.shape[0]
    # Initialize: pick the first point in the list as the start
    selected_indices = jnp.zeros(num_samples, dtype=jnp.int32)
    
    # Track the distance from every point to its NEAREST selected point
    # Start with infinity
    dist_to_set = jnp.full((n_pts,), jnp.inf)
    
    def scan_body(dist_to_set, i):
        # The next point is the one farthest from the current set
        idx = jnp.argmax(dist_to_set)
        
        # Calculate distance from the new point to all other points
        new_pt = pts[idx]
        dists = jnp.sum((pts - new_pt)**2, axis=-1) # Squared Euclidean
        
        # Update distances: dist to set is min(old_dist, dist_to_new_point)
        dist_to_set = jnp.minimum(dist_to_set, dists)
        
        return dist_to_set, idx

    # We manually pick the first point to start
    first_idx = 0
    dist_to_set = jnp.sum((pts - pts[first_idx])**2, axis=-1)
    
    # Run the loop for the remaining 24 points
    _, remaining_indices = jax.lax.scan(scan_body, dist_to_set, jnp.arange(1, num_samples))
    
    return jnp.concatenate([jnp.array([first_idx]), remaining_indices])



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

from typing import NamedTuple
import jax.numpy as jnp
import jax

class GPParams(NamedTuple):
    # Deviatoric GP
    dev_u_mean: jnp.ndarray
    dev_u_var_raw: jnp.ndarray  # Store raw for optimization
    dev_ls: jnp.ndarray
    dev_sig: jnp.ndarray
    
    # Volumetric GP
    vol_u_mean: jnp.ndarray
    vol_u_var_raw: jnp.ndarray
    vol_ls: jnp.ndarray
    vol_sig: jnp.ndarray
    
    # Trend Coefficients [p, q, r, s, t, a]
    poly_coeffs: jnp.ndarray

class GPInducingPoints(NamedTuple):
    # Inducing points are static (fixed after FPS)

    dev_z: jnp.ndarray
    vol_z: jnp.ndarray

class GPPrecomputed(NamedTuple):
    # Precomputed matrices for the current step
    dev_Kinv: jnp.ndarray
    dev_Kinv_S_Kinv: jnp.ndarray
    vol_Kinv: jnp.ndarray
    vol_Kinv_S_Kinv: jnp.ndarray
    dev_alpha: jnp.ndarray
    vol_alpha: jnp.ndarray

class SparseHyperelasticityGP :
    '''get g latent energy which will be joint gaussian with latent energy at testpoint'''
    def __init__(self, params, I_z) :
    
        # #  = I_obs[farthest_point_sampling(I_obs, n_ip)]
        # # dev_z = I_obs_dev[farthest_point_sampling(I_obs_dev, n_ip)]
        # # vol_z = j[farthest_point_sampling(j, n_ip)]
        # dev_obs, vol_obs = jax.vmap(transform_input_features)(I_obs)

        self.inducing_points = dict(
            dev_z=I_z[:, :2],
            vol_z=I_z[:, 2:]
        )
        # self.inducing_points["aug_dev_z"] = jnp.concat([jnp.array([[3.0, 3.0]]), self.inducing_points["dev_z"]])
        # self.inducing_points["aug_vol_z"] = jnp.concat([jnp.array([[1.0]]), self.inducing_points["vol_z"]])

        self.params = self.load_params(params)
        self.precomputed_weights = {}
        self.precompute_weights()

    def precompute_weights(self) :
        for alpha in ["dev", "vol"] :
            z = self.inducing_points[f"{alpha}_z"]
            # u_mean = self.params[f"{alpha}_u_mean"]
            # u_var = self.params[f"{alpha}_u_var"]
            ls = self.params[f"{alpha}_gp_lengthscales"]
            sig = self.params[f"{alpha}_gp_sigma_scaling"]
            Kzz = discovery_kernel(z, z, sig, ls) + 1e-6 * jnp.eye(z.shape[0])
            self.params[f"{alpha}_Kzz"] = Kzz
            self.precomputed_weights[f"{alpha}_Kzz"] = Kzz
            self.precomputed_weights[f"{alpha}_Kzz_inv"] = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))
    # def piola(self, F, key) :
    #     if key is None :
    #         return self.piola_dist(F).mean
    #     else :
    #         return jax.grad(lambda f: self.psi(f, key))(F)
    # def psi(self, f, key) :
    #     ref_f = jnp.eye(f.shape[-1])
    #     E = 0.5 * (f.T @ f - ref_f)
    #     def psi_sam(f, key) :
    #         psi_gp_dist = self.psi_gp_dist(f)
    #         if key is None :
    #             return psi_gp_dist.mean
    #         else : 
    #             return psi_gp_dist.mean + jnp.sqrt(psi_gp_dist.var) * jax.random.normal(key, psi_gp_dist.mean.shape)
    #     H = jax.grad(lambda f: psi_sam(f, key))(ref_f)
    #     return psi_sam(f, key) - psi_sam(ref_f, key) - jnp.sum(H * E)

    # def piola_dist(self, f) :
    #     ref_f = jnp.eye(f.shape[-1])
    #     piola_gp_dist = self.piola_gp_dist(f)
    #     piola_ref_dist = self.piola_gp_dist(ref_f)
    
    #     piola_mean = piola_gp_dist.mean - f @ piola_ref_dist.mean
    #     piola_var = piola_gp_dist.var + f @ piola_ref_dist.var
    #     return StressDist(piola_mean, piola_var)
    
    # def psi_dist(self, f) :
    #     ref_f = jnp.eye(f.shape[-1])
    #     E = 0.5 * (f.T @ f - ref_f)
    #     psi_gp_dist = self.psi_gp_dist(f)
    #     psi_ref_dist = self.psi_gp_dist(ref_f)
        
    #     piola_gp_ref_dist = self.piola_gp_dist(ref_f)

    #     norm_mean = psi_gp_dist.mean - psi_ref_dist.mean - jnp.sum(piola_gp_ref_dist.mean * E)
    #     norm_var = psi_gp_dist.var + psi_ref_dist.var + jnp.sum(piola_gp_ref_dist.var * E)
    #     return EnergyDist(norm_mean, norm_var)
    
    # def piola_gp_dist(self, f) :
    #     piola_gp_mean = jax.grad(self._predict_psi_gp_mean)(f)

    #     def get_diagonal_stress_var(f):
    #         # Flatten f to (9,) to work with a (9,9) Hessian more easily
    #         f_flat = f.reshape(-1)
            
    #         def scalar_cov_flat(flat_f1, flat_f2):
    #             return self._predict_psi_gp_covariance(flat_f1.reshape(3,3), flat_f2.reshape(3,3))

    #         # Calculate the full 9x9 Hessian and immediately take the diagonal
    #         # This results in a (9,) vector of variances for [P11, P12, P13, P21...]
    #         hessian_9x9 = jax.hessian(scalar_cov_flat, argnums=0)(f_flat, f_flat)
    #         diag_vars = jnp.diag(hessian_9x9)
            
    #         return diag_vars.reshape(3, 3)


    #     piola_gp_var = get_diagonal_stress_var(f)
    #     return StressDist(piola_gp_mean, piola_gp_var)
    
    # def psi_gp_dist(self, f):

    #     mean = self._predict_psi_gp_mean(f)
    #     variance = self._predict_psi_gp_covariance(f, f)
        
    #     return EnergyDist(mean = mean, var = variance)
            
    
    # def _predict_psi_gp_mean(self, f) :
    #     i, _ = invariants_and_derivatives(f)
    #     dev, vol = transform_input_features(i)

    #     dev_z = self.inducing_points[f"dev_z"]
    #     dev_ls = self.params[f"dev_gp_lengthscales"]
    #     dev_sig = self.params[f"dev_gp_sigma_scaling"]
    #     dev_m_u = self.params[f"dev_u_mean"]
        
    #     vol_z = self.inducing_points[f"vol_z"]
    #     vol_ls = self.params[f"vol_gp_lengthscales"]
    #     vol_sig = self.params[f"vol_gp_sigma_scaling"]
    #     vol_m_u = self.params[f"vol_u_mean"]

    #     def mean_fn(features, z, Kzz_inv, m_u, sigma, ls, part) :
    #         mu_trend_z = jax.vmap(self._trend_fn, in_axes=(0, None))(z, part) 
    #         Kiz = discovery_kernel(features, z, sigma, ls)
    #         return self._trend_fn(features, part) + Kiz @ Kzz_inv @ (m_u - mu_trend_z)
    #     return (mean_fn(dev, dev_z, self.precomputed_weights[f"dev_Kzz_inv"], dev_m_u, dev_sig, dev_ls, "dev") + mean_fn(vol, vol_z, self.precomputed_weights[f"vol_Kzz_inv"], vol_m_u, vol_sig, vol_ls, "vol")).squeeze()


    # def _predict_psi_gp_covariance(self, f1, f2) :
    #     i1, _ = invariants_and_derivatives(f1)
    #     i2, _ = invariants_and_derivatives(f2)
    #     dev_1, vol_1 = transform_input_features(i1)
    #     dev_2, vol_2 = transform_input_features(i2)

    #     # 2. Extract Params
    #     dev_z = self.inducing_points[f"dev_z"]
    #     dev_ls = self.params[f"dev_gp_lengthscales"]
    #     dev_sig = self.params[f"dev_gp_sigma_scaling"]
    #     dev_S_uu = jnp.diag(self.params[f"dev_u_var"])
    #     dev_Kzz_inv = self.precomputed_weights[f"dev_Kzz_inv"]

    #     vol_z = self.inducing_points[f"vol_z"]
    #     vol_ls = self.params[f"vol_gp_lengthscales"]
    #     vol_sig = self.params[f"vol_gp_sigma_scaling"]
    #     vol_S_uu = jnp.diag(self.params[f"vol_u_var"])
    #     vol_Kzz_inv = self.precomputed_weights[f"vol_Kzz_inv"]



    #     def covariance_fn(feat1, feat2, z, Kzz_inv, S_uu, sigma, ls) :
    #         # 3. Compute Kernel Blocks
    #         k12 = discovery_kernel(feat1, feat2, sigma, ls)  # Prior k(x, x')
    #         k1z = discovery_kernel(feat1, z, sigma, ls)      # k(x, z)
    #         kz2 = discovery_kernel(z, feat2, sigma, ls)      # k(z, x')
    #         v_standard = k12 - k1z @ Kzz_inv @ kz2
    #         v_learned = k1z @ Kzz_inv @ S_uu @ Kzz_inv @ kz2
    #         return jnp.maximum(v_standard + v_learned, 1e-9).squeeze()
    #     psi_gp_covariance = covariance_fn(dev_1, dev_2, dev_z, dev_Kzz_inv, dev_S_uu, dev_sig, dev_ls) + covariance_fn(vol_1, vol_2, vol_z, vol_S_uu, vol_Kzz_inv, vol_sig, vol_ls)
    #     return psi_gp_covariance
    
    def piola_quadrature(self, f_tensor, x_node):
        """
        Computes a deterministic stress point for Gauss-Hermite quadrature.
        """
        # 1. Get the predictive distribution (Mean: 3x3, Var: 3x3 diagonal)
        # This must include the subtraction of the reference state
        dist = self.piola_dist(f_tensor) 
        
        # 2. Extract mean and standard deviation
        # We use jnp.maximum to ensure numerical stability (no negative variance)
        p_mean = dist.mean
        p_std = jnp.sqrt(jnp.maximum(dist.var, 1e-9))
        
        # 3. Transform the GH node into stress space
        # Formula: P_j = mu + sqrt(2) * sigma * x_j
        # This maps the standard normal node to your specific GP uncertainty
        p_quad = p_mean + jnp.sqrt(2.0) * p_std * x_node
        
        return p_quad
    def kl_divergance(self) :
        params = self.params

        def kl(u_mean, u_var, z, trend, part) :
            Kzz = self.precomputed_weights[f"{part}_Kzz"]

            Kzz_inv = self.precomputed_weights[f"{part}_Kzz_inv"]

            trace_term = jnp.trace(Kzz_inv @ jnp.diag(u_var))
            # update prior
            mahalanobis_term = (u_mean - trend) @ Kzz_inv @ (u_mean - trend).T
            # not update prior
            # mahalanobis_term = (u_mean) @ Kzz_inv @ (u_mean).T

            log_det_K = jnp.log(jnp.linalg.det(Kzz))
            log_det_S = jnp.log(jnp.linalg.det(jnp.diag(u_var)))
            
            M = z.shape[0]
            return 0.5 * (trace_term + mahalanobis_term - M + log_det_K - log_det_S)

        total_kl = 0.0
        for alpha in ["dev", "vol"] :
            z = self.inducing_points[f"{alpha}_z"]
            mu = jax.vmap(self._trend_fn, in_axes=(0, None))(z, alpha) 
            kl_ = kl(params[f"{alpha}_u_mean"], params[f"{alpha}_u_var"], z, mu, alpha)
            total_kl += kl_
        return total_kl
    
    def _predict_gp_component(self, features, part: str):
        """Refactored component predictor (Dev or Vol)"""
        z = self.inducing_points[f"{part}_z"]
        ls = self.params[f"{part}_gp_lengthscales"]
        sig = self.params[f"{part}_gp_sigma_scaling"]
        m_u = self.params[f"{part}_u_mean"]
        S_uu = jnp.diag(self.params[f"{part}_u_var"])
        
        Kiz = discovery_kernel(features, z, sig, ls)
        Kii = discovery_kernel(features, features, sig, ls)
        Kzz = self.precomputed_weights[f"{part}_Kzz"]
        Kzz_inv = self.precomputed_weights[f"{part}_Kzz_inv"]

        mu_trend_star = self._trend_fn(features, part) 
        # Trend at the inducing points (Z)
        mu_trend_z = jax.vmap(self._trend_fn, in_axes=(0, None))(z, part) 
        
        # 3. Solve for Kzz_inv
        # Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))
        
        # 4. MEAN CALCULATION (Prior Mean + GP Correction)
        # mu = trend(x*) + Kiz @ Kzz_inv @ (u_mean - trend(z))
        # with prior update
        residual_at_z = m_u - mu_trend_z
        gp_correction = Kiz @ Kzz_inv @ residual_at_z
        mean = mu_trend_star + gp_correction
        # not update prior #
        # gp_mean = Kiz @ Kzz_inv @ (m_u)
        # mu_trend_z = jax.vmap(self._trend_fn, in_axes=(0, None))(z, part) 
        
        # mu_trend_star = self._trend_fn(features, part) 
        # mean = mu_trend_star + gp_mean
        
        # 5. VARIANCE CALCULATION
        # v_standard = Kii - Kiz @ Kzz_inv @ Kiz.T (Epistemic uncertainty of GP)
        # v_learned = Kiz @ Kzz_inv @ S_uu @ Kzz_inv @ Kiz.T (Uncertainty of inducing variables)
        v_standard = Kii - Kiz @ Kzz_inv @ Kiz.T
        v_learned = Kiz @ Kzz_inv @ S_uu @ Kzz_inv @ Kiz.T
        
        # Total variance is the sum of standard GP variance and learned uncertainty
        var = v_standard + v_learned
        
        return EnergyDist(mean.squeeze(), var.squeeze())

    def piola(self, F, key) :
        return jax.grad(lambda f : self.psi(f, key))(F)
        # dist = self.psi(F)
        # if key is None :
        #     return dist.mean
        # else :
        #     return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)
        
    def piola_dist(self, deformation_gradient) :
        piola_gp_dist = self._piola_gp_dist(deformation_gradient)
        piola_ref_dist = self._piola_gp_dist(jnp.eye(deformation_gradient.shape[-1])) 
        # var4th = self._get_piola_gp_cov(deformation_gradient, jnp.eye(deformation_gradient.shape[-1]))
        # var2nd = jnp.einsum('ijij->ij', var4th)
        var = piola_gp_dist.var + deformation_gradient @ piola_ref_dist.var @ deformation_gradient.T
        return StressDist(piola_gp_dist.mean - deformation_gradient @ piola_ref_dist.mean, var)
    def _get_piola_gp_cov(self, f1, f2) :
        stress_cov_tensor = jax.jacfwd(jax.jacrev(self._get_psi_gp_cov, argnums=1), argnums=0)(
                f1, f2
            )
        return stress_cov_tensor
    def _piola_gp_dist(self, deformation_gradient) :
        piola_gp_mean = jax.grad(lambda f : self._internal_energy_dist(f).mean)(deformation_gradient)
        stress_cov_tensor = jax.jacfwd(jax.jacrev(self._get_psi_gp_cov, argnums=1), argnums=0)(
                deformation_gradient, deformation_gradient
            )
            
            # Extract the variance for each component P_ij
            # Cov(P_ij, P_ij) is the diagonal of the 4th order tensor
        piola_var = jnp.einsum('ijij->ij', stress_cov_tensor)
        return StressDist(piola_gp_mean, piola_var)
    
    def psi_dist(self, F):
        ref_f = jnp.eye(F.shape[-1])
        E = 0.5 * (F.T @ F - ref_f)
        
        # 1. Means (These are fine as they are)
        mu_f = self._internal_energy_dist(F).mean
        mu_ref = self._internal_energy_dist(ref_f).mean
        p_ref = self._piola_gp_dist(ref_f).mean
        norm_mean = mu_f
        base_var = self._internal_energy_dist(F).var
        # total_var = base_var
        norm_mean = mu_f - mu_ref - jnp.sum(p_ref * E)

        # # 2. Correct Variance using the full Covariance Formula
        # # Var(Psi_norm) = Var(F) + Var(Ref) - 2*Cov(F, Ref) + Var(Stress_Correction)
        # # Plus cross-terms if you want to be perfectly rigorous.
        
        var_f = self._get_psi_gp_cov(F, F)
        var_ref = self._get_psi_gp_cov(ref_f, ref_f)
        cov_f_ref = self._get_psi_gp_cov(F, ref_f)
        
        # # Standard GP variance for (Psi(F) - Psi(I))
        base_var = var_f + var_ref - 2 * cov_f_ref
        
        # # Add stress correction variance (Simplified approximation)
        # # In a rigorous setup, you'd also need Cov(Psi, Grad_Psi)
        piola_ref_var = self._piola_gp_dist(ref_f).var
        psi_stress_ref_var = jnp.sum(piola_ref_var * (E @ E)) # Corrected E scaling
        total_var = base_var + psi_stress_ref_var
        return EnergyDist(norm_mean, total_var)
    
    def psi(self, F, key = None):
        dist = self.psi_dist(F)
    
        if key is None :
            return dist.mean
        else :
            return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)

    def _internal_energy(self, F, key = None) :
        dist = self._internal_energy_dist(F)
        if key is None :
            return dist.mean
        else :
            return dist.mean + jax.random.normal(key, dist.mean.shape) * jnp.sqrt(dist.var)


    def _internal_energy_dist(self, F):
        i_star, _ = invariants_and_derivatives(F)
        dev_f, vol_f = transform_input_features(i_star)
        
        dist_dev = self._predict_gp_component(dev_f, "dev")
        dist_vol = self._predict_gp_component(vol_f, "vol")
        
        # Add trends
        total_mean = dist_dev.mean + dist_vol.mean
#
        return EnergyDist(mean = total_mean, var = dist_vol.var + dist_dev.var)

    def _trend_fn(self, features, part):
        if part == "dev":
            return self._dev_mean(features)
        elif part == "vol":
            return self._vol_mean(features)

    def _dev_mean(self, dev_f):
        p = self.params
        # dev_t = 1 * (dev_f[0] - 3)**2 + 0.5 * (dev_f[0] - 3) + 1 * (dev_f[1] - 3)

        dev_t = p["c20"] * (dev_f[0] - 3)**2 + p["c02"]*(dev_f[1]-3)**2 + p["c11"]*(dev_f[0] - 3)*(dev_f[1] - 3) + p["c10"] * (dev_f[0] - 3) + p["c01"] * (dev_f[1] - 3)
        return dev_t

    def _vol_mean(self, vol_f):
        p = self.params
        # vol_t = 1.5*(vol_f[0] - 1)**2

        vol_t = p["k"]*(vol_f[0] - 1)**2 + p["q"] *jnp.log(vol_f[0])**2
        return vol_t 


    def _get_psi_gp_cov(self, F1, F2):
        # 1. Compute Invariants for both inputs
        # i_star should return (I1, I2, J)
        i1_star, _ = invariants_and_derivatives(F1)
        i2_star, _ = invariants_and_derivatives(F2)
        dev1, vol1 = transform_input_features(i1_star)
        dev2, vol2 = transform_input_features(i2_star)

        def sparse_cov(x1, x2, z, sigma, ls, S_uu):
            # k_x1z: (1, M), k_zx2: (M, 1), k_x1x2: (1, 1)
            Kzz = discovery_kernel(z, z, sigma, ls) + 1e-6 * jnp.eye(z.shape[0])
            Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))
            k_x1z = discovery_kernel(x1[None, :], z, sigma, ls)
            k_zx2 = discovery_kernel(z, x2[None, :], sigma, ls)
            k_x1x2 = discovery_kernel(x1[None, :], x2[None, :], sigma, ls)
            # term1: Kiz @ Kzz_inv @ Kzi (Standard GP variance reduction)
            term1 = k_x1z @ Kzz_inv @ k_zx2
            # term2: Kiz @ (Kzz_inv @ S @ Kzz_inv) @ Kzi (Variational uncertainty)
            term2 = k_x1z @ Kzz_inv @ S_uu @ Kzz_inv @ k_zx2
            
            return (k_x1x2 - term1 + term2).reshape()
        # Deviatoric part

        cov_dev = sparse_cov(
            dev1, dev2, self.inducing_points["dev_z"], 
            self.params["dev_gp_sigma_scaling"], self.params["dev_gp_lengthscales"],
            jnp.diag(self.params["dev_u_var"])
        )

        # Volumetric part
        cov_vol = sparse_cov(
            vol1, vol2, self.inducing_points["vol_z"], 
            self.params["vol_gp_sigma_scaling"], self.params["vol_gp_lengthscales"],
            jnp.diag(self.params["vol_u_var"])
        )

        return cov_dev + cov_vol

    def load_params(self, params) :
        check1 = jnp.std(self.inducing_points["dev_z"], axis = 0)
        check2 = jnp.std(self.inducing_points["vol_z"], axis = 0)
        params = {
            "dev_gp_lengthscales" : jnp.std(self.inducing_points["dev_z"], axis = 0) + enforce_softplus_positive(params["raw_dev_gp_lengthscales"]), 
            "vol_gp_lengthscales" : jnp.std(self.inducing_points["vol_z"], axis = 0) + enforce_softplus_positive(params["raw_vol_gp_lengthscales"]), 
            "dev_gp_sigma_scaling" : jnp.exp(params["raw_dev_gp_sigma_scaling"]),
            "vol_gp_sigma_scaling" : jnp.exp(params["raw_vol_gp_sigma_scaling"]),
            "dev_u_mean" : enforce_softplus_positive(params["raw_dev_u_mean"]),
            "dev_u_var" : enforce_softplus_positive(params["raw_dev_u_var"]),
            "vol_u_mean" : enforce_softplus_positive(params["raw_vol_u_mean"]),
            "vol_u_var" : enforce_softplus_positive(params["raw_vol_u_var"]),
            "c20": enforce_softplus_positive(params["raw_c20"]),
            "c02": enforce_softplus_positive(params["raw_c02"]),
            "c11": enforce_softplus_positive(params["raw_c11"]),
            "c10": enforce_softplus_positive(params["raw_c10"]),
            "c01": enforce_softplus_positive(params["raw_c01"]),
            "k": enforce_softplus_positive(params["raw_k"]),
            "q": enforce_softplus_positive(params["raw_q"]),
        }
        # not update prior
        # params = {
        #     "dev_gp_lengthscales" : jnp.std(self.inducing_points["dev_z"], axis = 0) + enforce_softplus_positive(params["raw_dev_gp_lengthscales"]), 
        #     "vol_gp_lengthscales" : jnp.std(self.inducing_points["vol_z"], axis = 0) + enforce_softplus_positive(params["raw_vol_gp_lengthscales"]), 
        #     "dev_gp_sigma_scaling" : jnp.exp(params["raw_dev_gp_sigma_scaling"]),
        #     "vol_gp_sigma_scaling" : jnp.exp(params["raw_vol_gp_sigma_scaling"]),
        #     "dev_u_mean" : params["raw_dev_u_mean"],
        #     "dev_u_var" : enforce_softplus_positive(params["raw_dev_u_var"]),
        #     "vol_u_mean" : params["raw_vol_u_mean"],
        #     "vol_u_var" : enforce_softplus_positive(params["raw_vol_u_var"]),
        #     "c20": enforce_softplus_positive(params["raw_c20"]),
        #     "c02": enforce_softplus_positive(params["raw_c02"]),
        #     "c11": enforce_softplus_positive(params["raw_c11"]),
        #     "c10": enforce_softplus_positive(params["raw_c10"]),
        #     "c01": enforce_softplus_positive(params["raw_c01"]),
        #     "k": enforce_softplus_positive(params["raw_k"]),
        #     "q": enforce_softplus_positive(params["raw_q"]),
        # }
        return params




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
