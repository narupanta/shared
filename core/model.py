
from .utils import *
from .datasetclass import *
from .dataclass import *
from .kernel import *
import jax.numpy as jnp
from typing import NamedTuple
import jax  

from jax import random, vmap, grad, jit

class SparseHyperelasticityGP:
    def __init__(self, raw_params, I_z: jnp.ndarray, min_dev, min_vol, max_dev, max_vol, sampling_mode = "pws", L=200):
        # 1. Inducing points split
        self.dev_z = I_z[:, :2]
        self.vol_z = I_z[:, 2:]
        self.min_dev = min_dev
        self.max_dev = max_dev

        self.min_vol = min_vol
        self.max_vol = max_vol
        self.sampling_mode = sampling_mode
        self.norm_dev = self.norm_dev(self.dev_z)
        self.norm_vol = self.norm_vol(self.vol_z)

        self.L = L  # Number of Random Fourier Features

        # 2. Setup Parameters and Weights
        self.params = self.load_params(raw_params)
        self.gpweight = self.precompute_weights(raw_params)
    def norm_dev(self, z) :
        return (z - self.min_dev)/(self.max_dev - self.min_dev)
    def norm_vol(self, z) :
        return (z - self.min_vol)/(self.max_vol - self.min_vol)

    def load_params(self, p):
        """Applies physical constraints to raw parameters."""
        dev_mu = jax.nn.softplus(p.raw_dev_u_mean)
        vol_mu = jax.nn.softplus(p.raw_vol_u_mean)
        dev_var = jax.nn.softplus(p.raw_dev_u_var)
        vol_var = jax.nn.softplus(p.raw_vol_u_var)

        # Force anchor points (First inducing point at zero energy)
        dev_z = jax.nn.softplus(p.raw_dev_z) + jnp.array([3.0, 3.0])
        vol_z = jax.nn.softplus(p.raw_vol_z)

        dev_z = dev_z.at[0].set(jnp.array([3.0, 3.0]))
        vol_z = vol_z.at[0].set(jnp.array([1.0]))

        dev_u_mean = dev_mu.at[0].set(0.0)
        dev_u_var  = dev_var.at[0].set(1e-6)
        vol_u_mean = vol_mu.at[0].set(0.0)
        vol_u_var  = vol_var.at[0].set(1e-6)

        return GPParams(
            dev_ls=self.max_dev.mean() * 2 * jax.nn.sigmoid(p.raw_dev_ls),
            # dev_ls=jnp.array([2.0, 2.0]),

            dev_sig=jnp.exp(p.raw_dev_sig),
            dev_u_mean=dev_u_mean,
            dev_u_var=dev_u_var,
            dev_z=dev_z,

            vol_ls= self.max_vol * 2 * jax.nn.sigmoid(p.raw_vol_ls),
            # vol_ls=jnp.array([1.0]),

            vol_sig=jnp.exp(p.raw_vol_sig),
            vol_u_mean=vol_u_mean,
            vol_u_var=vol_u_var,
            vol_z = vol_z,
            **{k: jax.nn.softplus(getattr(p, f"raw_{k}")) for k in ['c01', 'c02', 'c10', 'c11', 'c20', 'k', 'q', "s"]},
            sigma_free_x=jnp.exp(p.log_sigma_free_x),
            sigma_free_y=jnp.exp(p.log_sigma_free_y),
            sigma_fix_x=jnp.exp(p.log_sigma_fix_x),
            sigma_fix_y=jnp.exp(p.log_sigma_fix_y)
        )

    def _compute_component_weights(self, z, u_mean, u_var, ls, sig, trend_fn):
        Kzz = rbf(z, z, sig, ls) + 1e-6 * jnp.eye(z.shape[0])
        K_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))
        v_diff = u_mean - vmap(trend_fn)(z)
        
        M_mat = K_inv @ (Kzz - jnp.diag(u_var)) @ K_inv.T
        trace_term = jnp.trace(K_inv @ jnp.diag(u_var))
        mahalanobis_term = v_diff.T @ K_inv @ v_diff
        log_term = jnp.log(jnp.linalg.det(Kzz)) - jnp.log(jnp.linalg.det(jnp.diag(u_var)))
        
        return Kzz, K_inv, v_diff, trace_term, mahalanobis_term, M_mat, log_term

    def precompute_weights(self, params) -> GPWeights:
        p = self.load_params(params)
        # dev_z = self.dev_z
        # vol_z = self.vol_z
        dev_z = p.dev_z
        vol_z = p.vol_z
        d_res = self._compute_component_weights(dev_z, p.dev_u_mean, p.dev_u_var, p.dev_ls, p.dev_sig, self.dev_mean_func)
        v_res = self._compute_component_weights(vol_z, p.vol_u_mean, p.vol_u_var, p.vol_ls, p.vol_sig, self.vol_mean_func)
        # d_res = self._compute_component_weights(self.norm_dev, p.dev_u_mean, p.dev_u_var, p.dev_ls, p.dev_sig, self.dev_mean_func)
        # v_res = self._compute_component_weights(self.norm_vol, p.vol_u_mean, p.vol_u_var, p.vol_ls, p.vol_sig, self.vol_mean_func)



        return GPWeights(
            dev_Kzz=d_res[0], dev_Kzz_inv=d_res[1], dev_v=d_res[2], dev_trace_term=d_res[3], 
            dev_mahalanobis_term=d_res[4], dev_M_mat=d_res[5], dev_logterm=d_res[6],
            vol_Kzz=v_res[0], vol_Kzz_inv=v_res[1], vol_v=v_res[2], vol_trace_term=v_res[3], 
            vol_mahalanobis_term=v_res[4], vol_M_mat=v_res[5], vol_logterm=v_res[6]
        )

    def dev_mean_func(self, d):
        p = self.params
        i1_bar_3, i2_bar_3 = d[0] - 3, d[1] - 3
        return (p.c01 * i1_bar_3 + p.c10 * i2_bar_3 + p.c11 * i1_bar_3 * i2_bar_3 + 
                p.c02 * i1_bar_3**2 + p.c20 * i2_bar_3**2) * 0 
        # return (p.c01 * i1_bar_3 + p.c10 * i2_bar_3)
    def vol_mean_func(self, v):
        p = self.params
        j_minus_1 = v[0] - 1
        return (p.k * j_minus_1**2 + p.q * jnp.log(v[0])**2) * 0

    # --- Pathwise Sampling Logic ---

    def get_path_psi_fn(self, key):
        """Returns a differentiable scalar function psi(F) for one realization."""
        k1, k2, k3, k4, k5, k6 = random.split(key, 6)
        p = self.params
        
        # 1. Prior Sample Weights
        w_dev_prior = random.normal(k1, (self.L,))
        w_vol_prior = random.normal(k2, (self.L,))

        W_dev = random.normal(k3, (2, self.L)) 
        b_dev = random.uniform(k4, (self.L,)) * 2 * jnp.pi
        W_vol = random.normal(k5, (1, self.L)) 
        b_vol = random.uniform(k6, (self.L,)) * 2 * jnp.pi

        def f_prior_dev(d):
            phi = jnp.sqrt(2.0 * p.dev_sig**2 / self.L) * jnp.cos(jnp.dot(d, W_dev / p.dev_ls[:, None]) + b_dev)
            return jnp.dot(phi, w_dev_prior)

        def f_prior_vol(v):
            phi = jnp.sqrt(2.0 * p.vol_sig**2 / self.L) * jnp.cos(jnp.dot(v, W_vol / p.vol_ls[:, None]) + b_vol)

            return jnp.dot(phi, w_vol_prior)

        # 2. Inducing Sample u ~ q(u)
        u_dev = jax.random.multivariate_normal(k3, p.dev_u_mean, jnp.diag(p.dev_u_var))
        u_vol = jax.random.multivariate_normal(k3, p.vol_u_mean, jnp.diag(p.vol_u_var))

        # u_dev = p.dev_u_mean + jnp.sqrt(p.dev_u_var) * random.normal(k3, p.dev_u_mean.shape)
        # u_vol = p.vol_u_mean + jnp.sqrt(p.vol_u_var) * random.normal(k3, p.vol_u_mean.shape)

        # 3. Correction Vectors (Matheron's Rule)
        # We subtract the mean trend and the random prior path at inducing points
        v_dev_corr = jnp.linalg.solve(self.gpweight.dev_Kzz, u_dev - vmap(self.dev_mean_func)(self.params.dev_z) - vmap(f_prior_dev)(self.params.dev_z))
        v_vol_corr = jnp.linalg.solve(self.gpweight.vol_Kzz, u_vol - vmap(self.vol_mean_func)(self.params.vol_z) - vmap(f_prior_vol)(self.params.vol_z))

        def path_psi(f):
            # Transformation to Invariants
            invariants, _ = invariants_and_derivatives(f)
            dev, vol = transform_input_features(invariants)
            
            # Deviatoric Path
            k_dz = rbf(dev, self.params.dev_z, p.dev_sig, p.dev_ls)
            psi_dev = self.dev_mean_func(dev) + f_prior_dev(dev) + jnp.dot(k_dz, v_dev_corr)

            
            # Volumetric Path
            k_vz = rbf(vol, self.params.vol_z, p.vol_sig, p.vol_ls)
            psi_vol = self.vol_mean_func(vol) + f_prior_vol(vol) + jnp.dot(k_vz, v_vol_corr)
            
            return (psi_dev + psi_vol).squeeze()

        return path_psi
    def psi_dist(self, f_mesh) :
        posterior_mean = self.psi_gp_mean(f_mesh)
        posterior_covar = self.psi_gp_cov(f_mesh)
        return EnergyDist(posterior_mean, posterior_covar)

    def psi_gp_mean(self, f) :
        invariants, _ = jax.vmap(invariants_and_derivatives)(f)
        dev, vol = jax.vmap(transform_input_features)(invariants)
        gp_mean = self.dev_gp_mean(dev) + self.vol_gp_mean(vol)
        return gp_mean.squeeze()
    def psi_gp_cov(self, f) :
        invariants, _ = jax.vmap(invariants_and_derivatives)(f)
        dev, vol = jax.vmap(transform_input_features)(invariants)
        k_dz = rbf(dev, self.params.dev_z, self.params.dev_sig, self.params.dev_ls)
        k_dd = rbf(dev, dev, self.params.dev_sig, self.params.dev_ls)

        cov_mat_dev = k_dd - k_dz @ self.gpweight.dev_M_mat @ k_dz.T

        k_vz = rbf(vol, self.params.vol_z, self.params.vol_sig, self.params.vol_ls)
        k_vv = rbf(vol, vol, self.params.vol_sig, self.params.vol_ls)
        cov_mat_vol = k_vv - k_vz @ self.gpweight.vol_M_mat @ k_vz.T
        # constraint every entry to positive
        cov_mat_dev = jnp.maximum(cov_mat_dev, 1e-8)
        cov_mat_vol = jnp.maximum(cov_mat_vol, 1e-8)
        return jnp.diag(cov_mat_dev + cov_mat_vol)
    def dev_gp_mean(self, d) :
        mean_prior = jax.vmap(self.dev_mean_func)(d)
        dev_v = self.gpweight.dev_v
        k_dz = rbf(d, self.params.dev_z, self.params.dev_sig, self.params.dev_ls)
        gp_term = k_dz @ self.gpweight.dev_Kzz_inv @ dev_v
        return mean_prior + gp_term
    
    def vol_gp_mean(self, v) :
        mean_prior = jax.vmap(self.vol_mean_func)(v)
        vol_v = self.gpweight.vol_v
        k_vz = rbf(v, self.params.vol_z, self.params.vol_sig, self.params.vol_ls)
        gp_term = k_vz @ self.gpweight.vol_Kzz_inv @ vol_v
        return mean_prior + gp_term
    def piola_gp_var(self, f):
        """
        Computes the variance of the Piola Stress components using 
        double differentiation of the predictive covariance.
        """
        def psi_cov_single(f1, f2):
            # We need a scalar-output covariance function to differentiate
            # This mirrors your psi_gp_cov logic but for two distinct points
            invariants1, _ = invariants_and_derivatives(f1)
            dev1, vol1 = transform_input_features(invariants1)
            invariants2, _ = invariants_and_derivatives(f2)
            dev2, vol2 = transform_input_features(invariants2)
            
            # Deviatoric part
            k_d1z = rbf(dev1[None, :], self.params.dev_z, self.params.dev_sig, self.params.dev_ls)
            k_dz2 = rbf(self.params.dev_z, dev2[None, :], self.params.dev_sig, self.params.dev_ls)
            k_d1d2 = rbf(dev1[None, :], dev2[None, :], self.params.dev_sig, self.params.dev_ls)
            cov_dev = k_d1d2 - k_d1z @ self.gpweight.dev_M_mat @ k_dz2
            
            # Volumetric part
            k_v1z = rbf(vol1[None, :], self.params.vol_z, self.params.vol_sig, self.params.vol_ls)
            k_vz2 = rbf(self.params.vol_z, vol2[None, :], self.params.vol_sig, self.params.vol_ls)
            k_v1v2 = rbf(vol1[None, :], vol2[None, :], self.params.vol_sig, self.params.vol_ls)
            cov_vol = k_v1v2 - k_v1z @ self.gpweight.vol_M_mat @ k_vz2
            
            return (cov_dev + cov_vol).squeeze()

        # The variance of the gradient is the Hessian of the covariance function
        # at f1 = f2. We use jacfwd(jacrev) for the (2,2,2,2) tensor of covariances.
        hessian_cov = jax.jacfwd(jax.jacrev(psi_cov_single, argnums=0), argnums=1)
        return hessian_cov(f, f)
    def psi_det(self, f) :
        invariants, _ = invariants_and_derivatives(f)
        dev, vol = transform_input_features(invariants)
        return self.dev_mean_func(dev) + self.vol_mean_func(vol)
    def piola_det(self, f) :
        return jax.grad(self.psi_det)(f)
    def piola_dist(self, f_mesh):
        """
        Calculates the Mean Piola Stress for a mesh of deformation gradients.
        f_mesh shape: (N, 2, 2)
        """
        # Define the scalar energy function for one F
        def single_psi_mean(f):
            # We use your existing psi_gp_mean logic but for a single 2x2
            invariants, _ = invariants_and_derivatives(f)
            dev, vol = transform_input_features(invariants)
            # Note: ensuring these return scalars
            return (self.dev_gp_mean(dev[None, :]) + self.vol_gp_mean(vol[None, :])).reshape()

        # Vectorized Gradient (Mean Piola)
        piola_mean_fn = jax.vmap(jax.grad(single_psi_mean))
        piola_means = piola_mean_fn(f_mesh)

        # Vectorized Hessian (Piola Variance/Covariance)
        # This uses the double-differentiation logic discussed earlier
        def single_piola_var(f):
            # Compute the 2x2x2x2 Hessian of the covariance at f
            return jnp.einsum('ijij->ij', self.piola_gp_var(f)) # Re-use the function from previous turn

        piola_vars_fn = jax.vmap(single_piola_var)
        piola_vars = piola_vars_fn(f_mesh)
        # piola_vars = self.piola_gp_var(f_mesh)
        return StressDist(piola_means, piola_vars)
    
    def psi_mds(self, f, key) :
        dist = self.psi_dist(f)
        psi = jax.random.multivariate_normal(key, dist.mean, dist.var)
        return psi
    # def piola_mds(self, f, key) :
    #     dist = self.piola_dist(f)
    #     return jax.random.multivariate_normal(key, dist.mean, dist.var)
    def piola_mds(self, f_mesh, key):
        """
        Samples the ENTIRE correlated Piola stress field across a mesh.
        f_mesh: (N, 2, 2)
        """
        N = f_mesh.shape[0]
        
        # 1. Compute the Mean Stress Field (N, 2, 2) -> (4N,)
        dist_mean = self.piola_dist(f_mesh).mean.reshape(-1)
        
        # 2. Compute the Full Joint Covariance (4N, 4N)
        # This requires a double-loop or a vectorized Hessian over ALL pairs (fi, fj)


        # Vectorizing this over the mesh is the bottleneck
        # produces (N, N, 2, 2, 2, 2)
        K_full_tensor = jax.vmap(jax.vmap(self.piola_gp_var, in_axes=(None, 0)), in_axes=(0, None))(f_mesh, f_mesh)
        
        # Reshape to (4N, 4N)
        K_joint = K_full_tensor.transpose(0, 2, 1, 3, 4, 5).reshape(4*N, 4*N)
        K_joint += 1e-6 * jnp.eye(4*N) # Numerical stability jitter
        
        # 3. Sample
        sample_flat = jax.random.multivariate_normal(key, dist_mean, K_joint)
    
        return sample_flat.reshape(N, 2, 2)
    def psi_pws(self, f, key) :
        path_psi = self.get_path_psi_fn(key)
        return path_psi(f)
    def piola_pws(self, f, key) :
        path_psi = self.get_path_psi_fn(key)
        # Gradient of energy = Piola Stress
        piola_fn = grad(path_psi)
        # Vectorize over all elements in the mesh
        return piola_fn(f)

    def psi(self, f_mesh, key) :
        """Calculates Energy across a mesh for a single realization."""
        if self.sampling_mode == "mds" :
            return self.psi_mds(f_mesh, key)
        elif self.sampling_mode == "pws" :
            return self.psi_pws(f_mesh, key)
    def piola(self, f_mesh, key):
        """Calculates Piola Stress field across a mesh for a single realization."""
        if self.sampling_mode == "mds" :
            return self.psi_mds(f_mesh, key)
        elif self.sampling_mode == "pws" :
            return self.piola_pws(f_mesh, key)


    def kl_divergence(self):
        """Computes the KL divergence for ELBO training."""
        def component_kl(ma, log_t, tr, M):
            return 0.5 * (log_t - M + tr + ma)
        
        dev_kl = component_kl(self.gpweight.dev_mahalanobis_term, self.gpweight.dev_logterm, 
                              self.gpweight.dev_trace_term, self.params.dev_z.shape[0])
        vol_kl = component_kl(self.gpweight.vol_mahalanobis_term, self.gpweight.vol_logterm, 
                              self.gpweight.vol_trace_term, self.params.vol_z.shape[0])
        return (dev_kl + vol_kl) * 10 



from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
from .dataclass import GPParams, GPWeights, EnergyDist, StressDist

class ProbabilisticHyperelasticModel(ABC):
    """Abstract Base Class for all Hyperelastic GP models."""
    
    @abstractmethod
    def load_params(self, raw_params) -> GPParams:
        """Transform raw optimization variables into constrained physical parameters."""
        pass

    @abstractmethod
    def psi_dist(self, f: jnp.ndarray) -> EnergyDist:
        """Return the mean and variance of the strain energy density."""
        pass

    @abstractmethod
    def piola_dist(self, f: jnp.ndarray) -> StressDist:
        """Return the mean and variance of the Piola stress."""
        pass

    @abstractmethod
    def kl_divergence(self) -> jnp.ndarray:
        """Compute KL divergence for the ELBO."""
        pass

class Neohookean(ProbabilisticHyperelasticModel) :
    def __init__(self, raw_params: RawParams, I_z: jnp.ndarray) :
        self.params = self.load_params(raw_params)
    def load_params(self, raw_params: RawParams) :

        return Params(c01 = jax.nn.softplus(raw_params.raw_c01), c02 = jax.nn.softplus(raw_params.raw_c02),
                c10 = jax.nn.softplus(raw_params.raw_c10), c11 = jax.nn.softplus(raw_params.raw_c11),
                c20 = jax.nn.softplus(raw_params.raw_c20), k = jax.nn.softplus(raw_params.raw_k),
                q = jax.nn.softplus(raw_params.raw_q), s = jax.nn.softplus(raw_params.raw_s),
                c01_var=jax.nn.softplus(raw_params.raw_c01_var), c02_var=jax.nn.softplus(raw_params.raw_c02_var),
                c10_var=jax.nn.softplus(raw_params.raw_c10_var), c11_var=jax.nn.softplus(raw_params.raw_c11_var),
                c20_var=jax.nn.softplus(raw_params.raw_c20_var), k_var=jax.nn.softplus(raw_params.raw_k_var),
                q_var=jax.nn.softplus(raw_params.raw_q_var), s_var=jax.nn.softplus(raw_params.raw_s_var))
    



    def psi(self, f, key) :
        invariants, _ = invariants_and_derivatives(f)
        dev, vol = transform_input_features(invariants)
        i1_bar = dev[0]
        sample_c01 = self.params.c01 + jax.random.normal(key, (1,)) * self.params.c01_var
        sample_k = self.params.k = jax.random.normal(key, (1,)) * self.params.k_var
        dev = sample_c01 * (i1_bar - 3) 
        vol = sample_k * (vol[0] - 1)**2 
        return dev + vol
    
    def piola(self, f, key) :
        return jax.grad(lambda f: self.psi(f, key))(f)
    