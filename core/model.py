import jax
import jax.numpy as jnp
from jax import random, vmap, grad, jit

from .kernel import rbf
from .dataclass import EnergyDist, StressDist, GPParams, GPWeights
from .features import IsotropicFeatureExtractor

class SparseHyperelasticityGP:
    """
    Sparse Gaussian Process model for hyperelasticity.
    Uses pathwise sampling and Matheron's rule to condition on inducing points.
    Strictly assumes a zero-mean prior for the strain energy density components.
    """
    def __init__(self, raw_params, I_z: jnp.ndarray, min_dev, min_vol, max_dev, max_vol, sampling_mode="pws", beta=1.0, L=200, feature_extractor=None):
        self.feature_extractor = feature_extractor if feature_extractor is not None else IsotropicFeatureExtractor()
        # 1. Inducing points split
        self.dev_z = jnp.asarray(I_z[:, :2], dtype=jnp.float64)
        self.vol_z = jnp.asarray(I_z[:, 2:], dtype=jnp.float64)
        self.min_dev = jnp.asarray(min_dev, dtype=jnp.float64)
        self.max_dev = jnp.asarray(max_dev, dtype=jnp.float64)
        self.min_vol = jnp.asarray(min_vol, dtype=jnp.float64)
        self.max_vol = jnp.asarray(max_vol, dtype=jnp.float64)
        
        self.sampling_mode = sampling_mode
        self.L = L  # Number of Random Fourier Features for pathwise sampling
        self.beta = beta
        
        # 2. Setup Parameters and Weights
        self.params = self.load_params(raw_params)
        self.gpweight = self.precompute_weights(raw_params)

    # ---------------------------------------------------------
    # 1. Parameter Management
    # ---------------------------------------------------------
    def load_params(self, p) -> GPParams:
        """Applies physical constraints (e.g., positivity via softplus/exp) to raw parameters."""
        def to_f64(x):
            return jnp.asarray(x, dtype=jnp.float64)

        dev_mu = to_f64(jax.nn.softplus(p.raw_dev_u_mean))
        vol_mu = to_f64(jax.nn.softplus(p.raw_vol_u_mean))
        dev_var = to_f64(jax.nn.softplus(p.raw_dev_u_var))
        vol_var = to_f64(jax.nn.softplus(p.raw_vol_u_var))

        # Force anchor points (First inducing point at zero energy)
        dev_z = to_f64(jax.nn.softplus(p.raw_dev_z)) + to_f64(jnp.array([3.0, 3.0]))
        vol_z = to_f64(jax.nn.softplus(p.raw_vol_z))

        dev_z = dev_z.at[0].set(to_f64(jnp.array([3.0, 3.0])))
        vol_z = vol_z.at[0].set(to_f64(jnp.array([1.0])))

        dev_u_mean = dev_mu.at[0].set(0.0)
        dev_u_var  = dev_var.at[0].set(1e-8)
        vol_u_mean = vol_mu.at[0].set(0.0)
        vol_u_var  = vol_var.at[0].set(1e-8)

        return GPParams(
            dev_ls=to_f64(self.max_dev.mean() * 2 * jax.nn.sigmoid(p.raw_dev_ls)),
            dev_sig=to_f64(jnp.exp(p.raw_dev_sig)),
            dev_u_mean=dev_u_mean,
            dev_u_var=dev_u_var,
            dev_z=dev_z,

            vol_ls=to_f64(self.max_vol * 2 * jax.nn.sigmoid(p.raw_vol_ls)),
            vol_sig=to_f64(jnp.exp(p.raw_vol_sig)),
            vol_u_mean=vol_u_mean,
            vol_u_var=vol_u_var,
            vol_z=vol_z,
            vol_kappa=to_f64(jax.nn.softplus(p.raw_vol_kappa)),

            sigma_free_x=to_f64(jnp.exp(p.log_sigma_free_x)),
            sigma_free_y=to_f64(jnp.exp(p.log_sigma_free_y)),
            sigma_fix_x=to_f64(jnp.exp(p.log_sigma_fix_x)),
            sigma_fix_y=to_f64(jnp.exp(p.log_sigma_fix_y))
        )

    # ---------------------------------------------------------
    # 2. Core GP Mathematics & Weight Precomputation
    # ---------------------------------------------------------
    def _compute_component_weights(self, z, u_mean, u_var, ls, sig):
        """Helper to precompute reusable covariance matrices and vectors for GP."""
        Kzz = rbf(z, z, sig, ls) + 1e-6 * jnp.eye(z.shape[0], dtype=jnp.float64)
        K_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0], dtype=jnp.float64))
        
        # We strictly assume a zero-mean prior, so v_diff is just u_mean - 0
        v_diff = u_mean
        
        M_mat = K_inv @ (Kzz - jnp.diag(u_var)) @ K_inv.T
        trace_term = jnp.trace(K_inv @ jnp.diag(u_var))
        mahalanobis_term = v_diff.T @ K_inv @ v_diff
        log_term = jnp.log(jnp.linalg.det(Kzz)) - jnp.log(jnp.linalg.det(jnp.diag(u_var)))
        
        return Kzz, K_inv, v_diff, trace_term, mahalanobis_term, M_mat, log_term

    def precompute_weights(self, params) -> GPWeights:
        """Precomputes weights for both deviatoric and volumetric components."""
        p = self.load_params(params)
        d_res = self._compute_component_weights(p.dev_z, p.dev_u_mean, p.dev_u_var, p.dev_ls, p.dev_sig)
        v_res = self._compute_component_weights(p.vol_z, p.vol_u_mean, p.vol_u_var, p.vol_ls, p.vol_sig)

        return GPWeights(
            dev_Kzz=d_res[0], dev_Kzz_inv=d_res[1], dev_v=d_res[2], dev_trace_term=d_res[3], 
            dev_mahalanobis_term=d_res[4], dev_M_mat=d_res[5], dev_logterm=d_res[6],
            vol_Kzz=v_res[0], vol_Kzz_inv=v_res[1], vol_v=v_res[2], vol_trace_term=v_res[3], 
            vol_mahalanobis_term=v_res[4], vol_M_mat=v_res[5], vol_logterm=v_res[6]
        )

    # ---------------------------------------------------------
    # 3. Pathwise Sampling (Physics-Informed)
    # ---------------------------------------------------------
    def get_path_psi_fn(self, key):
        """
        Returns a differentiable scalar function psi(F) for one realization.
        This uses Matheron's rule to condition random prior features on the inducing points.
        """
        k1, k2, k3, k4, k5, k6 = random.split(key, 6)
        p = self.params
        
        # 1. Random Fourier Features for Prior Paths
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

        # 2. Sample Inducing Values u ~ q(u)
        u_dev = jax.random.multivariate_normal(k3, p.dev_u_mean, jnp.diag(p.dev_u_var))
        u_vol = jax.random.multivariate_normal(k3, p.vol_u_mean, jnp.diag(p.vol_u_var))

        # 3. Correction Vectors (Matheron's Rule)
        # We subtract the random prior path at inducing points (mean is zero)
        v_dev_corr = jnp.linalg.solve(self.gpweight.dev_Kzz, u_dev - vmap(f_prior_dev)(self.params.dev_z))
        v_vol_corr = jnp.linalg.solve(self.gpweight.vol_Kzz, u_vol - vmap(f_prior_vol)(self.params.vol_z))

        def path_psi(f):
            dev, vol = self.feature_extractor.extract(f)
            
            # Deviatoric Path (Prior + Update)
            k_dz = rbf(dev, self.params.dev_z, p.dev_sig, p.dev_ls)
            psi_dev = f_prior_dev(dev) + jnp.dot(k_dz, v_dev_corr)
            
            # Volumetric Path (Prior + Update)
            k_vz = rbf(vol, self.params.vol_z, p.vol_sig, p.vol_ls)
            psi_vol = f_prior_vol(vol) + jnp.dot(k_vz, v_vol_corr)
            
            return (psi_dev + psi_vol).squeeze()

        return path_psi

    def get_path_dev_vol_psi_fn(self, key):
        k1, k2, k3, k4, k5, k6 = random.split(key, 6)
        p = self.params
        
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

        u_dev = jax.random.multivariate_normal(k3, p.dev_u_mean, jnp.diag(p.dev_u_var))
        u_vol = jax.random.multivariate_normal(k3, p.vol_u_mean, jnp.diag(p.vol_u_var))

        v_dev_corr = jnp.linalg.solve(self.gpweight.dev_Kzz, u_dev - vmap(f_prior_dev)(self.params.dev_z))
        v_vol_corr = jnp.linalg.solve(self.gpweight.vol_Kzz, u_vol - vmap(f_prior_vol)(self.params.vol_z))

        def path_dev_vol_psi(f):
            dev, vol = self.feature_extractor.extract(f)
            k_dz = rbf(dev, self.params.dev_z, p.dev_sig, p.dev_ls)
            psi_dev = f_prior_dev(dev) + jnp.dot(k_dz, v_dev_corr)
            k_vz = rbf(vol, self.params.vol_z, p.vol_sig, p.vol_ls)
            psi_vol = f_prior_vol(vol) + jnp.dot(k_vz, v_vol_corr)
            
            return psi_dev.squeeze(), psi_vol.squeeze()
        return path_dev_vol_psi

    # ---------------------------------------------------------
    # 4. API Endpoints for Loss / Evaluation
    # ---------------------------------------------------------
    def psi(self, f_mesh, key):
        """Calculates Energy across a mesh for a single realization."""
        if self.sampling_mode == "mds":
            return self.psi_mds(f_mesh, key)
        elif self.sampling_mode == "pws":
            return self.psi_pws(f_mesh, key)

    def piola(self, f_mesh, key):
        """Calculates Piola Stress field across a mesh for a single realization."""
        if self.sampling_mode == "mds":
            return self.piola_mds(f_mesh, key)
        elif self.sampling_mode == "pws":
            return self.piola_pws(f_mesh, key)

    def psi_pws(self, f, key):
        path_psi = self.get_path_psi_fn(key)
        return path_psi(f)

    def piola_pws(self, f, key):
        path_psi = self.get_path_psi_fn(key)
        piola_fn = grad(path_psi)
        return piola_fn(f)

    def kl_divergence(self):
        """Computes the KL divergence for ELBO training."""
        def component_kl(ma, log_t, tr, M):
            return 0.5 * (log_t - M + tr + ma)
        
        dev_kl = component_kl(self.gpweight.dev_mahalanobis_term, self.gpweight.dev_logterm, 
                              self.gpweight.dev_trace_term, self.params.dev_z.shape[0])
        vol_kl = component_kl(self.gpweight.vol_mahalanobis_term, self.gpweight.vol_logterm, 
                              self.gpweight.vol_trace_term, self.params.vol_z.shape[0])
        return (dev_kl + vol_kl) * self.beta

    # ---------------------------------------------------------
    # 5. Analytical GP Moments (Mean & Covariance for MDS)
    # ---------------------------------------------------------
    def dev_gp_mean(self, d):
        k_dz = rbf(d, self.params.dev_z, self.params.dev_sig, self.params.dev_ls)
        gp_term = k_dz @ self.gpweight.dev_Kzz_inv @ self.gpweight.dev_v
        return gp_term # Zero mean prior
    
    def vol_gp_mean(self, v):
        k_vz = rbf(v, self.params.vol_z, self.params.vol_sig, self.params.vol_ls)
        gp_term = k_vz @ self.gpweight.vol_Kzz_inv @ self.gpweight.vol_v
        return gp_term

    def psi_gp_mean(self, f):
        dev, vol = jax.vmap(self.feature_extractor.extract)(f)
        gp_mean = self.dev_gp_mean(dev) + self.vol_gp_mean(vol)
        return gp_mean.squeeze()

    def psi_gp_cov(self, f):
        dev, vol = jax.vmap(self.feature_extractor.extract)(f)
        k_dz = rbf(dev, self.params.dev_z, self.params.dev_sig, self.params.dev_ls)
        k_dd = rbf(dev, dev, self.params.dev_sig, self.params.dev_ls)

        cov_mat_dev = k_dd - k_dz @ self.gpweight.dev_M_mat @ k_dz.T

        k_vz = rbf(vol, self.params.vol_z, self.params.vol_sig, self.params.vol_ls)
        k_vv = rbf(vol, vol, self.params.vol_sig, self.params.vol_ls)
        cov_mat_vol = k_vv - k_vz @ self.gpweight.vol_M_mat @ k_vz.T
        
        # Constraint every entry to positive
        cov_mat_dev = jnp.maximum(cov_mat_dev, 1e-8)
        cov_mat_vol = jnp.maximum(cov_mat_vol, 1e-8)
        return jnp.diag(cov_mat_dev + cov_mat_vol)

    def psi_joint_cov(self, f):
        """Returns the full N x N dense covariance matrix for psi."""
        dev, vol = jax.vmap(self.feature_extractor.extract)(f)
        k_dz = rbf(dev, self.params.dev_z, self.params.dev_sig, self.params.dev_ls)
        k_dd = rbf(dev, dev, self.params.dev_sig, self.params.dev_ls)

        cov_mat_dev = k_dd - k_dz @ self.gpweight.dev_M_mat @ k_dz.T

        k_vz = rbf(vol, self.params.vol_z, self.params.vol_sig, self.params.vol_ls)
        k_vv = rbf(vol, vol, self.params.vol_sig, self.params.vol_ls)
        cov_mat_vol = k_vv - k_vz @ self.gpweight.vol_M_mat @ k_vz.T
        
        cov_full = cov_mat_dev + cov_mat_vol
        jitter = 1e-4 * jnp.eye(f.shape[0])
        return cov_full + jitter

    def piola_gp_var(self, f):
        """
        Computes the variance of the Piola Stress components using 
        double differentiation of the predictive covariance.
        """
        def psi_cov_single(f1, f2):
            dev1, vol1 = self.feature_extractor.extract(f1)
            dev2, vol2 = self.feature_extractor.extract(f2)
            
            k_d1z = rbf(dev1[None, :], self.params.dev_z, self.params.dev_sig, self.params.dev_ls)
            k_dz2 = rbf(self.params.dev_z, dev2[None, :], self.params.dev_sig, self.params.dev_ls)
            k_d1d2 = rbf(dev1[None, :], dev2[None, :], self.params.dev_sig, self.params.dev_ls)
            cov_dev = k_d1d2 - k_d1z @ self.gpweight.dev_M_mat @ k_dz2
            
            k_v1z = rbf(vol1[None, :], self.params.vol_z, self.params.vol_sig, self.params.vol_ls)
            k_vz2 = rbf(self.params.vol_z, vol2[None, :], self.params.vol_sig, self.params.vol_ls)
            k_v1v2 = rbf(vol1[None, :], vol2[None, :], self.params.vol_sig, self.params.vol_ls)
            cov_vol = k_v1v2 - k_v1z @ self.gpweight.vol_M_mat @ k_vz2
            
            return (cov_dev + cov_vol).squeeze()

        hessian_cov = jax.jacfwd(jax.jacrev(psi_cov_single, argnums=0), argnums=1)
        return hessian_cov(f, f)

    def psi_det(self, f):
        dev, vol = self.feature_extractor.extract(f)
        return self.dev_gp_mean(dev[None, :]).reshape() + self.vol_gp_mean(vol[None, :]).reshape()

    def piola_det(self, f):
        return jax.grad(self.psi_det)(f)

    def psi_dist(self, f_mesh):
        f_mesh = jnp.asarray(f_mesh, dtype=jnp.float64)
        posterior_mean = self.psi_gp_mean(f_mesh)
        posterior_covar = self.psi_gp_cov(f_mesh)
        return EnergyDist(posterior_mean, posterior_covar)

    def dev_psi_dist(self, f_mesh):
        dev, _ = jax.vmap(self.feature_extractor.extract)(f_mesh)
        mean = self.dev_gp_mean(dev)
        k_dz = rbf(dev, self.params.dev_z, self.params.dev_sig, self.params.dev_ls)
        k_dd = rbf(dev, dev, self.params.dev_sig, self.params.dev_ls)
        cov = k_dd - k_dz @ self.gpweight.dev_M_mat @ k_dz.T
        var = jnp.diag(jnp.maximum(cov, 1e-8))
        return EnergyDist(mean.squeeze(), var)
        
    def vol_psi_dist(self, f_mesh):
        _, vol = jax.vmap(self.feature_extractor.extract)(f_mesh)
        mean = self.vol_gp_mean(vol)
        k_vz = rbf(vol, self.params.vol_z, self.params.vol_sig, self.params.vol_ls)
        k_vv = rbf(vol, vol, self.params.vol_sig, self.params.vol_ls)
        cov = k_vv - k_vz @ self.gpweight.vol_M_mat @ k_vz.T
        var = jnp.diag(jnp.maximum(cov, 1e-8))
        return EnergyDist(mean.squeeze(), var)

    def piola_dist(self, f_mesh):
        """
        Calculates the Mean Piola Stress for a mesh of deformation gradients.
        f_mesh shape: (N, 2, 2)
        """
        f_mesh = jnp.asarray(f_mesh, dtype=jnp.float64)
        def single_psi_mean(f):
            dev, vol = self.feature_extractor.extract(f)
            return (self.dev_gp_mean(dev[None, :]) + self.vol_gp_mean(vol[None, :])).reshape()

        piola_mean_fn = jax.vmap(jax.grad(single_psi_mean))
        piola_means = piola_mean_fn(f_mesh)

        def single_piola_var(f):
            return jnp.einsum('ijij->ij', self.piola_gp_var(f))

        piola_vars_fn = jax.vmap(single_piola_var)
        piola_vars = piola_vars_fn(f_mesh)
        return StressDist(piola_means, piola_vars)
    
    def psi_mds(self, f, key):
        dist = self.psi_dist(f)
        psi = jax.random.multivariate_normal(key, dist.mean, dist.var)
        return psi

    def piola_mds(self, f_mesh, key):
        """
        Samples the ENTIRE correlated Piola stress field across a mesh.
        f_mesh: (N, 2, 2)
        """
        N = f_mesh.shape[0]
        dist_mean = self.piola_dist(f_mesh).mean.reshape(-1)
        
        K_full_tensor = jax.vmap(jax.vmap(self.piola_gp_var, in_axes=(None, 0)), in_axes=(0, None))(f_mesh, f_mesh)
        K_joint = K_full_tensor.transpose(0, 2, 1, 3, 4, 5).reshape(4*N, 4*N)
        K_joint += 1e-6 * jnp.eye(4*N) # Numerical stability jitter
        
        sample_flat = jax.random.multivariate_normal(key, dist_mean, K_joint)
        return sample_flat.reshape(N, 2, 2)
