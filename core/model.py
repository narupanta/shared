# gpjax_version.py
import os

import numpy as np
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
from core.utils import *
from core.datasetclass import *
import jax.numpy as jnp
import jax.random as jr
# from sklearn.preprocessing import MinMaxScaler
import pickle
import gpjax as gpx
import optax as ox
from gpjax.kernels.computations import DenseKernelComputation
key = jr.key(123)
class MinMaxScaler:
    """A minimal, JAX-compatible MinMaxScaler with small-epsilon safety."""

    def __init__(self, feature_range=(0.0, 1.0), eps: float = 1e-8):
        self.feature_range = feature_range
        self.eps = eps
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X):
        X = jnp.asarray(X)
        self.data_min_ = jnp.min(X, axis=0)
        self.data_max_ = jnp.max(X, axis=0)
        raw_range = self.data_max_ - self.data_min_
        # avoid exact zero by using epsilon
        self.data_range_ = jnp.where(jnp.abs(raw_range) < self.eps, 1.0, raw_range)
        return self

    def transform(self, X):
        X = jnp.asarray(X)
        scale = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        X_scaled = (X - self.data_min_) * scale + self.feature_range[0]
        return X_scaled

    def inverse_transform(self, X):
        X = jnp.asarray(X)
        scale = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        return (X - self.feature_range[0]) / scale + self.data_min_

    def to_numpy_dict(self):
        """Return a numpy-serializable dict for saving with pickle."""
        return {
            "feature_range": tuple(self.feature_range),
            "data_min": np.asarray(self.data_min_),
            "data_max": np.asarray(self.data_max_),
            "data_range": np.asarray(self.data_range_),
            "eps": float(self.eps),
        }

    @classmethod
    def from_numpy_dict(cls, d):
        scaler = cls(feature_range=d.get("feature_range", (0.0, 1.0)), eps=d.get("eps", 1e-8))
        scaler.data_min_ = jnp.asarray(d["data_min"])
        scaler.data_max_ = jnp.asarray(d["data_max"])
        scaler.data_range_ = jnp.asarray(d["data_range"])
        return scaler
class TensorBasisGPModel:
    def __init__(
        self,
        means: list[gpx.mean_functions.Zero],
        kernels: list[gpx.kernels.RBF],
        train_x=None,
        train_y=None,
    ):
        self.number_gps = len(means)
        self.means = means
        self.kernels = kernels

        # These start empty until user loads data
        self.train_x = None
        self.train_y = None
        self.datapoints = None
        self.priors = None
        self.likelihoods = None
        self.posteriors = None
        self.opt_posteriors = None

        # If data was passed, load immediately
        if train_x is not None and train_y is not None:
            self.load(train_x, train_y)

    # ------------------------------------------------------
    # LOAD TRAINING DATA LATER (lazy initialization)
    # ------------------------------------------------------
    def load(self, train_x, train_y):
        """Load training data and create GP objects."""
        self.train_x = train_x
        self.train_y = train_y

        # Create datasets for each GP output
        self.datapoints = [
            gpx.Dataset(X=train_x, y=train_y[:, i, None])
            for i in range(self.number_gps)
        ]

        # Create priors
        self.priors = [
            gpx.gps.Prior(mean_function=mean, kernel=kernel)
            for mean, kernel in zip(self.means, self.kernels)
        ]

        # Create likelihoods
        self.likelihoods = [
            gpx.likelihoods.Gaussian(
                num_datapoints=self.datapoints[i].n,
                obs_stddev=jnp.array(1e-3),
            )
            for i in range(self.number_gps)
        ]

        # Create initial posteriors (before optimization)
        self.posteriors = [
            prior * likelihood
            for prior, likelihood in zip(self.priors, self.likelihoods)
        ]

        # Optimized posterior initially same as unoptimized
        self.opt_posteriors = list(self.posteriors)

    # ------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------
    def optimize_hyperparameters(self):
        if self.datapoints is None:
            raise ValueError("Training data not loaded. Call `load()` first.")

        opt_posteriors = []
        histories = []

        for datapoint, posterior in zip(self.datapoints, self.posteriors):
            opt_posterior, history = gpx.fit_scipy(
                model=posterior,
                objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
                train_data=datapoint,
                trainable=gpx.parameters.Parameter,
            )
            opt_posteriors.append(opt_posterior)
            histories.append(history)

        self.opt_posteriors = opt_posteriors
        return opt_posteriors, histories

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------
    def predict_coeffs(self, invariants):
        if self.opt_posteriors is None:
            raise ValueError("Model not trained — call optimize_hyperparameters().")

        pred_means_list = []
        pred_stds_list = []

        for idx, opt_posterior in enumerate(self.opt_posteriors):
            latent_dist = opt_posterior.predict(invariants, self.datapoints[idx])
            predictive_dist = opt_posterior.likelihood(latent_dist)

            predictive_mean = predictive_dist.mean
            predictive_std = jnp.sqrt(predictive_dist.variance)

            pred_means_list.append(predictive_mean)
            pred_stds_list.append(predictive_std)

        pred_means = jnp.stack(pred_means_list, axis=-1)
        pred_stds = jnp.stack(pred_stds_list, axis=-1)

        return pred_means, pred_stds

    def predict_cauchy_stress(self, deformation_gradient):
        b = B_func(deformation_gradient)

        invariants = jnp.stack(
            [I1_func(b), I2_func(b), I3_func(b)], axis=-1
        )

        coeff_means, coeff_stds = self.predict_coeffs(invariants)

        c1, c2, c3 = coeff_means[..., 0], coeff_means[..., 1], coeff_means[..., 2]
        s1, s2, s3 = coeff_stds[..., 0], coeff_stds[..., 1], coeff_stds[..., 2]

        I = jnp.eye(3)

        sigma_mean = (
            c1[..., None, None] * I +
            c2[..., None, None] * b +
            c3[..., None, None] * (b @ b)
        )

        sigma_std = (
            s1[..., None, None] * jnp.abs(I) +
            s2[..., None, None] * jnp.abs(b) +
            s3[..., None, None] * jnp.abs(b @ b)
        )

        return sigma_mean, sigma_std

    def predict_piola_stress(self, deformation_gradient):
        sigma_mean, sigma_std = self.predict_cauchy_stress(deformation_gradient)
        detF = J_func(deformation_gradient)

        FinvT = jnp.linalg.inv(jnp.swapaxes(deformation_gradient, -2, -1))

        P_mean = detF[:, None, None] * (sigma_mean @ FinvT)
        P_std = detF[:, None, None] * (sigma_std @ FinvT)

        return P_mean, P_std

    # ------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        np.save(os.path.join(save_path, "train_x.npy"), np.asarray(self.train_x))
        np.save(os.path.join(save_path, "train_y.npy"), np.asarray(self.train_y))

        opt_posteriors_params_dict = to_state_dict(self.opt_posteriors)

        with open(os.path.join(save_path, "opt_posterior_params.pkl"), "wb") as f:
            pickle.dump(opt_posteriors_params_dict, f)

    # ------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------
    def load_model(self, model_path):
        train_x = np.load(os.path.join(model_path, "train_x.npy"))
        train_y = np.load(os.path.join(model_path, "train_y.npy"))

        # Rebuild GP structure with priors, likelihoods, posteriors
        self.load(train_x, train_y)

        # Load optimized parameters
        with open(os.path.join(model_path, "opt_posterior_params.pkl"), "rb") as f:
            opt_posterior_params_dict = pickle.load(f)

        # Rehydrate posterior tree
        self.opt_posteriors = from_state_dict(self.opt_posteriors, opt_posterior_params_dict)


class SVTBGPModel:
    def __init__(self, train_x = None, train_y = None, incuding_points = None) :
        self.train_x = None
        self.train_y = None
        self.datapoints = None
        self.priors = None
        self.likelihoods = None
        self.posteriors = None
        self.opt_posteriors = None
        self.qs = None
        self.incuding_points = None
        if train_x is not None and train_y is not None and incuding_points is not None:
            self.load(train_x, train_y, incuding_points)
    # ------------------------------------------------------
    # LOAD TRAINING DATA LATER (lazy initialization)
    # ------------------------------------------------------
    def load(self, train_x, train_y, incuding_points):
        """Load training data and create GP objects."""
        self.train_x = train_x
        self.train_y = train_y
        self.datapoints = [
            gpx.Dataset(X=train_x, y=train_y[:, i, None])
            for i in range(3)
        ]
        ini_lengthscale = jnp.std(train_x, axis = 0)
        # ini_lengthscale = 0.1 * (jnp.max(train_x, axis = 0) - jnp.min(train_x, axis = 0)) 
        # Create datasets for each GP output
        mean = gpx.mean_functions.Zero()
        kernel = gpx.kernels.Matern52(active_dims=[0,1,2], lengthscale = ini_lengthscale, n_dims=3)
        # kernel = gpx.kernels.RationalQuadratic(active_dims=[0,1,2], lengthscale = ini_lengthscale, n_dims=3) 

        self.priors = [
            gpx.gps.Prior(mean_function=mean, kernel=kernel)
        ] * 3

        # Create likelihoods
        self.likelihoods = [
            gpx.likelihoods.Gaussian(
                num_datapoints=self.datapoints[0].n,
                obs_stddev=1
            )
        ] * 3

        self.posteriors = [
            prior * likelihood
            for prior, likelihood in zip(self.priors, self.likelihoods)
        ]
        self.opt_posteriors = self.posteriors
        
        self.inducing_inputs = jnp.stack([jnp.linspace(jnp.min(self.train_x, axis = 0)[i], jnp.max(self.train_x, axis = 0)[i], incuding_points) 
                                     for i in range(self.train_x.shape[1])])
        self.qs = [gpx.variational_families.VariationalGaussian(posterior=p, inducing_inputs=self.inducing_inputs) for p in self.posteriors]
    def optimization(self, num_iters) :
        schedule = ox.warmup_cosine_decay_schedule(
                                            init_value=0.00,
                                            peak_value=0.02,
                                            warmup_steps=75,
                                            decay_steps=10000,
                                            end_value=0.001,
                                        )

        opt_posteriors = []
        histories = []
        for q, d in zip(self.qs, self.datapoints) :
            opt_posterior, history = gpx.fit(model=q,
                    objective=lambda p, d: -gpx.objectives.elbo(p, d),
                    train_data=d,
                    optim=ox.adam(learning_rate=schedule),
                    num_iters=num_iters,
                    key=jr.key(42),
                    batch_size=128,
                    trainable=gpx.parameters.Parameter)
            opt_posteriors.append(opt_posterior)
            histories.append(history)
        self.opt_posteriors = opt_posteriors
        return opt_posteriors, histories
    def predict_coeffs(self, invariants):
        if self.opt_posteriors is None:
            raise ValueError("Model not trained — call optimize_hyperparameters().")

        pred_means_list = []
        pred_stds_list = []

        for idx, opt_posterior in enumerate(self.opt_posteriors):
            latent_dist = opt_posterior(invariants)
            predictive_dist = opt_posterior.posterior.likelihood(latent_dist)

            predictive_mean = predictive_dist.mean
            predictive_std = jnp.sqrt(predictive_dist.variance)

            pred_means_list.append(predictive_mean)
            pred_stds_list.append(predictive_std)

        pred_means = jnp.stack(pred_means_list, axis=-1)
        pred_stds = jnp.stack(pred_stds_list, axis=-1)

        return pred_means, pred_stds

    def predict_cauchy_stress(self, deformation_gradient):
        b = B_func(deformation_gradient)

        invariants = jnp.stack(
            [I1_func(b), I2_func(b), I3_func(b)], axis=-1
        )

        coeff_means, coeff_stds = self.predict_coeffs(invariants)

        c1, c2, c3 = coeff_means[..., 0], coeff_means[..., 1], coeff_means[..., 2]
        s1, s2, s3 = coeff_stds[..., 0], coeff_stds[..., 1], coeff_stds[..., 2]

        I = jnp.eye(3)

        sigma_mean = (
            c1[..., None, None] * I +
            c2[..., None, None] * b +
            c3[..., None, None] * (b @ b)
        )

        sigma_std = (
            s1[..., None, None] * jnp.abs(I) +
            s2[..., None, None] * jnp.abs(b) +
            s3[..., None, None] * jnp.abs(b @ b)
        )

        return sigma_mean, sigma_std

    def predict_piola_stress(self, deformation_gradient):
        sigma_mean, sigma_std = self.predict_cauchy_stress(deformation_gradient)
        detF = J_func(deformation_gradient)

        FinvT = jnp.linalg.inv(jnp.swapaxes(deformation_gradient, -2, -1))

        P_mean = detF[:, None, None] * (sigma_mean @ FinvT)
        P_std = detF[:, None, None] * (sigma_std @ FinvT)

        return P_mean, P_std
    
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)

        np.save(os.path.join(save_path, "train_x.npy"), np.asarray(self.train_x))
        np.save(os.path.join(save_path, "train_y.npy"), np.asarray(self.train_y))

        opt_posteriors_params_dict = to_state_dict(self.opt_posteriors)

        with open(os.path.join(save_path, "opt_posterior_params.pkl"), "wb") as f:
            pickle.dump(opt_posteriors_params_dict, f)

    # ------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------
    def load_model(self, model_path):
        train_x = np.load(os.path.join(model_path, "train_x.npy"))
        train_y = np.load(os.path.join(model_path, "train_y.npy"))
        # Rebuild GP structure with priors, likelihoods, posteriors
        self.load(train_x, train_y, 50)

        # Load optimized parameters
        with open(os.path.join(model_path, "opt_posterior_params.pkl"), "rb") as f:
            opt_posterior_params_dict = pickle.load(f)

        # Rehydrate posterior tree
        self.opt_posteriors = from_state_dict(self.opt_posteriors, opt_posterior_params_dict)


class CoefficientKernel(gpx.kernels.AbstractKernel):
    def __init__(
        self,
        kernel1: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1, 2]),
        kernel2: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1, 2]),
        kernel3: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0, 1, 2]),
    ):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3 
        super().__init__(compute_engine=DenseKernelComputation())

    def __call__(self, X, Xp):
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0

        z = jnp.array(X[3], dtype=int)
        zp = jnp.array(Xp[3], dtype=int)

        # achieve the correct value via 'switches' that are either 1 or 0
        k1_switch = (z == 0) * (zp == 0)
        k2_switch = (z == 1) * (zp == 1)
        k3_switch = (z == 2) * (zp == 2)

        return  k1_switch * self.kernel1(X, Xp) + k2_switch * self.kernel2(X, Xp) + k3_switch * self.kernel3(X, Xp)
    
class SVTBGPModelCK :
    def __init__(self, train_x = None, train_y = None, incuding_points = None) :
        self.train_x = None
        self.train_y = None
        self.datapoints = None
        self.priors = None
        self.likelihoods = None
        self.posteriors = None
        self.opt_posteriors = None
        self.qs = None
        self.incuding_points = None
        if train_x is not None and train_y is not None and incuding_points is not None:
            self.load(train_x, train_y, incuding_points)
    # ------------------------------------------------------
    # LOAD TRAINING DATA LATER (lazy initialization)
    # ------------------------------------------------------
    def load(self, train_x, train_y, incuding_points):
        """Load training data and create GP objects."""
        self.train_x = train_x
        self.train_y = train_y
        self.datapoints = [
            gpx.Dataset(X=train_x, y=train_y[:, i, None])
            for i in range(3)
        ]
        ini_lengthscale = jnp.std(train_x, axis = 0)
        # ini_lengthscale = 0.1 * (jnp.max(train_x, axis = 0) - jnp.min(train_x, axis = 0)) 
        # Create datasets for each GP output
        mean = gpx.mean_functions.Zero()
        kernel = gpx.kernels.RBF(active_dims=[0,1,2], lengthscale = ini_lengthscale, n_dims=3) + gpx.kernels.Polynomial(active_dims=[0, 1, 2], degree = 2, n_dims = 3)
        # kernel = gpx.kernels.RationalQuadratic(active_dims=[0,1,2], lengthscale = ini_lengthscale, n_dims=3) 

        self.priors = [
            gpx.gps.Prior(mean_function=mean, kernel=kernel)
        ] * 3

        # Create likelihoods
        self.likelihoods = [
            gpx.likelihoods.Gaussian(
                num_datapoints=self.datapoints[0].n,
                obs_stddev=1
            )
        ] * 3

        self.posteriors = [
            prior * likelihood
            for prior, likelihood in zip(self.priors, self.likelihoods)
        ]
        self.opt_posteriors = self.posteriors
        
        self.inducing_inputs = jnp.stack([jnp.linspace(jnp.min(self.train_x, axis = 0)[i], jnp.max(self.train_x, axis = 0)[i], incuding_points) 
                                     for i in range(self.train_x.shape[1])])
        self.qs = [gpx.variational_families.VariationalGaussian(posterior=p, inducing_inputs=self.inducing_inputs) for p in self.posteriors]
    def optimization(self, num_iters) :
        schedule = ox.warmup_cosine_decay_schedule(
                                            init_value=0.00,
                                            peak_value=0.02,
                                            warmup_steps=75,
                                            decay_steps=10000,
                                            end_value=0.001,
                                        )

        opt_posteriors = []
        histories = []
        for q, d in zip(self.qs, self.datapoints) :
            opt_posterior, history = gpx.fit(model=q,
                    objective=lambda p, d: -gpx.objectives.elbo(p, d),
                    train_data=d,
                    optim=ox.adam(learning_rate=schedule),
                    num_iters=num_iters,
                    key=jr.key(42),
                    batch_size=128,
                    trainable=gpx.parameters.Parameter)
            opt_posteriors.append(opt_posterior)
            histories.append(history)
        self.opt_posteriors = opt_posteriors
        return opt_posteriors, histories