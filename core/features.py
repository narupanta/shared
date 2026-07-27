import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple
from .utils import invariants_and_derivatives, transform_input_features

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extracts features (e.g., invariants) from the deformation gradient F.
        Returns:
            Tuple of features to be fed into the GP components.
            Currently expects (dev_features, vol_features).
        """
        pass

class IsotropicFeatureExtractor(FeatureExtractor):
    def extract(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extracts standard isotropic invariants (I1, I2, J) and splits them
        into deviatoric and volumetric features.
        """
        invariants, _ = invariants_and_derivatives(f)
        dev, vol = transform_input_features(invariants)
        return dev, vol
