import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Any

class MaskedDense(nn.Module):
    features: int
    mask: jnp.ndarray
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, inputs):
        in_dim = inputs.shape[-1]
        kernel = self.param('kernel', nn.initializers.lecun_normal(), (in_dim, self.features), self.dtype)
        bias = self.param('bias', nn.initializers.zeros, (self.features,), self.dtype)
        return jnp.dot(inputs, kernel * self.mask) + bias

class FlaxMADE(nn.Module):
    num_params: int
    hidden_dims: tuple = (48,)
    dtype: Any = jnp.float64

    def setup(self):
        np.random.seed(42)
        m = [np.arange(1, self.num_params + 1)]
        for h in self.hidden_dims:
            m.append(np.random.randint(1, self.num_params, size=h))
        
        masks = []
        for i in range(len(self.hidden_dims)):
            masks.append( (m[i][:, None] <= m[i+1][None, :]).astype(np.float64) )
        
        final_mask = (m[-1][:, None] < m[0][None, :]).astype(np.float64)
        final_mask = np.repeat(final_mask, 2, axis=1)

        self.layers = [MaskedDense(self.hidden_dims[i], jnp.array(masks[i]), dtype=self.dtype) for i in range(len(self.hidden_dims))]
        self.final_layer = MaskedDense(self.num_params * 2, jnp.array(final_mask), dtype=self.dtype)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.leaky_relu(layer(x))
        out = self.final_layer(x)
        out = out.reshape(x.shape[:-1] + (self.num_params, 2))
        return out[..., 0], out[..., 1]

class DeepFlaxIAF(nn.Module):
    num_params: int
    num_layers: int = 16
    dtype: Any = jnp.float64
    
    def setup(self):
        hidden_width = self.num_params * 4
        self.mades = [FlaxMADE(num_params=self.num_params, hidden_dims=(hidden_width,), dtype=self.dtype) for _ in range(self.num_layers)]

    def __call__(self, key, num_samples):
        z = jax.random.normal(key, (num_samples, self.num_params), dtype=self.dtype)
        for made in self.mades:
            # IAF Sampling: loc and scale depend entirely on z (previous layer output), 
            # so we only need ONE forward pass per layer!
            loc, log_scale = made(z)
            log_scale = jnp.clip(log_scale, a_min=-10.0, a_max=3.0)
            scale = jnp.exp(log_scale)
            z = loc + scale * z
        return z

class Critic(nn.Module):
    hidden_dims: tuple = (768, 768)
    dtype: Any = jnp.float64
    
    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.Dense(h, dtype=self.dtype)(x)
            x = nn.softplus(x)
        return nn.Dense(1, dtype=self.dtype)(x)
