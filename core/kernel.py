import jax.numpy as jnp

def rbf(f1, f2, sigma_scaling, lengthscales) -> jnp.array:
    """
    Computes the ARD RBF Gram/Covariance matrix.
    X1, X2 are (N, D) and (M, D).
    lengthscales is a vector (D,).
    """
    # 1. Calculate the rawre inputs are at least 2D for broadcasting: (N, D) and (M, D)
    f1 = jnp.atleast_2d(f1)
    f2 = jnp.atleast_2d(f2)
    r = f1[:, None, :] - f2[None, :, :]

    exponent = jnp.sum((r / lengthscales)**2, axis=-1)
    ret = sigma_scaling**2 * jnp.exp(-0.5 * exponent)
    return ret