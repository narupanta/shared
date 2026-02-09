import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad
from functools import partial
from tqdm import tqdm
key = jax.random.PRNGKey(0)

# ---------------------------
# Example training data
# ---------------------------
def make_data(key, n=20000):
    x = jnp.linspace(-5, 5, n).reshape(-1, 1)
    y = jnp.sin(x) + 0.2 * jax.random.normal(key, x.shape)
    return x, y

X, Y = make_data(key)

# ---------------------------
# Kernel (RBF)
# ---------------------------
def rbf_kernel(x1, x2, lengthscale, variance):
    sqdist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sqdist / lengthscale**2)

# ---------------------------
# Model parameters
# ---------------------------
def init_params(key, m=20):
    k1, k2 = jax.random.split(key)

    inducing = jnp.linspace(-5, 5, m).reshape(-1, 1)

    params = {
        "log_lengthscale": jnp.array(0.0),
        "log_variance": jnp.array(0.0),
        "log_noise": jnp.array(-1.0),
        "Z": inducing,
        "m": jnp.zeros((m, 1)),              # variational mean
        "L": jnp.eye(m),                     # variational chol
    }
    return params

params = init_params(key)

# ---------------------------
# ELBO objective
# ---------------------------
def elbo(params, X, Y):
    lengthscale = jnp.exp(params["log_lengthscale"])
    variance = jnp.exp(params["log_variance"])
    noise = jnp.exp(params["log_noise"])
    Z = params["Z"]
    m = params["m"]
    L = params["L"]

    Kuu = rbf_kernel(Z, Z, lengthscale, variance) + 1e-6 * jnp.eye(Z.shape[0])
    Kuf = rbf_kernel(Z, X, lengthscale, variance)
    Kff_diag = variance * jnp.ones(X.shape[0])

    Kuu_inv = jnp.linalg.inv(Kuu)

    S = L @ L.T
    A = Kuu_inv @ Kuf

    # Predictive mean & variance
    mean = (A.T @ m).reshape(-1)
    var = Kff_diag - jnp.sum(Kuf * (Kuu_inv @ Kuf), axis=0)
    var += jnp.sum((A.T @ S) * A.T, axis=1)
    var += noise

    # Likelihood term
    resid = Y.reshape(-1) - mean
    ll = -0.5 * jnp.sum(resid**2 / var + jnp.log(var))

    # KL divergence q(u)||p(u)
    trace_term = jnp.trace(Kuu_inv @ S)
    quad_term = m.T @ Kuu_inv @ m
    cj = jnp.maximum(jnp.diag(L), 1e-9)
    logdet_q = 2 * jnp.sum(jnp.log(cj))
    logdet_p = jnp.linalg.slogdet(Kuu)[1]

    kl = 0.5 * (trace_term + quad_term - Z.shape[0] + logdet_p - logdet_q) 
    # kl = jnp.array(0)

    return ll - kl.squeeze()

# ---------------------------
# Training step
# ---------------------------
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(params)

@jit
def train_step(params, opt_state, X, Y):
    loss, grads = value_and_grad(lambda p: -elbo(p, X, Y))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ---------------------------
# Train loop
# ---------------------------
pbar = tqdm(range(50000))
for i in pbar:
    params, opt_state, loss = train_step(params, opt_state, X, Y)
    pbar.set_postfix({"step": f"{i:04d}", "loss":f"{loss:.6f}"})
    # if i % 50 == 0:
    #     print(f"step {i}, loss {loss:.3f}")

# ---------------------------
# Prediction function
# ---------------------------
def predict(params, Xtrain, Xtest):
    lengthscale = jnp.exp(params["log_lengthscale"])
    variance = jnp.exp(params["log_variance"])
    noise = jnp.exp(params["log_noise"])
    Z = params["Z"]
    m = params["m"]
    L = params["L"]

    Kuu = rbf_kernel(Z, Z, lengthscale, variance) + 1e-6 * jnp.eye(Z.shape[0])
    Kus = rbf_kernel(Z, Xtest, lengthscale, variance)

    Kuu_inv = jnp.linalg.inv(Kuu)
    A = Kuu_inv @ Kus

    mean = (A.T @ m).reshape(-1)

    S = L @ L.T
    var = variance - jnp.sum(Kus * (Kuu_inv @ Kus), axis=0)
    var += jnp.sum((A.T @ S) * A.T, axis=1)
    var += noise

    return mean, var

Xtest = jnp.linspace(-6, 6, 200).reshape(-1, 1)
mean, var = predict(params, X, Xtest)

print("Prediction complete.")

import matplotlib.pyplot as plt
import numpy as np

# Convert JAX arrays to NumPy for plotting
X_np = np.array(X)
Y_np = np.array(Y)
Xtest_np = np.array(Xtest)
mean_np = np.array(mean)
std_np = np.sqrt(np.array(var))

plt.figure(figsize=(8, 5))

# Training data
plt.scatter(X_np, Y_np, s=20, label="Training data")

# Predictive mean
plt.plot(Xtest_np, mean_np, label="Predictive mean")

# Uncertainty band
plt.fill_between(
    Xtest_np.flatten(),
    mean_np - 2 * std_np,
    mean_np + 2 * std_np,
    alpha=0.2,
    label="±2 std dev",
)

# Inducing points
Z_np = np.array(params["Z"])
plt.scatter(Z_np, np.zeros_like(Z_np), marker="x", s=80, label="Inducing points")

plt.legend()
plt.title("Sparse GP Regression")
plt.xlabel("x")
plt.ylabel("y")

plt.savefig("sparse_gp_result.png", dpi=300, bbox_inches="tight")
plt.close()

