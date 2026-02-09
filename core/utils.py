import jax
import jax.numpy as jnp
import numpy as np
from sklearn.neighbors import NearestNeighbors
# -------------------------------
# Tensor utility functions
# -------------------------------

def J_func(f):
    """Determinant of deformation gradient tensor."""
    return jnp.linalg.det(f)  # shape [N]

def B_func(f):
    """Left Cauchy-Green tensor."""
    return f @ jnp.swapaxes(f, -2, -1)  # [N,d,d]
def C_func(f) :
    return jnp.swapaxes(f, -2, -1) @ f  # [N,d,d]
def I1_func(B):
    """First invariant (trace)."""
    return jnp.trace(B, axis1=-2, axis2=-1)

def I2_func(B):
    """Second invariant."""
    trB = jnp.trace(B, axis1=-2, axis2=-1)
    trBB = jnp.trace(B @ B, axis1=-2, axis2=-1)
    return 0.5 * (trB**2 - trBB)

def I3_func(B):
    """Third invariant (determinant)."""
    return jnp.linalg.det(B)

# -------------------------------
# Strain energy functions
# -------------------------------


# -------------------------------
# Solve for coefficients (batched)
# -------------------------------

def solve_for_coefficients_batched(lambda_B, lambda_sigma):
    """
    Batched version of solve_for_coefficients.

    Args:
        lambda_B: array (batch_size, 3)
        lambda_sigma: array (batch_size, 3)

    Returns:
        coefficients_c: (batch_size, 3)
        V: (batch_size, 3, 3)
    """
    if lambda_B.ndim != 2 or lambda_B.shape[1] != 3:
        raise ValueError("lambda_B must have shape (batch_size, 3)")
    if lambda_sigma.ndim != 2 or lambda_sigma.shape[1] != 3:
        raise ValueError("lambda_sigma must have shape (batch_size, 3)")
    if lambda_B.shape[0] != lambda_sigma.shape[0]:
        raise ValueError("Batch sizes of lambda_B and lambda_sigma must match.")

    # Construct Vandermonde matrices
    col1 = jnp.ones_like(lambda_B)
    col2 = lambda_B
    col3 = lambda_B**2
    V = jnp.stack((col1, col2, col3), axis=-1)  # (batch_size, 3, 3)

    # Compute pseudoinverse and solve for coefficients
    V_pinv = jnp.linalg.pinv(V)
    coefficients_c = jnp.einsum("bij,bj->bi", V_pinv, lambda_sigma)
    return coefficients_c, V

# -------------------------------
# Generate random F tensors (Plane Stress)
# -------------------------------

def generate_random_F_plane_stress(n_samples, lambda_range=(0.5, 2.5), seed=None):
    """
    Generates n_samples of 3x3 Deformation Gradient (F) tensors 
    constrained to 2D Plane Strain (F33=1, F_i3=0, F_3i=0 for i=1,2).
    """
    key = jax.random.PRNGKey(seed if seed is not None else 0)
    low, high = lambda_range
    n_shear = n_samples // 10

    def single_sample(key):
        k1, k2 = jax.random.split(key)
        lambdas_2D = jax.random.uniform(k1, (2,), minval=low, maxval=high)
        V_2D = jnp.diag(lambdas_2D)
        theta = jax.random.uniform(k2, (), minval=0, maxval=2 * jnp.pi)
        R_2D = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)]
        ])
        F_2D = R_2D @ V_2D
        F = jnp.eye(3)
        F = F.at[:2, :2].set(F_2D)
        return F

    # Generate all random samples
    keys = jax.random.split(key, n_samples)
    F_samples = jax.vmap(single_sample)(keys)

    # Add shear states
    key_shear = jax.random.split(key, n_shear + 1)[-1]
    gamma_shear = jax.random.uniform(key_shear, (n_shear,), minval=low, maxval=high)
    F_shear = jnp.tile(jnp.eye(3), (n_shear, 1, 1))
    F_shear = F_shear.at[:, 0, 1].set(gamma_shear)

    return jnp.concatenate([F_samples, F_shear], axis=0)


from jax import vmap

def sum_negative_conjugate_mll(posteriors, datasets):
    """
    Computes the Sum Negative Marginal Log-Likelihood (Sum MLL) 
    across a tuple of independent GPs and their datasets.
    """
    # Vectorize the individual negative MLL function across the list/tuple of models and datasets
    all_mlls = vmap(
        lambda p, d: -gpx.objectives.conjugate_mll(p, d), 
        in_axes=(0, 0) # Map along the first axis of both the posteriors (p) and datasets (d)
    )(posteriors, datasets)
    
    # Sum the result
    return jnp.sum(all_mlls)

def fto3x3(f) :
    f3x3 = jnp.array([[f[0,0], f[0,1], 0.0],
                      [f[1,0], f[1,1], 0.0],
                      [0.0, 0.0, 1.0]])
    return f3x3

@jax.vmap
def transformation_jacobian(coords_elem) :
    x1, y1 = coords_elem[0]
    x2, y2 = coords_elem[1]
    x3, y3 = coords_elem[2]

    # Jacobian of shape function derivatives
    J = jnp.array([
        [x2 - x1, y2 - y1],
        [x3 - x1, y3 - y1]
    ])
    return J

@jax.vmap
def deformation_gradient_element(coords_elem, disp_elem):
    x1, y1 = coords_elem[0]
    x2, y2 = coords_elem[1]
    x3, y3 = coords_elem[2]

    # Jacobian of shape function derivatives
    J = jnp.array([
        [x2 - x1, y2 - y1],
        [x3 - x1, y3 - y1]
    ])

    # Area factor
    detJ = jnp.linalg.det(J)

    # Shape function derivatives in reference space
    dN_ref = jnp.array([
        [-1., -1.],
        [ 1.,  0.],
        [ 0.,  1.]
    ])

    # Convert to physical derivatives: dN/dx = inv(J)^T * dN_ref
    dNdx = jnp.transpose(jnp.linalg.solve(J, dN_ref.T))

    # Gradient of displacement
    gradu = disp_elem.T @ dNdx  # 2x3 @ 3x2 = 2x2

    # Deformation gradient
    F = jnp.eye(2) + gradu
    return F, dNdx

def detrend_3d_jax(X, y):
    N = X.shape[0]
    X_design = jnp.hstack([jnp.ones((N, 1)), X])  # shape (N, 4)

    # Solve via lstsq
    beta, residuals, rank, s = jnp.linalg.lstsq(X_design, y)

    trend = X_design @ beta
    y_detrended = y - trend

    return y_detrended, beta, trend

def calculate_min_ls(z):
    # For a 2D/3D point cloud, a quick way is to use the 
    # average distance to the nearest neighbor.
    nbrs = NearestNeighbors(n_neighbors=2).fit(z)
    distances, _ = nbrs.kneighbors(z)
    avg_dist = jnp.mean(distances[:, 1])
    return avg_dist * 0.5 # Minimum allowable lengthscale

def invariants_and_derivatives(F):
    f = fto3x3(F)
    C = f.T @ f
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C @ C))
    I3 = jnp.linalg.det(C)
    # derivatives wrt F (2x2)
    dI1_dF = 2*f
    dI2_dF = 2*(I1*f - f @ C)
    dI3_dF = 2*jnp.linalg.det(f)**2 * jnp.linalg.inv(f).T
    dI_dF = jnp.stack([dI1_dF, dI2_dF, dI3_dF])  # (3,2,2)
    return jnp.array([I1, I2, I3]), dI_dF


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

def transform_input_features(invariants) :
    i3 = jnp.maximum(invariants[2], 1e-6)
    j = jnp.sqrt(i3)
    i1_dev = i3**(-1/3)*invariants[0]
    i2_dev = i3**(-2/3)*invariants[1]
    dev_feature = jnp.stack([i1_dev, i2_dev], axis = -1)
    # vol_feature = jnp.stack([j, -2 * j], axis = -1)
    vol_feature = jnp.array([j])

    return dev_feature, vol_feature