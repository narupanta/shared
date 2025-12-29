import jax
import jax.numpy as jnp
import jax.random as jr
from .utils import deformation_gradient_element, transformation_jacobian, invariants_and_derivatives, fto3x3
from .model import SparseHyperelasticityGP, matern52_kernel, discovery_kernel

def _neumann_cell_force(coords_el, types_el, t3, t4):
    """
    coords_el: (3,2) float - coordinates of the 3 nodes of the element
    types_el:  (3,) int - node_type for these 3 nodes (global node_type[cells])
    t3, t4: scalars - traction magnitudes for types 3 and 4
    returns: (3,2) local nodal traction vector for this element
    """
    edges = jnp.array([[0, 1],
                       [1, 2],
                       [2, 0]])  # three local edges
    f_cell = jnp.zeros((3, 2))

    def body_fun(idx, f):
        i = edges[idx, 0]
        j = edges[idx, 1]

        ti = types_el[i]
        tj = types_el[j]

        # Only apply traction if both nodes of the edge have the same neumann type.
        is_right = (ti == 3) & (tj == 3)
        is_top   = (ti == 4) & (tj == 4)

        # choose traction vector for edge
        t_edge = jnp.where(is_right, jnp.array([t3, 0.0]),
                 jnp.where(is_top,   jnp.array([0.0, t4]),
                                         jnp.array([0.0, 0.0])))

        xi = coords_el[i]
        xj = coords_el[j]
        L = jnp.linalg.norm(xj - xi)

        # nodal contribution from this edge: each edge contributes L/2 * T to each of its two nodes
        fe_local = 0.5 * L * t_edge  # shape (2,)

        f = f.at[i].add(fe_local)
        f = f.at[j].add(fe_local)
        return f

    f_cell = jax.lax.fori_loop(0, 3, body_fun, f_cell)
    return f_cell  # (3,2)


def physical_loss(params, Z_I, coords, cells, u,
                  n_nodes, node_type, load_parameter):
    """
    Virtual Field Method Weak form loss
    params: Hyperparameter of Gaussian Process
    coords: (C, 3, 2) per-element nodal coords
    cells:  (C, 3) global node indices per element
    u: displacement (format compatible with deformation_gradient_element)
    reaction_forces: target reaction vector (4,) or similar used previously
    n_nodes: total number of nodes
    bc: (n_nodes, 2) boundary code mask (0 free, 1..4 etc)
    node_type: (n_nodes, 1) ints: 0 free, 1/2 fixed (dirichlet), 3 right, 4 top
    load_parameter: (2,) or (2,1) - [t3, t4]
    """

    # --- INTERNAL FORCES (unchanged) ---
    F, dNdx = deformation_gradient_element(coords, u)   # (C,2,2), (C,3,2,2?) matches your API
    dA = jnp.linalg.det(transformation_jacobian(coords)) / 2  # (C,)
    # params["g"] = params["g_mean"]
    hyperGP = SparseHyperelasticityGP(params["lengthscales"], params["log_scale_variance"], params["log_sigma_poly"], params["log_offset"], params["log_growth_constant"],params["poly_degree"], params["g_mean"], Z_I)
    f = jax.vmap(fto3x3)(F)
    piola = jax.vmap(hyperGP.piola_stress)(f)[:, :2, :2]

    # piola = piola_func(f)[:, :2, :2] 
    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)    # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2)
    f_int_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_int_cell)
    
    # --- NEUMANN EDGE-LENGTH TRACTION ---
    # normalize load_parameter to flat array
    t3 = load_parameter * 0.9# TO ADD ADJUSTABLE load_parameters
    t4 = load_parameter  # TO ADD ADJUSTABLE load_parameters
    # node_type may be (n_nodes,1) so flatten
    node_type_flat = jnp.asarray(node_type).reshape(-1)  # (n_nodes,)
    types_per_cell = node_type_flat[cells]               # (C,3)

    # vectorize per-element traction computation
    per_cell_vmap = jax.vmap(_neumann_cell_force, in_axes=(0, 0, None, None))
    f_neu_cells = per_cell_vmap(coords, types_per_cell, t3, t4)  # (C,3,2)

    # assemble global neumann nodal forces
    f_neu_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_neu_cells)

    # --- Residual R = int(grad v : P) dx  -  int(v·T) ds(Neumann)
    R_nodes = f_int_nodes - f_neu_nodes

    # only free DOFs contribute to the residual loss (bc == 0)
    blm_loss = jnp.sum(R_nodes[(node_type != 1) & (node_type != 2)] ** 2)

    fixed_nodes_loss1 = jnp.sum((jnp.sum(R_nodes[node_type == 1], axis = 0) + jnp.sum(f_neu_nodes[node_type == 3], axis = 0))**2)
    fixed_nodes_loss2 = jnp.sum((jnp.sum(R_nodes[node_type == 2], axis = 0) + jnp.sum(f_neu_nodes[node_type == 4], axis = 0))**2)

    return blm_loss + fixed_nodes_loss1 + fixed_nodes_loss2, (blm_loss, fixed_nodes_loss1, fixed_nodes_loss2)

def reparametrize(latent_mean, latent_log_var, key) :
    eps = jax.random.normal(key, latent_mean.shape)
    # u_sample = mean + Cholesky * noise
    sample = latent_mean + jnp.exp(latent_log_var) @ eps
    return sample

import jax
import jax.numpy as jnp

def calculate_kl_divergence(params, Z_I):
    """
    KL[q(u) || p(u)] where p(u) = N(0, Kuu) and q(u) = N(m_u, L_S L_S^T)
    """
    m_u = params['g_mean']  # (M, 1)
    L_S = jnp.exp(params['g_log_var'])  # (M, M) lower triangular Cholesky
    # L_S = jnp.exp(jnp.ones_like(m_u) * -10)  # (M, M) lower triangular Cholesky
    
    scale_variance = jnp.exp(params["log_scale_variance"])
    j_inducing = jnp.sqrt(Z_I[:, 2] + 1e-9)
    inducing_points_dev = jnp.stack([j_inducing**(-2/3)*Z_I[:, 0], j_inducing**(-4/3)*Z_I[:, 1]], axis = -1)
    # learnable_mean = jnp.exp(params["log_growth_constant1"]) * (inducing_points_dev[:, 0] - 3) + jnp.exp(params["log_growth_constant2"]) * (jnp.sqrt(inducing_points_dev[:, 2]) - 1)**2
    Kzz = discovery_kernel(inducing_points_dev, inducing_points_dev, params)
    # 1. Compute Prior Covariance Kuu
    Kzz = Kzz + 1e-6 * jnp.eye(Z_I.shape[0])

    Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(inducing_points_dev.shape[0]))
    # L_K = jnp.linalg.cholesky(Kuu)
    
    # 2. Trace Term: tr(Kuu^-1 * S)
    # Solve L_K * V = L_S -> V = L_K^-1 * L_S. Then tr(V V^T)
    # V = jax.lax.linalg.triangular_solve(L_K, L_S, lower=True)
    trace_term = jnp.trace(Kzz_inv @ jnp.diag(L_S))
    
    # 3. Mahalanobis Term: m_u^T * Kuu^-1 * m_u
    # alpha = jax.lax.linalg.triangular_solve(L_K, m_u, lower=True)
    # mahalanobis_term = jnp.sum(jnp.square(alpha))
    mahalanobis_term = m_u @ Kzz_inv @ m_u.T
    
    # 4. Log-determinant Term: ln |Kuu| - ln |S|
    log_det_K = jnp.log(jnp.linalg.det(Kzz))
    log_det_S = jnp.log(jnp.linalg.det(jnp.diag(L_S)))
    
    M = Z_I.shape[0]
    kl = 0.5 * (trace_term + mahalanobis_term - M + log_det_K - log_det_S)
    # jax.debug.breakpoint()
    return kl

def elbo_loss(params, Z_I, coords, cells, u, n_nodes, node_type, load_parameter, key):
    # Unpack parameters for clarity
    # lengthscales = params["lengthscales"]
    # variance = params["variance"]
    # growth_constant = params["growth_constant"]
    g_mean = params["g_mean"]
    sigma_physic = jnp.exp(params["log_sigma_physic"])

    params["g"] = reparametrize(g_mean, params["g_log_var"], key)
    # params["g"] = g_mean
    physic_loss = physical_loss(params, Z_I, coords, cells, u,
                  n_nodes, node_type, load_parameter)

    log_likelihood = - (1.0 / (2 * (sigma_physic**2))) * physic_loss - n_nodes * jnp.log(2 * jnp.pi * (sigma_physic**2))
    kl_div = calculate_kl_divergence(params, Z_I)/ sigma_physic**2
    # jax.debug.breakpoint()
    elbo = log_likelihood - kl_div
    total_loss = -elbo
    return total_loss, (log_likelihood, kl_div, physic_loss)