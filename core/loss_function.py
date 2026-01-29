import jax
import jax.numpy as jnp
import jax.random as jr
from .utils import deformation_gradient_element, transformation_jacobian, invariants_and_derivatives, fto3x3
from .model import SparseHyperelasticityGP, matern52_kernel, discovery_kernel, enforce_softplus_positive, transform_input_features

# def piola_quadrature() :
#     return
# def total_loss(p, model, u_array, reactions, coords, cells, node_type, key) :
#     model.params = model.load_params(p)
#     model.precompute_weights()
#     # piola_func = lambda f: model.piola(f, key)
#     sigma_physic = jnp.exp(p["log_sigma_physic"])
#     sigma_glob = jnp.exp(p["log_sigma_glob"])
#     GH_X = jnp.array([-2.0201, -0.9585, 0.0, 0.9585, 2.0201])
#     GH_W = jnp.array([0.0112, 0.2220, 0.5333, 0.2220, 0.0112])

#     def quadrature_step(x_node, weight):
#         # Deterministic Piola at a specific uncertainty point
#         piola_func = lambda f: model.piola_quadrature(f, x_node) 
        
#         # Use SCAN instead of VMAP for the 50,000 steps to save RAM
#         def body_fn(carry, inputs):
#             u, react = inputs
#             free_res, react_res = physical_loss_per_loadstep_force_controlled(
#                 u, react, piola_func, coords, cells, node_type
#             )
#             return None, (free_res, react_res)
        
#         _, (res_free, res_fix) = jax.lax.scan(body_fn, None, (u_array, reactions))
#         return weight * jnp.sum(res_free), weight * jnp.sum(res_fix)

#     # Sum the weighted losses across the 5 nodes
#     sum_free, sum_fix = jax.vmap(quadrature_step)(GH_X, GH_W)
#     ell_, (sum_free_loss, sum_fix_loss) = ell(sigma_physic, sigma_glob, u_array, reactions, piola_func, coords, cells, node_type, key)
#     kl_div = model.kl_divergance()
#     total_loss = -ell_ + kl_div
#     return total_loss, (ell_, kl_div, sum_free_loss, sum_fix_loss)
def total_loss(p, model, u_array, loads, reactions, coords, cells, node_type, key) :
    model.params = model.load_params(p)
    model.precompute_weights()
    piola_func = lambda f: model.piola(f, key)
    sigma_physic = jnp.exp(p["log_sigma_physic"])
    sigma_glob = jnp.exp(p["log_sigma_glob"])
    ell_, (sum_free_loss, sum_fix_loss) = ell(sigma_physic, sigma_glob, u_array, loads, reactions, piola_func, coords, cells, node_type, key)
    kl_div = model.kl_divergance()
    total_loss = -ell_ + kl_div
    return total_loss, (ell_, kl_div, sum_free_loss, sum_fix_loss)


def ell(sigma_physic, sigma_glob, u_array, loads, reactions, piola_func, coords, cells, node_type, key) :

    sum_free_loss, sum_fix_loss = total_physical_loss(u_array, loads, reactions, piola_func, coords, cells, node_type)
    free_dofs = u_array.shape[0] * (u_array.shape[1] * u_array.shape[2] - jnp.sum((node_type == 1))/2 - jnp.sum(node_type == 2)/2) 
    # n_reaction = reactions.shape[0] * reactions.shape[1]

    free_log_likelihood = - (1.0 / (2 * (sigma_physic**2))) * sum_free_loss - free_dofs/2.0 * jnp.log(2 * jnp.pi * (sigma_physic**2))
    fix_log_likelihood = - (1.0 / (2 * (sigma_glob**2))) * sum_fix_loss - 2/2.0 * jnp.log(2 * jnp.pi * (sigma_glob**2))
    expected_log_likelihood = fix_log_likelihood + free_log_likelihood
    return expected_log_likelihood, (sum_free_loss, sum_fix_loss)

def total_physical_loss(u_array, loads, reactions, piola_func, coords, cells, node_type) :

    # plpl = jax.vmap(physical_loss_per_loadstep, in_axes=(0, 0, None, None, None, None))

    plpl = jax.vmap(physical_loss_per_loadstep_force_controlled, in_axes=(0, 0, 0, None, None, None, None))
    # def plpl(u_array, loads, reactions, piola_func, coords, cells, node_type) :
    #     ret1, ret2 = 0, 0
    #     for loadstep in range(len(loads)) :
    #         free_node_residual, reaction_loss = physical_loss_per_loadstep_force_controlled(u_array[loadstep], loads[loadstep], reactions[loadstep], piola_func, coords, cells, node_type)
    #         ret1 += free_node_residual
    #         ret2 += reaction_loss
    #     return ret1, ret2

    free_node_residual, reaction_loss = plpl(u_array, loads, reactions, piola_func, coords, cells, node_type)
    return jnp.sum(free_node_residual), jnp.sum(reaction_loss)

def physical_loss_per_loadstep_force_controlled(u, load, reaction, piola_func, coords, cells, node_type) :
    u_cells = u[cells]
    coord_cells = coords[cells]
    n_nodes = coords.shape[0]
    F, dNdx = deformation_gradient_element(coord_cells, u_cells)   # (C,2,2), (C,3,2,2?) matches your API
    dA = jnp.linalg.det(transformation_jacobian(coord_cells)) / 2  # (C,)

    
    f = jax.vmap(fto3x3)(F)
    piola = jax.vmap(piola_func)(f)[:, :2, :2]

    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)    # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2)
    f_int_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_int_cell)
    
    # --- NEUMANN EDGE-LENGTH TRACTION ---
    # normalize load_parameter to flat array
    t3 = load[0]# TO ADD ADJUSTABLE load_parameters
    t4 = load[1]  # TO ADD ADJUSTABLE load_parameters
    # node_type may be (n_nodes,1) so flatten
    node_type_flat = jnp.asarray(node_type).reshape(-1)  # (n_nodes,)
    types_per_cell = node_type_flat[cells]               # (C,3)

    # vectorize per-element traction computation
    per_cell_vmap = jax.vmap(_neumann_cell_force, in_axes=(0, 0, None, None))
    f_neu_cells = per_cell_vmap(coord_cells, types_per_cell, t3, t4)  # (C,3,2)

    # assemble global neumann nodal forces
    f_neu_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_neu_cells)

    # --- Residual R = int(grad v : P) dx  -  int(v·T) ds(Neumann)
    R_nodes = f_int_nodes - f_neu_nodes
    free_node_in = (node_type != 1) & (node_type != 2)
    free_node_on_dbc_left = (node_type == 1)
    free_node_on_dbc_bottom = (node_type == 2)
    # only free DOFs contribute to the residual loss (bc == 0)
    free_node_in_loss = R_nodes[free_node_in] ** 2
    free_node_on_dbc_left_loss = R_nodes[free_node_on_dbc_left, 1] ** 2
    free_node_on_dbc_bottom_loss = R_nodes[free_node_on_dbc_bottom, 0] ** 2

    # loss at the dirichlet nodes
    # fixed_nodes_loss1 = jnp.sum((jnp.sum(f_int_nodes[node_type == 1], axis = 0) + jnp.sum(f_neu_nodes[node_type == 3], axis = 0))**2)
    # fixed_nodes_loss2 = jnp.sum((jnp.sum(f_int_nodes[node_type == 2], axis = 0) + jnp.sum(f_neu_nodes[node_type == 4], axis = 0))**2)

    fixed_nodes_loss1 = jnp.sum((jnp.sum(R_nodes[node_type == 1, 0]) - reaction[0])**2)
    fixed_nodes_loss2 = jnp.sum((jnp.sum(R_nodes[node_type == 2, 1]) - reaction[1])**2)

    # total_physic_loss = blm_loss + fixed_nodes_loss1 + fixed_nodes_loss2
    free_loss = jnp.concat([free_node_in_loss.flatten(), free_node_on_dbc_left_loss, free_node_on_dbc_bottom_loss]) 
    fix_loss = jnp.stack([fixed_nodes_loss1, fixed_nodes_loss2])
    return free_loss, fix_loss
def physical_loss_per_loadstep(u, reaction, piola_func, coords, cells, node_type) :
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
    u_cells = u[cells]
    coord_cells = coords[cells]
    n_nodes = coords.shape[0]
    # --- INTERNAL FORCES (unchanged) ---
    F, dNdx = deformation_gradient_element(coord_cells, u_cells)   # (C,2,2), (C,3,2,2?) matches your API
    dA = jnp.linalg.det(transformation_jacobian(coord_cells)) / 2  # (C,)

    # hyperGP = SparseHyperelasticityGP(params, Z_I)
    f = jax.vmap(fto3x3)(F)

    piola = jax.vmap(piola_func)(f)[:, :2, :2]
    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)    # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2)
    f_int_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_int_cell)

    free_r0 = jnp.sum(f_int_nodes[node_type == 0] ** 2)
    free_r1 = jnp.sum(f_int_nodes[node_type == 1, 1] ** 2)
    free_r2 = jnp.sum(f_int_nodes[node_type == 2, 0] ** 2)
    free_r3 = jnp.sum(f_int_nodes[node_type == 3, 1] ** 2)
    free_r4 = jnp.sum(f_int_nodes[node_type == 4, 0] ** 2)
    free_r_total = free_r0 + free_r1 + free_r2 + free_r3 + free_r4

    fnl_left = (jnp.sum(f_int_nodes[node_type == 1, 0]) - reaction[0])**2
    fnl_bottom = (jnp.sum(f_int_nodes[node_type == 2, 1]) - reaction[1])**2
    fnl_right = (jnp.sum(f_int_nodes[node_type == 3, 0]) - reaction[2])**2
    fnl_top = (jnp.sum(f_int_nodes[node_type == 4, 1]) - reaction[3])**2

    reaction_loss = fnl_left + fnl_bottom + fnl_right + fnl_top

    return free_r_total, reaction_loss





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


def physical_loss(params, model, coords, cells, u,
                  n_nodes, node_type, load_parameter, key = None):
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

    hyperGP = model
    hyperGP.params = hyperGP.load_params(params)
    hyperGP.precompute_weights()
    
    f = jax.vmap(fto3x3)(F)
    piola = jax.vmap(lambda x: hyperGP.piola(x, key))(f)[:, :2, :2]

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
    free_node_in = (node_type != 1) & (node_type != 2)
    free_node_on_dbc_left = (node_type == 1)
    free_node_on_dbc_bottom = (node_type == 2)
    # only free DOFs contribute to the residual loss (bc == 0)
    free_node_in_loss = R_nodes[free_node_in] ** 2
    free_node_on_dbc_left_loss = R_nodes[free_node_on_dbc_left, 1] ** 2
    free_node_on_dbc_bottom_loss = R_nodes[free_node_on_dbc_bottom, 0] ** 2

    # loss at the dirichlet nodes
    fixed_nodes_loss1 = jnp.sum((jnp.sum(f_int_nodes[node_type == 1], axis = 0) + jnp.sum(f_neu_nodes[node_type == 3], axis = 0))**2)
    fixed_nodes_loss2 = jnp.sum((jnp.sum(f_int_nodes[node_type == 2], axis = 0) + jnp.sum(f_neu_nodes[node_type == 4], axis = 0))**2)

    # total_physic_loss = blm_loss + fixed_nodes_loss1 + fixed_nodes_loss2
    free_loss = jnp.concat([free_node_in_loss.flatten(), free_node_on_dbc_left_loss, free_node_on_dbc_bottom_loss]) 
    fix_loss = jnp.stack([fixed_nodes_loss1, fixed_nodes_loss2])
    return free_loss, fix_loss
    # def total_physic_loss(params, model, coords, cells, u_list, n_nodes, node_type, load_parameters, key = None) :

    #     jax.vmap(physical_loss, in_axes = (None, None, None, 0, None, None, None, 0, 0))

def calculate_kl_divergence(params, model):
    """
    KL[q(u) || p(u)] where p(u) = N(0, Kuu) and q(u) = N(m_u, L_S L_S^T)
    """
    dev_z = model.inducing_points["dev_z"]
    vol_z = model.inducing_points["vol_z"]
    params = {
            "dev_gp_lengthscales" : jnp.exp(params["raw_dev_gp_lengthscales"]), 
            "vol_gp_lengthscales" : jnp.exp(params["raw_vol_gp_lengthscales"]), 
            "dev_gp_sigma_scaling" : jnp.exp(params["raw_dev_gp_sigma_scaling"]),
            "vol_gp_sigma_scaling" : jnp.exp(params["raw_vol_gp_sigma_scaling"]),
            "dev_u_mean" : enforce_softplus_positive(params["raw_dev_u_mean"]),
            "dev_u_var" : enforce_softplus_positive(params["raw_dev_u_var"]),
            "vol_u_mean" : enforce_softplus_positive(params["raw_vol_u_mean"]),
            "vol_u_var" : enforce_softplus_positive(params["raw_vol_u_var"]),
            "p": enforce_softplus_positive(params["raw_p"]),
            "q": enforce_softplus_positive(params["raw_q"]),
            "r": enforce_softplus_positive(params["raw_r"]),
            "s": enforce_softplus_positive(params["raw_s"]),
            "t": enforce_softplus_positive(params["raw_t"]),
            "c": enforce_softplus_positive(params["raw_c"]),
            "a": enforce_softplus_positive(params["raw_a"])
        }

    def kl(u_mean, u_var, z, sigma_scaling, lengthscales) :
        Kzz = discovery_kernel(z, z, sigma_scaling, lengthscales)
        Kzz = Kzz + 1e-6 * jnp.eye(z.shape[0])

        Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0]))
        trace_term = jnp.trace(Kzz_inv @ jnp.diag(u_var))
        mahalanobis_term = u_mean @ Kzz_inv @ u_mean.T
        
        log_det_K = jnp.log(jnp.linalg.det(Kzz))
        log_det_S = jnp.log(jnp.linalg.det(jnp.diag(u_var)))
        
        M = z.shape[0]
        return 0.5 * (trace_term + mahalanobis_term - M + log_det_K - log_det_S)
    kl_dev = kl(params["dev_u_mean"], params["dev_u_var"], dev_z, params["dev_gp_sigma_scaling"], params["dev_gp_lengthscales"])
    kl_vol = kl(params["vol_u_mean"], params["vol_u_var"], vol_z, params["vol_gp_sigma_scaling"], params["vol_gp_lengthscales"])
    return kl_dev + kl_vol

def elbo_loss(params, model, coords, cells, u, n_nodes, node_type, load_parameter, key):
    # Unpack parameters for clarity
    # dev_z = model.inducing_points.dev_z
    # vol_z = model.inducing_points.vol_z
    # sigma_physic = 1e-3
    sigma_physic = jnp.exp(params["log_sigma_physic"])
    sigma_glob = jnp.exp(params["log_sigma_glob"])

    # sigma_physic = 1e-2
    # sigma_glob = 0.5
    free_loss, fix_loss = physical_loss(params, model, coords, cells, u,
                  n_nodes, node_type, load_parameter, key)
    sum_free_loss = jnp.sum(free_loss)
    sum_fix_loss = jnp.sum(fix_loss)
    physic_loss = sum_free_loss + sum_fix_loss
    free_dofs = free_loss.shape[0]
    fix_dim = 2
    free_log_likelihood = - (1.0 / (2 * (sigma_physic**2))) * sum_free_loss - free_dofs/2.0 * jnp.log(2 * jnp.pi * (sigma_physic**2))
    fix_log_likelihood = - (1.0 / (2 * (sigma_glob**2))) * sum_fix_loss - fix_dim/2.0 * jnp.log(2 * jnp.pi * (sigma_glob**2))
    log_likelihood = free_log_likelihood + fix_log_likelihood
    # log_likelihood = - (1.0 / (2 * (sigma_physic**2))) * physic_loss
    kl_div = model.kl_divergance()
    # kl_div = 0.0
    # jax.debug.breakpoint()
    elbo = log_likelihood - kl_div
    # elbo = log_likelihood
    total_loss = -elbo
    return total_loss, (log_likelihood, kl_div, sum_free_loss, sum_fix_loss)