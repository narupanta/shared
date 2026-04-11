import jax
import jax.numpy as jnp
import jax.random as jr
from .utils import deformation_gradient_element, transformation_jacobian, invariants_and_derivatives, fto3x3
from .model import SparseHyperelasticityGP
from .material_models import BaseMaterialModel
from .dataclass import GPRawParams, GPParams

def total_stochastic_loss(p, model: SparseHyperelasticityGP, f3x3: jnp.ndarray, cells, n_nodes, f_neu_nodes, node_type, dNdX, dA, key: jnp.array, n_s) :
    model.params = model.load_params(p)
    model.gpweight = model.precompute_weights(p)

    main_key = jr.split(key, n_s + 1)
    subkey = main_key[1:]

    piola2x2 = lambda f,k : model.piola(f, k)[:2, :2]
    piola_cells = jax.vmap(piola2x2, in_axes=(0, None))
    piola_steps = jax.vmap(piola_cells, in_axes=(0, None))
    piola_sampling = jax.vmap(piola_steps, in_axes=(None, 0))
    piola2x2_cells = piola_sampling(f3x3, subkey)

    # vmapped_ell
    vmapped_ell = jax.vmap(ell, in_axes=(None, None, None, None, None, 0, None, None))
    ell_, (free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, sum_free_loss, sum_fix_loss) = vmapped_ell(model.params, cells, n_nodes, f_neu_nodes, node_type, piola2x2_cells, dNdX, dA)
    kl_div = model.kl_divergence()
    total_loss = -jnp.mean(ell_) + kl_div
    return total_loss, (jnp.mean(ell_), kl_div, jnp.mean(free_x_log_likelihood), jnp.mean(free_y_log_likelihood), jnp.mean(fix_x_log_likelihood), jnp.mean(fix_y_log_likelihood), jnp.mean(sum_free_loss), jnp.mean(sum_fix_loss))

def ell(p, cells, n_nodes, f_neu_nodes, node_type, piola2x2_cells, dNdX, dA) :
    sigma_free_x = p.sigma_free_x
    sigma_free_y = p.sigma_free_y
    sigma_fix_x = p.sigma_fix_x
    sigma_fix_y = p.sigma_fix_y

    free_loss, fix_loss = jax.vmap(vfm_loss, in_axes=(None, None, 0, None, 0, None, None))(cells, n_nodes, f_neu_nodes, node_type, piola2x2_cells, dNdX, dA)

    fix_x_loss = fix_loss[:, 0]
    fix_y_loss = fix_loss[:, 1]
    free_x_loss = free_loss[:, :, 0]
    free_y_loss = free_loss[:, :, 1]

    n_freedofs = free_loss.shape[0] * free_loss.shape[1]
    n_fixedofs = fix_loss.shape[0] * fix_loss.shape[1]

    free_x_log_likelihood = - (1.0 / (2 * (sigma_free_x**2))) * jnp.sum(free_x_loss**2) - n_freedofs/2.0 * jnp.log(2 * jnp.pi * (sigma_free_x**2))
    free_y_log_likelihood = - (1.0 / (2 * (sigma_free_y**2))) * jnp.sum(free_y_loss**2) - n_freedofs/2.0 * jnp.log(2 * jnp.pi * (sigma_free_y**2))
    fix_x_log_likelihood = - (1.0 / (2 * (sigma_fix_x**2))) * jnp.sum(fix_x_loss**2) - n_fixedofs/2.0 * jnp.log(2 * jnp.pi * (sigma_fix_x**2))
    fix_y_log_likelihood = - (1.0 / (2 * (sigma_fix_y**2))) * jnp.sum(fix_y_loss**2) - n_fixedofs/2.0 * jnp.log(2 * jnp.pi * (sigma_fix_y**2))

    expected_log_likelihood = free_x_log_likelihood + free_y_log_likelihood + (fix_x_log_likelihood + fix_y_log_likelihood)
    return expected_log_likelihood, (free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, jnp.sum(free_loss**2) , jnp.sum(fix_loss**2))

def total_physical_loss(u_array, loads, piola_func, coords, cells, node_type) :

    plpl = jax.vmap(physical_loss_per_loadstep_force_controlled, in_axes=(0, 0, None, None, None, None))

    free_node_residual, reaction_loss = plpl(u_array, loads, piola_func, coords, cells, node_type)
    return free_node_residual, reaction_loss

def vfm_loss(cells, n_nodes, f_neu_nodes, node_type, piola2x2, dNdx, dA) :

    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola2x2, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)    # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2)
    f_int_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_int_cell)

    # --- Residual R = int(grad v : P) dx  -  int(v·T) ds(Neumann)
    R_nodes = f_int_nodes - f_neu_nodes
    free_node_in = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
    free_node_on_dbc_left = (node_type[:, 1] == 1)
    free_node_on_dbc_bottom = (node_type[:, 2] == 1)
    # only free DOFs contribute to the residual loss (bc == 0)
    free_dof_domain_loss = R_nodes[free_node_in]
    free_dof_on_dbc_left_loss = R_nodes[free_node_on_dbc_left, 1]
    free_dof_on_dbc_bottom_loss = R_nodes[free_node_on_dbc_bottom, 0]
    neu_nodes_right = (node_type[:, 3] == 1)
    neu_nodes_top = (node_type[:, 4] == 1)
    total_traction_force = f_neu_nodes[neu_nodes_right|neu_nodes_top].sum(axis = 0)
    fixed_nodes_loss1 = jnp.sum(R_nodes[node_type[:, 1] == 1, 0]) + total_traction_force[0]
    fixed_nodes_loss2 = jnp.sum(R_nodes[node_type[:, 2] == 1, 1]) + total_traction_force[1]

    free_x_loss = jnp.concat([free_dof_domain_loss[:, 0], free_dof_on_dbc_bottom_loss]) 
    free_y_loss = jnp.concat([free_dof_domain_loss[:, 1], free_dof_on_dbc_left_loss])

    free_loss = jnp.stack([free_x_loss, free_y_loss], axis = -1)
    fix_loss = jnp.stack([fixed_nodes_loss1, fixed_nodes_loss2])
    return free_loss, fix_loss


def neumann_cell_force(coords_el, onehot_types_el, t3, t4):
    """
    onehot_types_el: (3, 5) array - one-hot encoded types for 3 nodes
    Columns: [0: Internal, 1: FixX, 2: FixY, 3: Right(t3), 4: Top(t4)]
    """
    edges = jnp.array([[0, 1], [1, 2], [2, 0]])
    f_cell = jnp.zeros((3, 2))

    for idx in range(3):
        i, j = edges[idx]
        
        # Check if BOTH nodes on this edge share the Neumann 'Right' bit (index 3)
        is_right = (onehot_types_el[i, 3] == 1) & (onehot_types_el[j, 3] == 1)
        
        # Check if BOTH nodes on this edge share the Neumann 'Top' bit (index 4)
        is_top = (onehot_types_el[i, 4] == 1) & (onehot_types_el[j, 4] == 1)

        L = jnp.linalg.norm(coords_el[j] - coords_el[i])

        # Apply forces independently
        # The corner edge connecting a 'Type 3' and a 'Type 4' node 
        # will now correctly pass the check if you set the corner's 
        # one-hot bits for both 3 and 4 to 1.
        f_cell = f_cell.at[i, 0].add(jnp.where(is_right, 0.5 * L * t3, 0.0))
        f_cell = f_cell.at[j, 0].add(jnp.where(is_right, 0.5 * L * t3, 0.0))
        
        f_cell = f_cell.at[i, 1].add(jnp.where(is_top, 0.5 * L * t4, 0.0))
        f_cell = f_cell.at[j, 1].add(jnp.where(is_top, 0.5 * L * t4, 0.0))

    return f_cell

# Support codes
######################################################################################################################################################################

# def total_stochastic_loss(p, model: SparseHyperelasticityGP, u_array: jnp.ndarray, loads: jnp.ndarray, coords: jnp.ndarray, cells: jnp.ndarray, node_type: jnp.ndarray, keys: jnp.array) :
#     model.params = model.load_params(p)
#     model.gpweight = model.precompute_weights(p)
#     sigma_free_x = model.params.sigma_free_x
#     sigma_free_y = model.params.sigma_free_y
#     sigma_fix_x = model.params.sigma_fix_x
#     sigma_fix_y = model.params.sigma_fix_y
#     # sigma_free_x = 1e-3
#     # sigma_free_y = 1e-3
#     # sigma_fix_x = 1e-3
#     # sigma_fix_y = 1e-3
#     vmapped_ell = jax.vmap(ell, in_axes=(None, None, None, None, None, None, None, None, None, None, 0))
#     ell_, (free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, sum_free_loss, sum_fix_loss) = vmapped_ell(sigma_free_x, sigma_free_y, sigma_fix_x, sigma_fix_y, u_array, loads, model, coords, cells, node_type, keys)
#     kl_div = model.kl_divergence()
#     total_loss = -jnp.mean(ell_) + kl_div
#     return total_loss, (jnp.mean(ell_), kl_div, jnp.mean(free_x_log_likelihood), jnp.mean(free_y_log_likelihood), jnp.mean(fix_x_log_likelihood), jnp.mean(fix_y_log_likelihood), jnp.mean(sum_free_loss), jnp.mean(sum_fix_loss))


def physical_loss_per_loadstep_force_controlled(u: jnp.ndarray, load: jnp.ndarray, piola_func, coords: jnp.ndarray, cells: jnp.ndarray, node_type: jnp.ndarray) :
    u_cells = u[cells]
    coord_cells = coords[cells]
    n_nodes = coords.shape[0]
    F, dNdx = deformation_gradient_element(coord_cells, u_cells)   # (C,2,2), (C,3,2,2?) matches your API
    dA = jnp.linalg.det(transformation_jacobian(coord_cells)) / 2  # (C,)

    
    f = jax.vmap(fto3x3)(F)
    piola = jax.vmap(piola_func)(f)
    piola2x2 = piola[:, :2, :2]

    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola2x2, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)    # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2)
    f_int_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_int_cell)
    
    # --- NEUMANN EDGE-LENGTH TRACTION ---
    # normalize load_parameter to flat array
    t3, t4 = load
    types = node_type[cells]     # (C, 3)

    # element nodal traction forces (C, 3, 2)
    f_neu_cells = jax.vmap(
        neumann_cell_force, in_axes=(0, 0, None, None)
    )(coord_cells, types, t3, t4)

    # assemble global neumann nodal forces
    f_neu_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_neu_cells)

    # --- Residual R = int(grad v : P) dx  -  int(v·T) ds(Neumann)
    R_nodes = f_int_nodes - f_neu_nodes
    free_node_in = (node_type[:, 1] != 1) & (node_type[:, 2] != 1)
    free_node_on_dbc_left = (node_type[:, 1] == 1)
    free_node_on_dbc_bottom = (node_type[:, 2] == 1)
    # only free DOFs contribute to the residual loss (bc == 0)
    free_dof_domain_loss = R_nodes[free_node_in]
    free_dof_on_dbc_left_loss = R_nodes[free_node_on_dbc_left, 1]
    free_dof_on_dbc_bottom_loss = R_nodes[free_node_on_dbc_bottom, 0]
    neu_nodes_right = (node_type[:, 3] == 1)
    neu_nodes_top = (node_type[:, 4] == 1)
    total_traction_force = f_neu_nodes[neu_nodes_right|neu_nodes_top].sum(axis = 0)
    fixed_nodes_loss1 = jnp.sum(R_nodes[node_type[:, 1] == 1, 0]) + total_traction_force[0]
    fixed_nodes_loss2 = jnp.sum(R_nodes[node_type[:, 2] == 1, 1]) + total_traction_force[1]

    free_x_loss = jnp.concat([free_dof_domain_loss[:, 0], free_dof_on_dbc_bottom_loss]) 
    free_y_loss = jnp.concat([free_dof_domain_loss[:, 1], free_dof_on_dbc_left_loss])

    free_loss = jnp.stack([free_x_loss, free_y_loss], axis = -1)
    fix_loss = jnp.stack([fixed_nodes_loss1, fixed_nodes_loss2])
    return free_loss, fix_loss

def physical_loss_displacement_controlled(u, loads, piola_func, coords, cells, node_type) :

    u_cells = u[cells]
    coord_cells = coords[cells]
    n_nodes = coords.shape[0]
    # --- INTERNAL FORCES (unchanged) ---
    F, dNdx = deformation_gradient_element(coord_cells, u_cells)   # (C,2,2), (C,3,2,2?) matches your API
    dA = jnp.linalg.det(transformation_jacobian(coord_cells)) / 2  # (C,)
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

    fnl_left = (jnp.sum(f_int_nodes[node_type == 1, 0]) - loads[0])**2
    fnl_bottom = (jnp.sum(f_int_nodes[node_type == 2, 1]) - loads[1])**2
    fnl_right = (jnp.sum(f_int_nodes[node_type == 3, 0]) - loads[2])**2
    fnl_top = (jnp.sum(f_int_nodes[node_type == 4, 1]) - loads[3])**2

    reaction_loss = fnl_left + fnl_bottom + fnl_right + fnl_top

    return free_r_total, reaction_loss

def total_supervised_loss(p: GPRawParams, model: SparseHyperelasticityGP, true_psi: BaseMaterialModel, f_array: jnp.ndarray, key) :
    model.params = model.load_params(p)
    model.gpweight = model.precompute_weights(p)
    psi_func = jax.vmap(lambda f:model.psi(f, key))
    sigma_physic = model.params.sigma_phys_x
    f_array = jax.vmap(fto3x3)(f_array)
    pred_psi = psi_func(f_array)
    ell_, (sum_free_loss, sum_fix_loss) = supervised_ell(pred_psi, true_psi, sigma_physic)
    kl_div = model.kl_divergance()
    total_loss = -ell_ + kl_div
    return total_loss, (ell_, kl_div, sum_free_loss, sum_fix_loss)


def supervised_ell(pred_psi, true_psi, sigma_data) :
    n_data = pred_psi.shape[0]
    ell_ = -(1.0 / (2 * (sigma_data)**2)) * jnp.sum((pred_psi - true_psi)**2) - n_data/2.0 * jnp.log(2 * jnp.pi * (sigma_data**2))
    return ell_, (jnp.sum((pred_psi - true_psi)**2), sigma_data)