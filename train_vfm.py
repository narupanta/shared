import optax
from tqdm import tqdm
# Import some useful modules.
import jax
import jax.numpy as jnp
import os

# Import JAX-FEM specific modules.
# from jax_fem.problem import Problem
# from jax_fem.solver import solver
# from jax_fem.utils import save_sol
# from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
import jax.random as jr 
jax.config.update("jax_enable_x64", False)

from core.utils import *
from core.model import SparseHyperelasticityGP
from core.material_models import get_material
from core.datasetclass import  TractionDataset
import jax

# List all devices JAX can use
devices = jax.devices()
print(f"Available devices: {devices}")

# Check the default device (where computations happen by default)
print(f"Default device: {jax.default_backend()}")

from core.material_models import BaseMaterialModel
class MooneyRivlin(BaseMaterialModel):
    def __init__(self, params, jit_P: bool = True):
        super().__init__(jit_P=jit_P)
        self.params = params
        self.dev_params = self.params[:-1]
        self.vol_param = self.params[-1]

    def phi(self, f) -> jnp.ndarray:
        I,_ = invariants_and_derivatives(f)
        i3 = I[2] + 1e-6
        i1_dev = i3 ** (-1/3) * I[0]
        i2_dev = i3 ** (-2/3) *I[1]

        X = i1_dev - 3.0
        Y = i2_dev - 3.0
        
        # --- Deviatoric Terms (W) ---
        # Assuming dev_params = [c01, c02, c10, c11, c12, c20, c21, c22]
        # Using the standard N=2 Polynomial Model terms (C10, C01, C20, C11, C02)
        dev_terms = (
            # C10 * X
            self.dev_params[0] * X + 
            # C01 * Y
            self.dev_params[1] * Y + 
            # C20 * X**2
            self.dev_params[2] * X**2 +
            # C11 * X * Y
            self.dev_params[3] * X * Y + 
            # C02 * Y**2
            self.dev_params[4] * Y**2  

            # self.dev_params[5] * X*Y**2 + 

            # self.dev_params[6] * X**2 * Y + 

            # self.dev_params[7] * X**2 * Y ** 2

            # Add C12, C21, C22 terms here if required by your specific model definition
        )
        
        # --- Volumetric Terms (U) ---
        # Assuming vol_params = [d0, d1] are D2 and D1 parameters (inverse bulk moduli)
        J = jnp.sqrt(i3)
        J_minus_1 = J - 1.0

        # Assuming the volumetric function U(J) = (1/D1)(J-1)^2 + (1/D2)(J-1)^4
        # with D1=d1 and D2=d0 (or vice versa, depending on convention)
        
        # D1 is typically the lower order term (quadratic, hence d1)
        # D2 is typically the higher order term (quartic, hence d0)
        vol_terms = (
            # (1/D1) * (J - 1)**2
            (self.vol_param) * J_minus_1**2 
        #     # (1/D2) * (J - 1)**4
        #     (self.vol_params[1]) * J_minus_1**4
        )
        
        return dev_terms + vol_terms
    
# trainable parameters
def _neumann_cell_force(coords_el, onehot_types_el, t3, t4):
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

def total_energy(u, psi, coords, cells):
    # gather per-element quantities
    u_cells = u[cells]            # (C, 3, 2)
    coord_cells = coords[cells]   # (C, 3, 2)

    # kinematics
    F, _ = deformation_gradient_element(coord_cells, u_cells)

    # element areas
    dA = jnp.linalg.det(
        transformation_jacobian(coord_cells)
    ) / 2.0                       # (C,)

    # strain energy density per element
    psi_e = psi(F)      # (C,)

    # total internal energy
    return jnp.sum(psi_e * dA)
def external_work(u, coords, cells, node_type, load):
    u_cells = u[cells]           # (C, 3, 2)
    coord_cells = coords[cells]  # (C, 3, 2)

    t3, t4 = load
    types = node_type[cells]     # (C, 3)

    # element nodal traction forces (C, 3, 2)
    f_neu_cells = jax.vmap(
        _neumann_cell_force, in_axes=(0, 0, None, None)
    )(coord_cells, types, t3, t4)

    # element external work
    # sum over nodes and dimensions
    W_ext_cells = jnp.sum(u_cells * f_neu_cells, axis=(1, 2))  # (C,)

    return jnp.sum(W_ext_cells)

def physical_loss(params, coords, cells, u,
                  n_nodes, node_type, load, reactions):
    material_model = MooneyRivlin(jnp.exp(params))
    psi = jax.vmap(material_model.phi)

    f_int_nodes = jax.grad(total_energy)(u, psi, coords, cells)

    # f_ext_nodes = jax.grad(external_work)(u, coords, cells, node_type, load)
    f_ext_nodes = jnp.zeros((n_nodes, 2))
    R_nodes = f_int_nodes - f_ext_nodes
    free_mask = ((node_type[:, 1] != 1) &
             (node_type[:, 2] != 1))
    fix_x_mask = (node_type[:, 1] == 1)
    fix_y_mask = (node_type[:, 2] == 1)
    r_free0 = jnp.sum((R_nodes**2) *
                    free_mask[:, None])
    r_free1 = jnp.sum(
        (R_nodes[:, 1]**2) * fix_x_mask
    )

    r_free2 = jnp.sum(
        (R_nodes[:, 0]**2) * fix_y_mask
    )
    # r_free0 = R_nodes[(node_type[:, 1] != 1) & (node_type[:, 2] != 1)]
    # r_free1 = R_nodes[node_type[:, 1] == 1, 1]
    # r_free2 = R_nodes[node_type[:, 2] == 1, 0]
    free_r_total = r_free0 + r_free1 + r_free2
    ext_force = jnp.sum(f_ext_nodes, axis = 0)
    fixed_nodes_loss1 = (jnp.sum(R_nodes[:,0] * fix_x_mask) + ext_force[0])**2
    fixed_nodes_loss2 = (jnp.sum(R_nodes[:,1] * fix_y_mask) + ext_force[1])**2 
    reaction_loss = fixed_nodes_loss1 + fixed_nodes_loss2
    return free_r_total  + reaction_loss, (jnp.sum(R_nodes[:,0] * fix_x_mask) , jnp.sum(R_nodes[:,1] * fix_y_mask), ext_force[0], ext_force[1])
 
def total_physical_loss(n_t, params, coords, cells, u_list, n_nodes, node_type, loads, reactions) :

    total_loss, (fnl1, fnl2, totalnl1, totalnl2) = jax.vmap(physical_loss, in_axes = (None, None, None, 0, None, None, 0, 0))(params, coords, cells, u_list, n_nodes, node_type, loads, reactions)

    return jnp.sum(total_loss),  (jnp.mean(fnl1), jnp.mean(fnl2), jnp.mean(totalnl1), jnp.mean(totalnl2)) 
 

if __name__ == "__main__" :
    # base_save_path = "saved_model"  # change as needed
    # os.makedirs(base_save_path, exist_ok=True)

    # # Subfolder with datetime
    # timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    # save_path = os.path.join(base_save_path, timestamp)
    # os.makedirs(save_path, exist_ok=True)
    material_model = "isihara_fix"
    dataset = TractionDataset("dataset",material_model)
    F_all = []
    u_list = []
    load_parameters = []
    reaction_list = []
    loads = []
    n_t = len(dataset)
    for loadstep in range(0, n_t) :
        # if (loadstep != n_t - 1) :
        #     continue 
        data = dataset[loadstep]
        coords = data["mesh_pos"][:,:2]
        cells = data["cells"]
        # u = data["u"]
        percent_noise = 0.0001
        node_type = data["node_type"]
        ux = data["u"][:, 0]
        ux[(data["node_type"][:, 1] != 1)] += np.random.normal(0, percent_noise * 1, ux.shape)[(data["node_type"][:, 1] != 1)]
        # ux[(data["node_type"] != 3)] += np.random.normal(0, percent_noise * 1, ux.shape)[(data["node_type"] != 3)]
        uy = data["u"][:, 1]
        uy[(data["node_type"][:, 2] != 1)] += np.random.normal(0, percent_noise * 1, uy.shape)[(data["node_type"][:, 2] != 1)]
        # uy[(data["node_type"] != 4)] += np.random.normal(0, percent_noise * 1, uy.shape)[(data["node_type"] != 4)]

        # Combine components into the full displacement vector u
        u = np.column_stack((ux, uy))
        # u[node_type == 0] = u[node_type == 0] + jax.random.normal(jr.key(0), u.shape)[node_type == 0] * 0.01 * mean_u
        load_parameter = data["load_parameter"]
        # u = denoise(coords, u_noisy)
        coord_cells = coords[cells]
        u_cells = u[cells]

        F, dNdx = deformation_gradient_element(coord_cells, u_cells)
        load_parameters.append(load_parameter)
        u_list.append(u)
        F_all.append(F)
        reaction_list.append(data["reaction"])
        loads.append(data["load"])
        # reaction_list.append(jnp.array([0, 0, 0, 0]))
    # I_obs_all,_ = invariants_and_derivatives_vmap(F_all_stacked)
    coords = jax.device_put(jnp.array(coords))
    cells = jax.device_put(jnp.array(cells))
    node_type = jax.device_put(jnp.array(node_type))

    u_list = jax.device_put(jnp.array(u_list))
    loads = jax.device_put(jnp.array(loads))
    reaction_list = jax.device_put(jnp.array(reaction_list))

    # u_list = jnp.array(u_list)
    # load_parameters = jnp.array(load_parameters)
    # reaction_list = jnp.array(reaction_list)
    # loads = jnp.array(loads)
    I_obs, _ = jax.vmap(invariants_and_derivatives)(F)
    params = jnp.zeros(6) 

    # choose optimizer
    # opt = optax.lbfgs(1e-3)
    opt = optax.adam(5e-3)
    opt_state = opt.init(params)

    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda params: total_physical_loss(n_t, params, coords, cells, u_list, coords.shape[0], node_type, loads, reaction_list),
        has_aux=True
    ))
    @jax.jit
    def train_step(opt_state, params) :
        (loss, (fnl1, fnl2, totalnl1, totalnl2)), grads = loss_and_grad(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, loss, params
    bar = tqdm(range(50000))
    for step in bar :
        opt_state, loss, params = train_step(opt_state, params)
        bar.set_postfix({"step": f"{step:04d}", "loss":f"{loss:.6f}", 
                        #  "fnl1":f"{fnl1:.6f}", 
                        #  "fnl2":f"{fnl2:.6f}", 
                        #  "totalnl1":f"{totalnl1:.6f}", 
                        #  "totalnl2":f"{totalnl2:.6f}",
                        "params": f"{jnp.exp(params)}"})
# print(f"step {step:04d}  loss={loss:.6f} {jnp.exp(params)}")
    # @jax.jit
    # def train_block(carry, _):
    #     """This function runs entirely on the GPU without CPU interference."""
    #     opt_state, params = carry
    #     (loss, aux), grads = loss_and_grad(params)
    #     updates, opt_state = opt.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)
    #     return (opt_state, params), loss

    # # Configuration
    # total_steps = 50000
    # block_size = 100  # We update tqdm every 100 iterations
    # num_blocks = total_steps // block_size

    # bar = tqdm(total=total_steps)
    # for _ in range(num_blocks):
    #     # jax.lax.scan runs 100 steps in one go on the GPU
    #     (opt_state, params), block_losses = jax.lax.scan(
    #         train_block, (opt_state, params), None, length=block_size
    #     )
        
    #     # Get the last loss from the block for the display
    #     last_loss = block_losses[-1]
        
    #     # Update tqdm once per block instead of once per step
    #     bar.update(block_size)
    #     bar.set_postfix({
    #         "loss": f"{last_loss:.6f}",
    #         "p_sum": f"{jnp.sum(jnp.exp(params)):.2f}" # Keep display simple
    #     })