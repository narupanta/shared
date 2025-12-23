import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import Adam


from core.datasetclass import TractionDataset
from core.utils import * 
from core.model import StrainGPModel, NHPrior, NHPriorFunc
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
@jax.vmap
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
@jax.jit
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


# @jax.checkpoint
def physical_loss(strain_energy_func, coords, cells, u,
                  n_nodes, node_type):
    """
    params: (mu, kappa)
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
    # p_pos = jnp.exp(params)
    # strain_energy = strain_energy_func(f, p_pos)
    # piola = jax.grad(strain_energy_func)
    # piola = P_mr(f, p_pos)[:, :2, :2]   # (C,2,2)
    invariant_stacked, dI_dF_stacked = invariants_and_derivatives(F)
    deriv_prior_mean = jax.vmap(jax.grad(strain_energy_func))(invariant_stacked)
    piola = jnp.einsum("cnij, cn -> cij", dI_dF_stacked, deriv_prior_mean) [: ,:2, :2]

    # piola = piola_func(f)[:, :2, :2] 
    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)    # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2)
    f_int_nodes = jnp.zeros((n_nodes, 2)).at[cells].add(f_int_cell)

    # --- NEUMANN EDGE-LENGTH TRACTION ---
    # normalize load_parameter to flat array
    t3 = 1.3
    t4 = 1.3 * 1.1
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

    # return blm_loss + fixed_nodes_loss1 + fixed_nodes_loss2, f_int_nodes, f_neu_nodes, R_nodes
    return blm_loss + fixed_nodes_loss1 + fixed_nodes_loss2


import jax.numpy as jnp
import jax

@jax.jit
def rbf_kernel(X1: jnp.array, X2: jnp.array, variance: float, lengthscales: jnp.array) -> jnp.array:
    """
    Computes the ARD RBF Gram/Covariance matrix.
    X1, X2 are (N, D) and (M, D).
    lengthscales is a vector (D,).
    """
    # 1. Calculate the raw difference tensor: (N, M, D)
    diff = X1[:, None, :] - X2[None, :, :]
    
    # 2. Apply the weighted (inverse lengthscale squared) distance for each dimension
    # This is the core ARD step: (x_d - x'_d)^2 / (l_d)^2
    # The exponent is a sum over the D dimension: (N, M)
    exponent = jnp.sum((diff / lengthscales)**2, axis=-1)
    
    # RBF formula: k(x, x') = sigma^2 * exp(- 0.5 * sum_d (x_d - x'_d)^2 / (l_d)^2)
    K = variance * jnp.exp(-0.5 * exponent)
    return K

class PosteriorStrainEnergyFunction :
    def __init__(self, lengthscales, variance, inducing_points, u_induced) :
        self.lengthscales = lengthscales 
        self.variance = variance
        self.inducing_points = inducing_points
        self.u_induced = u_induced
    def __call__(self, i_star) :
        if len(i_star.shape) != 2 :
            i_star = i_star[None, :]
        Kzz = rbf_kernel(self.inducing_points, self.inducing_points, self.variance, self.lengthscales)
        Kzz += 1e-6 * jnp.eye(self.inducing_points.shape[0])
        Kiz = rbf_kernel(i_star, self.inducing_points, self.variance, self.lengthscales)
        Kzz_inv = jnp.linalg.solve(Kzz, jnp.eye(self.inducing_points.shape[0]))
        Psi_I = Kiz @ Kzz_inv @ self.u_induced
        return Psi_I[0]
def model(u_obs, Z_I, beta=1e-2):
    M = Z_I.shape[0]

    # Kernel hyperparameters (MAP)
    lengthscales = numpyro.param(
        "lengthscales", jnp.array([1.0, 1.0, 1.0]), constraint=dist.constraints.positive
    )
    variance = numpyro.param(
        "variance", 1.0, constraint=dist.constraints.positive
    )
    nh_mean_func = jax.vmap(NHPriorFunc(mu = 0.5, kappa = 2.0))
    Kzz = rbf_kernel(Z_I, Z_I, variance, lengthscales)
    Kzz += 1e-6 * jnp.eye(M)
    u_ = nh_mean_func(Z_I)
    u = numpyro.sample(
        "u",
        dist.MultivariateNormal(u_, Kzz)
        # dist.MultivariateNormal(jnp.zeros(Kzz.shape[0]), Kzz)
    ) # define as nh, mr prior
    post_psi_func = PosteriorStrainEnergyFunction(lengthscales, variance, Z_I, u)



    # # GP prior

    # # initial_u = nh_prior_mean(Z_I)
    # u = numpyro.sample(
    #     "u",
    #     dist.MultivariateNormal(u_, Kzz)
    # ) # define as nh, mr prior


    # Physics residual
    r = physical_loss(post_psi_func, coord_cells, cells, u_cells,
                  coords.shape[0], node_type)   # <-- deterministic functional (VFM)
    # Energy-based likelihood
    numpyro.factor("physics", -beta * r)

def guide(u_obs, Z_I, beta=1.0):
    M = Z_I.shape[0]
    nh_mean_func = jax.vmap(NHPriorFunc(mu = 1.0, kappa = 3.0))
    u = nh_mean_func(Z_I)
    m = numpyro.param("u_loc", u)
    L = numpyro.param(
        "u_scale_tril",
        jnp.eye(M),
        constraint=dist.constraints.lower_cholesky
    )

    numpyro.sample(
        "u",
        dist.MultivariateNormal(m, scale_tril=L)
    )


if __name__ == "__main__" :
    dataset = TractionDataset("dataset","NH")
    data = dataset[1]
    coords = data["mesh_pos"][:,:2]
    cells = data["cells"]
    u = data["u"]
    node_type = data["node_type"]
    load_parameter = data["load_parameter"]

    coord_cells = coords[cells]
    u_cells = u[cells]

    F, dNdx = deformation_gradient_element(coord_cells, u_cells)
    invariant_stacked, dI_dF_stacked = invariants_and_derivatives(F)

    I_obs = invariant_stacked
    inducing_points = 960

    Z_i1 = jnp.linspace(I_obs[:, 0].min(), I_obs[:, 0].max(), inducing_points)
    Z_i2 = jnp.linspace(I_obs[:, 1].min(), I_obs[:, 1].max(), inducing_points)
    Z_i3 = jnp.linspace(I_obs[:, 2].min(), I_obs[:, 2].max(), inducing_points)

    Z_stacked = jnp.stack([Z_i1, Z_i2, Z_i3], axis = -1)

    optimizer = Adam(1)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optimizer, Trace_ELBO())

    state = svi.init(
        jax.random.PRNGKey(0),
        u, I_obs, Z_stacked
    )
    # svi.run()
    # for step in range(500):
    #     svi.run(jax.random.PRNGKey(0))
    #     svi_state, loss = svi.update(
    #         svi_state, u, I_obs, Z_stacked
    #     )
    #     if step % 1 == 0:
    #         print(f"step {step}, loss = {loss:.2f}")
    num_iterations = 10000
    chunk_size = 20
    for i in range(0, num_iterations, chunk_size):
        # Run a chunk of steps
        # We pass the 'state' from the previous chunk into the next one
        svi_result = svi.run(jax.random.PRNGKey(i), chunk_size, u, I_obs, Z_stacked, init_state=state)
        
        # Update the current state to the last state of the chunk
        state = svi_result.state
        # svi_result.params["auto_loc"]
        # svi_result.params["lengthscales"]
        # svi_result.params["variance"]
        post_psi_func = PosteriorStrainEnergyFunction(svi_result.params["lengthscales"], svi_result.params["variance"], Z_stacked, svi_result.params["auto_loc"])



        # # GP prior

        # # initial_u = nh_prior_mean(Z_I)
        # u = numpyro.sample(
        #     "u",
        #     dist.MultivariateNormal(u_, Kzz)
        # ) # define as nh, mr prior


        # Physics residual
        r = physical_loss(post_psi_func, coord_cells, cells, u_cells,
                    coords.shape[0], node_type)
        print(r)
        print(f"Iteration {i + chunk_size}: Loss {svi_result.losses[-1]}")