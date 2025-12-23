# Import some useful modules.
import jax
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import rectangle_mesh, get_meshio_cell_type, Mesh
import torch
# import numpy as np
from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI
from core.utils import *
import jax.numpy as jnp
import numpy as np
from core.model import TensorBasisGPModel, SVTBGPModel
import gpjax as gpx
# Define constitutive relationship.
def F2x2_to_F3x3(f):
    # f is 2×2
    return jnp.array([
        [f[0,0], f[0,1], 0.0],
        [f[1,0], f[1,1], 0.0],
        [0.0,    0.0,    1.0],
    ])
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad)
    # to obtain the 'P_fn' function.
    def __init__(self, model, **kwargs) :
        super().__init__(**kwargs)
        self.model = model
    def get_tensor_map(self):

        def P_fn(f2):
            F = F2x2_to_F3x3(f2)[None, ...]     # convert 2×2 → 3×3
            P3 = self.model.predict_piola_stress(F)[0]  # model outputs 3×3
            return P3[0][:2, :2]
        # def psi(F):
        #     # mu = E / (2. * (1. + nu))
        #     # kappa = E / (3. * (1. - 2. * nu))
        #     J = jnp.linalg.det(F)
        #     Jinv = J**(-2. / 3.)
        #     I1 = jnp.trace(F.T @ F)
        #     energy = (1 / 2.) * (Jinv * I1 - 3.) + (3 / 2.) * (J - 1.)**2.
        #     return energy
        # P_fn = jax.grad(psi)

        # def MooneyRivlinPhi(f):
        #     B_train = B_func(f)
        #     I1_train = I1_func(B_train)
        #     I2_train = I2_func(B_train)
        #     I3_train = I3_func(B_train)
        #     """Mooney–Rivlin strain energy function."""
        #     c1, c2, c3 = 0.162, 0.0059, 10.0
        #     term1 = c1 * (I3_train**(-0.5) * I1_train - 3)
        #     term2 = c2 * (I3_train**(-2/3) * I2_train - 3)
        #     term3 = c3 * (I3_train**0.5 - 1)**2
        #     return term1 + term2 + term3  # [N]
        # P_fn = jax.grad(MooneyRivlinPhi)
        # def P_fn(F) :
        #     E = 10.
        #     nu = 0.3
        #     mu = E / (2. * (1. + nu))
        #     kappa = E / (3. * (1. - 2. * nu))
        #     J = jnp.linalg.det(F)
        #     Jinv = J**(-2. / 3.)
        #     I1 = jnp.trace(F.T @ F)
        #     Jinv = J**(-2. / 3.)
        #     term1 = mu * Jinv * F
        #     term2 = (-mu/3 * J ** (-2./3.) * I1 + kappa * J * (J - 1)) * jnp.linalg.inv(F.T)
        #     return term1 + term2


        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
msh_data = read_from_msh("mesh_with_hole.msh", MPI.COMM_WORLD, 0, 2)  # 2D mesh
domain = msh_data.mesh

# 2. Extract node coordinates (geometry points)
# shape (num_nodes, gdim)
node_coords = domain.geometry.x  # numpy array
num_nodes, gdim = node_coords.shape
print(f"Nodes: {num_nodes}, dimension: {gdim}")

# 3. Extract cell connectivity (triangles)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, 0)
cells = domain.topology.connectivity(tdim, 0).array.reshape(-1, 3)  # triangles
print(f"Cells: {cells.shape[0]}")



# Specify mesh-related information (first-order hexahedron element).
ele_type = 'TRI3'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 1., 1., 1.
# mesh = rectangle_mesh(Nx=20,
#                        Ny=20,
#                        domain_x=Lx,
#                        domain_y=Ly)
mesh = Mesh(node_coords[:, :2], cells)


# Define boundary locations.
def bottom(point):
    return jnp.isclose(point[1], 0., atol=1e-6)
def left(point):
    return jnp.isclose(point[0], 0., atol=1e-6)


def top(point):
    return jnp.isclose(point[1], 1., atol=1e-6)

def right(point):
    return jnp.isclose(point[0], 1., atol=1e-6)


# Define value function.
def zero_dirichlet_val(point):
    return 0.


def dirichlet_val_top(point):
    return 0.3

def dirichlet_val_right(point):
    return 0.0

# dirichlet_bc_info = [
#     [bottom] * 2 + [top] * 2, 
#     [0, 1] * 2,
#     [zero_dirichlet_val] * 2 + [zero_dirichlet_val, dirichlet_val_top]
                    #  ]
# dirichlet_bc_info = [
#     [left] + [bottom] + [top] + [right], 
#     [0, 1] + [1] + [0],
#     [zero_dirichlet_val] * 2 + [dirichlet_val_top] + [dirichlet_val_right]]
dirichlet_bc_info = [
    [bottom] * 2 + [top] * 2, 
    [0, 1] + [1] + [0],
    [zero_dirichlet_val] * 2 + [dirichlet_val_top] + [dirichlet_val_right]]
model_date_time = "20251117T232502"
model_path = f"saved_model/{model_date_time}"  # change as needed


# model_eval = TensorBasisGPModel(means, kernels, None, None)
model_eval = SVTBGPModel()
model_eval.load_model(model_path)
# Create an instance of the problem.
problem = HyperElasticity(mesh = mesh,
                          vec=2,
                          dim=2,
                          ele_type=ele_type,
                          dirichlet_bc_info=dirichlet_bc_info,
                          model = model_eval)


# Solve the defined problem.
data_dir = ""
petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_monitor": None,
        "snes_atol": 1e-10,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
# sol_list = solver(problem, solver_options = {'jax_solver': {'precond': True}})
sol_list = solver(problem,         
                  solver_options = {
                      'petsc_solver': petsc_options
        })

u = sol_list[0]
x = node_coords[:, 0]
y = node_coords[:, 1]
ux = u[:, 0]
uy = u[:, 1]
umag = jnp.sqrt(ux**2 + uy**2)

# triangulation object
triang = tri.Triangulation(x, y, cells)

# deformed mesh nodes
scale = 1.0   # deformation scale factor
x_def = x + scale * ux
y_def = y + scale * uy
triang_def = tri.Triangulation(x_def, y_def, cells)

# ---------------------------------------------------
# Plot
# ---------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# ---- 1) ux contour ----
tcf1 = axs[0,0].tricontourf(triang, ux, levels=30)
axs[0,0].set_title('$u_x$')
fig.colorbar(tcf1, ax=axs[0,0])
axs[0,0].set_aspect('equal')

# ---- 2) uy contour ----
tcf2 = axs[0,1].tricontourf(triang, uy, levels=30)
axs[0,1].set_title('$u_y$')
fig.colorbar(tcf2, ax=axs[0,1])
axs[0,1].set_aspect('equal')

# ---- 3) |u| magnitude contour ----
tcf3 = axs[1,0].tricontourf(triang, umag, levels=30)
axs[1,0].set_title(r'$|u| = \sqrt{u_x^2 + u_y^2}$')
fig.colorbar(tcf3, ax=axs[1,0])
axs[1,0].set_aspect('equal')

# ---- 4) original + deformed mesh ----
axs[1,1].triplot(triang, color="gray", linewidth=0.5, label="original")
axs[1,1].triplot(triang_def, color="red", linewidth=0.7, label="deformed")
axs[1,1].set_title("Deformed over original mesh")
axs[1,1].legend()
axs[1,1].set_aspect('equal')

plt.tight_layout()
plt.savefig(f"poc_jaxfem_GP_NH_GT_{model_date_time}.png")
plt.close()