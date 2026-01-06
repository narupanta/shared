# Import some useful modules.
import jax
import jax.numpy as jnp
import os

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io.gmsh import read_from_msh
import ufl
import gmsh
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from core.utils import fto3x3
# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad)
    # to obtain the 'P_fn' function.
    def get_surface_maps(self):
        def surface_map(u, x):
            return jnp.array([0., 0.1])
        return [surface_map]

    def set_params(self, params):
        surface_params = params
        # Generally, [[surface1_params1, surface1_params2, ...], [surface2_params1, surface2_params2, ...], ...]
        self.internal_vars_surfaces = [[surface_params]]
    def get_tensor_map(self):

        def psi(F):
            f = fto3x3(F)
            J = jnp.linalg.det(f)
            Jinv = (J)**(-2. / 3.)
            I1 = jnp.trace(f.T @ f)
            energy = 0.5 * (Jinv * I1 - 3.) + 1.5 * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = jnp.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
# Specify mesh-related information (first-order hexahedron element).
msh_data = read_from_msh("mesh_with_hole.msh", MPI.COMM_WORLD, 0, 2) 
domain = msh_data.mesh

node_coords = domain.geometry.x[:, :2]  # numpy array
num_nodes, gdim = node_coords.shape
# 3. Extract cell connectivity (triangles)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, 0)
cells = domain.geometry.dofmap
ele_type = 'TRI3'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')

mesh = Mesh(node_coords, cells)


# Define boundary locations.
def left(point):
    return jnp.isclose(point[0], 0., atol=1e-6)
def bottom(point):
    return jnp.isclose(point[1], 0., atol=1e-6)
def right(point):
    return jnp.isclose(point[0], 1., atol=1e-6)
def top(point):
    return jnp.isclose(point[1], 1.0, atol=1e-6)

zero_dbc = lambda point : 0
top_dbc = lambda point : 0.1
dirichlet_bc_info = [
    [bottom] * 2 , 
    [0, 1],
    [zero_dbc, zero_dbc]
                     ]


# Create an instance of the problem.
problem = HyperElasticity(mesh,
                          vec=2,
                          dim=2,
                          ele_type=ele_type,
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns = [top])
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

# Solve the defined problem.
sol_list = solver(problem, solver_options={'petsc_solver': petsc_options})
# Store the solution to local file.
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)