import numpy as np
import matplotlib.pyplot as plt
import jax

jax.config.update("jax_platform_name", "cpu")
from mpi4py import MPI
import ufl
from dolfinx import io, fem
from dolfinx.common import timing, list_timings
from dolfinx_materials.jaxmat import JAXMaterial
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem

import jaxmat.materials as jm

from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import numpy as np
from dolfinx.io.gmsh import read_from_msh
import ufl
import gmsh
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from create_geometry import generate_perforated_plate
import jax
import jax.numpy as jnp
from dolfinx_materials.utils import nonsymmetric_tensor_to_vector, project
def fem_solve() :
    mesh_data = generate_perforated_plate(1.0, 1.0, 0.05, (0.03, 0.03))
    domain = mesh_data.mesh
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    def left(x):
        return np.isclose(x[0], 0)
    def bottom(x):
        return np.isclose(x[1], 0)
    def right(x):
        return np.isclose(x[0], 1.0)
    def top(x):
        return np.isclose(x[1], 1.0)

    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, left)
    right_facets = mesh.locate_entities_boundary(domain, fdim, right)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top)
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)

    marked_facets = np.hstack([left_facets, bottom_facets, right_facets, top_facets])
    marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(bottom_facets, 2), np.full_like(right_facets, 3), np.full_like(top_facets, 4)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )

    u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)

    # To apply the boundary condition, we identity the dofs located on the facets marked by the `MeshTag`.
    zero = fem.Constant(domain, default_scalar_type(0.0))
    # control_ux = fem.Constant(domain, default_scalar_type(load_parameter * 0))
    control_uy = fem.Constant(domain, default_scalar_type(0.1))

    # Locate DOFs (note the use of collapsed spaces)
    # left_dofs = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(1))
    bottom_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(2))
    # bottom_y_dofs = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(2))
    # right_dofs = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(3))
    top_y_dof = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(4))
    top_x_dof = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(4))
    bcs = [
        fem.dirichletbc(u_bc, bottom_dofs, V),
        fem.dirichletbc(control_uy, top_y_dof, V.sub(1)),
        fem.dirichletbc(zero, top_x_dof, V.sub(0))
        ]

    # Next, we define the body force on the reference configuration (`B`), and nominal (first Piola-Kirchhoff) traction (`T`).

    B = fem.Constant(domain, default_scalar_type((0, 0)))
    T = fem.Constant(domain, default_scalar_type((0, 0)))

    # Define the test and solution functions on the space $V$

    v = ufl.TestFunction(V)
    u = fem.Function(V)
    nh = jm.CompressibleNeoHookean(1.0, 3.0, jm.hyperelasticity.SquaredVolumetric())
    behavior = jm.Hyperelasticity(nh)
  
    material = JAXMaterial(behavior)
    qmap = QuadratureMap(domain, 1, material)

    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    Id2 = ufl.Identity(2)
    def F_2d_to_3d(u_):
        F_2d = Id2 + ufl.grad(u_)
        F_3d = ufl.as_matrix([[F_2d[0,0], F_2d[0,1], 0],[F_2d[1,0], F_2d[1,1], 0],[0,0,1]])
        return nonsymmetric_tensor_to_vector(F_3d)
    def dF(u_, v_):
        return ufl.derivative(F_2d_to_3d(u_), u_, v_)
    qmap.register_gradient("F", F_2d_to_3d(u))

    P = qmap.fluxes["PK1"]
    Res = ufl.inner(P, dF(u, v)) * qmap.dx
    Jac = qmap.derivative(Res, u, du)
    qmap.initialize_state(); qmap.update()

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_atol": 1e-6,
        "snes_rtol": 1e-6,
        "snes_monitor": None,
        "log_view": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = NonlinearMaterialProblem(
        qmap,
        Res,
        u,
        bcs=bcs,
        J=Jac,
        petsc_options_prefix="hyperelasticity",
        petsc_options=petsc_options,
    )
    problem.solve()
    return u.x.array

if __name__ == "__main__" :
    fem_solve()