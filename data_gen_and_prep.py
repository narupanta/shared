
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import numpy as np
from dolfinx.io.gmsh import read_from_msh
import ufl
import gmsh
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from petsc4py import PETSc

import jax
import jax.numpy as jnp
import jax.random as jr
from core.utils import deformation_gradient_element, transformation_jacobian, invariants_and_derivatives, fto3x3
from core.model import SparseHyperelasticityGP
from core.material_models import BaseMaterialModel
from core.dataclass import GPRawParams, GPParams, PrecomputedVFMData
from core.loss_function import total_stochastic_loss, total_physical_loss, _neumann_cell_force


def facets_to_nodes(domain, facet_ids):
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, 0)
    facet_to_vertices = domain.topology.connectivity(fdim, 0)
    all_nodes = []
    for f in facet_ids:
        all_nodes.extend(facet_to_vertices.links(f))
    return np.unique(np.array(all_nodes, dtype=np.int32))

def FEM_solve(load_start, load_end, load_steps, mat_model = "neohookean") :
    import gmsh
    import sys

    gmsh.initialize()
    model = gmsh.model.occ

    # 1. Parameters
    L_x, L_y = 1.0, 1.0
    R_hole = 0.1
    mesh_size_far = 0.08     # Coarse at corners
    mesh_size_near = 0.02  # Very dense at circle

    # 2. Geometry
    rect = model.addRectangle(0.0, 0.0, 0.0, L_x, L_y)
    circle = model.addDisk(0.0, 0.0, 0.0, R_hole, R_hole)

    # Boolean Cut
    # returns [(2, tag)], [ [(2, tag)], ... ]
    out_tags, _ = model.cut([(2, rect)], [(2, circle)])
    model.synchronize()

    # 3. Automatic Hole Identification
    # We get all curves (dim=1) and find the one that is part of the hole
    all_curves = gmsh.model.getEntities(1)
    hole_curve_tag = []

    for dim, tag in all_curves:
        # Get the bounding box of the curve
        min_x, min_y, _, max_x, max_y, _ = gmsh.model.getBoundingBox(dim, tag)
        # If the curve is within the hole area, it's our target
        if max_x <= R_hole + 1e-6 and min_x >= -R_hole - 1e-6:
            if max_y <= R_hole + 1e-6 and min_y >= -R_hole - 1e-6:
                hole_curve_tag.append(tag)

    # 4. Define Distance Field on the Hole
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", hole_curve_tag)

    # 5. Define Threshold (The "Halo" Effect)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size_near)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size_far)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.02) # Fineness stays constant for this distance
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.25) # Gradually becomes coarse until this distance

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # 6. Strict Mesh Options
    # This prevents the outer boundary from dictating the mesh size
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # 7. Physical Groups & Generate
    surf_tag = out_tags[0][1]
    gmsh.model.addPhysicalGroup(2, [surf_tag], 1, name="domain")

    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")

    # Launch GUI to verif

    gmsh.finalize()

    # 8. Read into DOLFINx
    mesh_ = read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, 2)
    domain = mesh_.mesh
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    print(domain.geometry.dim)

    # -

    # We create two python functions for determining the facets to apply boundary conditions to


    # +
    def left(x):
        return np.isclose(x[0], 0)


    def right(x):
        return np.isclose(x[0], L_x)


    def top(x):
        return np.isclose(x[1], L_y)


    def bottom(x):
        return np.isclose(x[1], 0)


    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, left)
    right_facets = mesh.locate_entities_boundary(domain, fdim, right)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top)
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
    # -

    # Next, we create a  marker based on these two functions

    # Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two

    marked_facets = np.hstack([left_facets, bottom_facets, right_facets, top_facets])
    marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(bottom_facets, 2), np.full_like(right_facets, 3), np.full_like(top_facets, 4)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )

    # We then create a function for supplying the boundary condition on the left side, which is fixed.

    u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)

    # To apply the boundary condition, we identity the dofs located on the facets marked by the `MeshTag`.
    zero = fem.Constant(domain, default_scalar_type(0.0))

    # Locate DOFs (note the use of collapsed spaces)
    left_dofs = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(1))
    bottom_dofs = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(2))
    bottom_dofs_xy = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(2))
    right_dofs = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(3))
    top_dofs = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(4))
    # bcs = [
    #     fem.dirichletbc(zero, left_dofs, V.sub(0)), 
    #     fem.dirichletbc(zero, bottom_dofs, V.sub(1)),
    #     # fem.dirichletbc(u_bc, bottom_dofs_xy, V),
    #     # fem.dirichletbc(control_ux, right_dofs, V.sub(0)),
    #     # fem.dirichletbc(control_uy, top_dofs, V.sub(1))
    #     ]


    # Next, we define the body force on the reference configuration (`B`), and nominal (first Piola-Kirchhoff) traction (`T`).

    B = fem.Constant(domain, default_scalar_type((0, 0)))
    T_right = fem.Constant(domain, default_scalar_type((0.0, 0.0)))
    T_top = fem.Constant(domain, default_scalar_type((0.0, 0.0)))
    top_dis = fem.Constant(domain, default_scalar_type(0.0))
    right_dis = fem.Constant(domain, default_scalar_type(0.0))
    # Define the test and solution functions on the space $V$

    bcs = [
        fem.dirichletbc(zero, left_dofs, V.sub(0)), 
        fem.dirichletbc(zero, bottom_dofs, V.sub(1)),
        # fem.dirichletbc(top_dis, top_dofs, V.sub(1)), 
        # fem.dirichletbc(right_dis, right_dofs, V.sub(0)),
        ]
    
    v = ufl.TestFunction(V)
    u = fem.Function(V)

    # Define kinematic quantities used in the problem

    # +
    # Spatial dimension
    d = len(u)

    # Identity tensor
    I = ufl.variable(ufl.Identity(3))
    def grad_3D(u):
        return ufl.as_matrix([[u[0].dx(0), u[0].dx(1), 0], 
                            [u[1].dx(0), u[1].dx(1), 0], 
                            [0, 0, 0]])
    F = ufl.variable(I + grad_3D(u))
    # Deformation gradient
    # F = ufl.variable(I + ufl.grad(u))

    # Right Cauchy-Green tensor
    C = ufl.variable(F.T * F)

    # Invariants of deformation tensors
    Ic = ufl.variable(ufl.tr(C))
    I2 = ufl.variable(0.5 * (ufl.tr(C)**2 - ufl.tr(C * C)))
    J = ufl.variable(ufl.det(F))
    # -

    # Define the elasticity model via a stored strain energy density function $\psi$,
    # and create the expression for the first Piola-Kirchhoff stress:

    # Elasticity parameters

    # Stored strain energy density (compressible neo-Hookean model)

    # psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2
    if mat_model == "neohookean" :
        psi = 0.5 * (J**(-2/3) * Ic - 3) + 1.5 * (J - 1)**2 # neohookean
    elif mat_model == "isihara" :
        psi =  0.5 * (J**(-2/3) * Ic - 3) + (J**(-2/3) * Ic - 3)**2 +  (J**(-4/3) * I2 - 3) + 1.5 * (J - 1)**2 #isihara
    elif mat_model == "gentthomas" :
        psi = 0.5 * (J**(-2/3) *Ic - 3) + ufl.ln(J**(-4/3) * I2/3) + 1.5 * (J - 1)**2 #Gent Thomas
    else :
        raise ValueError("Unknown material model")
    P = ufl.diff(psi, F)

    metadata = {"quadrature_degree": 4}
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)

    # Define the residual of the equation (we want to find u such that residual(u) = 0)

    residual = (
        ufl.inner(grad_3D(v), P) * dx - ufl.inner(v, T_right) * ds(3) - ufl.inner(v, T_top) * ds(4)
        # ufl.inner(grad_3D(v), P) * dx - ufl.inner(v, T_top) * ds(4)

    )

    # As the varitional form is non-linear and written on residual form,
    # we use the non-linear problem class from DOLFINx to set up required structures to use a Newton solver.

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
    problem = NonlinearProblem(
        residual,
        u,
        bcs=bcs,
        petsc_options=petsc_options,
        petsc_options_prefix="hyperelasticity",
        
    )
    # load_per_step = (load_end - load_start)/load_steps
    load_steps = np.linspace(load_start, load_end, load_steps)
    u_steps = {}

    mesh_pos = domain.geometry.x
    fdim = domain.topology.dim - 1 
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(-1, 3)
    num_nodes = domain.geometry.x.shape[0]
    node_type = np.zeros(num_nodes, dtype=int)
    node_type_onehot = np.zeros((num_nodes, 5), dtype=int)

    # 1. Identify node indices for each boundary
    left_indices   = facets_to_nodes(domain, left_facets)
    bottom_indices = facets_to_nodes(domain, bottom_facets)
    right_indices  = facets_to_nodes(domain, right_facets)
    top_indices    = facets_to_nodes(domain, top_facets)
    node_type_onehot[left_indices, 1] = 1
    node_type_onehot[bottom_indices, 2] = 1
    node_type_onehot[right_indices, 3] = 1
    node_type_onehot[top_indices, 4] = 1

    # 3. Label Internal Nodes (nodes that have no boundary bits set)
    internal_mask = np.sum(node_type_onehot[:, 1:], axis=1) == 0
    node_type_onehot[internal_mask, 0] = 1

    right_x = []
    top_y = []
    left_x = []
    bottom_y = []
    reactions = []
    loads = []
    asymetric_factor = 0.9
    for i, n in enumerate(load_steps):
        T_right.value[0] = n * asymetric_factor
        T_top.value[1] = n
        # top_dis.value = n
        # right_dis.value = (n/2.0)
        # bcs_t = [
        #     fem.dirichletbc(zero, left_dofs, V.sub(0)), 
        #     fem.dirichletbc(zero, bottom_dofs, V.sub(1)),
        #     fem.dirichletbc(top_dis, top_dofs, V.sub(1)), 
        #     fem.dirichletbc(right_dis, right_dofs, V.sub(0)),]
        # problem.bcs = bcs_t  

        problem.solve()

        u_array = u.x.array[:].reshape(mesh_pos.shape[0], -1)
        u_steps[i] = u_array.copy()

        res_form = fem.form(residual)
        R_vec = fem.petsc.assemble_vector(res_form)
        R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # 2. Extract reactions for specific boundaries
        # We use the Dofs we already located (left_dofs, right_dofs, etc.)
        # Note: R_vec.array contains [ux0, uy0, ux1, uy1, ...]
        all_reactions = R_vec.array
        
        # Total Reaction at Right Edge (Traction equivalent)
        # Right edge has index 3. We sum the X-components (sub(0))
        force_right_x = np.sum(all_reactions[right_dofs])
        right_x.append(force_right_x)
        
        # Total Reaction at Top Edge
        # Top edge has index 4. We sum the Y-components (sub(1))
        force_top_y = np.sum(all_reactions[top_dofs])
        top_y.append(force_top_y)

        force_left_x = np.sum(all_reactions[left_dofs])
        left_x.append(force_left_x)
        
        # Total Reaction at Top Edge
        # Top edge has index 4. We sum the Y-components (sub(1))
        force_bottom_y = np.sum(all_reactions[bottom_dofs])
        bottom_y.append(force_bottom_y)
        reactions.append(np.array([force_left_x, force_bottom_y, force_right_x, force_top_y]))
        loads.append(np.array([asymetric_factor *n, n]))


    return dict(mesh_pos = mesh_pos, cells = cells, u = u_steps, node_type = node_type_onehot, load_steps = load_steps, 
                rigth_xs = right_x,
                 left_xs = left_x,
                  top_ys = top_y,
                   bottom_ys = bottom_y,
                   reactions = reactions,
                   loads = loads)
def precompute_vfm_weight(u, coords, cells, node_type, load) :
    u_cells = u[cells]
    coord_cells = coords[cells]
    F, dNdx = deformation_gradient_element(coord_cells, u_cells)
    dA = jnp.linalg.det(transformation_jacobian(coord_cells)) / 2  # (C,)
    t3, t4 = load
    types = node_type[cells]     # (C, 3)

    # element nodal traction forces (C, 3, 2)
    f_neu_cells = jax.vmap(
        _neumann_cell_force, in_axes=(0, 0, None, None)
    )(coord_cells, types, t3, t4)

    # assemble global neumann nodal forces
    f_neu_nodes = jnp.zeros((coords.shape[0], 2)).at[cells].add(f_neu_cells)

    return PrecomputedVFMData(f_neu_nodes, dNdx, dA, F)

if __name__ == "__main__" :
    # 1, 10 for isihara
    # 1, 2.2 for neohookean
    # 1, 3.0 for gent thomas
    load_start = 1.0
    load_end = 10.0
    load_steps = 10
    mat_model = "isihara"
    load_noise = 0.02
    disp_noise = 0.0001 
    data = FEM_solve(load_start, load_end, load_steps, mat_model)

    for step in data["u"].keys() :
        data_ = dict(mesh_pos = data["mesh_pos"], cells = data["cells"], u = data["u"][step], node_type = data["node_type"], load_parameter = data["load_steps"][step], reaction = data["reactions"][step], load = data["loads"][step])
        np.savez_compressed(f"/home/mmdiscovery/shared/dataset/{mat_model}/disp_{step}.npz", **data_)


    material_model = "isihara"
    dataset_name = "isihara"
    dataset = TractionDataset(f"dataset/{dataset_name}")
    F_all = []
    reactions = []
    u_all = [] 
    loads = []
    range_ = range(0, len(dataset), 2)
    range_ = [0, len(dataset)//2, len(dataset) - 1]
    for loadstep in range_ :
        # if loadstep != len(dataset) - 1 :
        #     continue
        data = dataset[loadstep]
        coords = data["mesh_pos"][:,:2]
        cells = data["cells"]
        # u = data["u"]
        u_percent_noise = disp_noise
        node_type = data["node_type"]
        ux = data["u"][:, 0]
        ux[(data["node_type"][:, 1] != 1)] += np.random.normal(0, u_percent_noise * 1, ux.shape)[(data["node_type"][:, 1] != 1)]
        uy = data["u"][:, 1]
        uy[(data["node_type"][:, 2] != 1)] += np.random.normal(0, u_percent_noise * 1, uy.shape)[(data["node_type"][:, 2] != 1)]

        # Combine components into the full displacement vector u
        u = np.column_stack((ux, uy))

        reaction = data["reaction"]
        coord_cells = coords[cells]
        u_cells = u[cells]
        load = data["load"]
        F, dNdx = deformation_gradient_element(coord_cells, u_cells)
        loads.append(load)
        u_all.append(u)
        F_all.append(F)
        reactions.append(reaction)

    F_array = jnp.array(F_all)
    reactions_array = jnp.array(reactions)
    u_array = jnp.array(u_all)
    loads = jnp.array(loads)
    load_noise_percentage = load_noise
    load_noise = load_noise_percentage * (jnp.max(loads) + jnp.min(loads))/2
    loads = loads + jax.random.normal(0, load_noise, loads.shape)
    I_all_dev, j = jax.vmap(transform_input_features)(I_obs_all)

    preprocessed_data = jax.vmap(precompute_vfm_weight)(u_array, coords, cells, node_type, loads)