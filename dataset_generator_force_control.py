
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import numpy as np
from dolfinx.io.gmsh import read_from_msh
import ufl
import gmsh
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import jax
import jax.numpy as jnp
import argparse
from core.utils import deformation_gradient_element, transformation_jacobian
from core.loss_function import neumann_cell_force
from pathlib import Path

def facets_to_nodes(domain, facet_ids):
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, 0)
    facet_to_vertices = domain.topology.connectivity(fdim, 0)
    all_nodes = []
    for f in facet_ids:
        all_nodes.extend(facet_to_vertices.links(f))
    return np.unique(np.array(all_nodes, dtype=np.int32))

def FEM_solve(material_model_name, loads) :

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
    gmsh.write("mesh/training_mesh.msh")

    # Launch GUI to verif

    gmsh.finalize()

    # 8. Read into DOLFINx
    mesh_ = read_from_msh("mesh/training_mesh.msh", MPI.COMM_WORLD, 0, 2)
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

    # To apply the boundary condition, we identity the dofs located on the facets marked by the `MeshTag`.
    zero = fem.Constant(domain, default_scalar_type(0.0))

    # Locate DOFs (note the use of collapsed spaces)
    left_dofs = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(1))
    bottom_dofs = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(2))
    right_dofs = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, facet_tag.find(3))
    top_dofs = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, facet_tag.find(4))


    # Next, we define the body force on the reference configuration (`B`), and nominal (first Piola-Kirchhoff) traction (`T`).

    T_right = fem.Constant(domain, default_scalar_type((0.0, 0.0)))
    T_top = fem.Constant(domain, default_scalar_type((0.0, 0.0)))

    bcs = [
        fem.dirichletbc(zero, left_dofs, V.sub(0)), 
        fem.dirichletbc(zero, bottom_dofs, V.sub(1)),
        ]
    
    v = ufl.TestFunction(V)
    u = fem.Function(V)

    # Define kinematic quantities used in the problem

    # +
    # Identity tensor
    I = ufl.variable(ufl.Identity(3))
    def grad_3D(u):
        return ufl.as_matrix([[u[0].dx(0), u[0].dx(1), 0], 
                            [u[1].dx(0), u[1].dx(1), 0], 
                            [0, 0, 0]])
    F = ufl.variable(I + grad_3D(u))

    # Right Cauchy-Green tensor
    C = ufl.variable(F.T * F)

    # Invariants of deformation tensors
    Ic = ufl.variable(ufl.tr(C))
    I2 = ufl.variable(0.5 * (ufl.tr(C)**2 - ufl.tr(C * C)))
    J = ufl.variable(ufl.det(F))

    # Elasticity parameters

    # Stored strain energy density (compressible neo-Hookean model)
    if material_model_name == "isihara" :
        psi =  0.5 * (J**(-2/3) * Ic - 3) + (J**(-2/3) * Ic - 3)**2 +  (J**(-4/3) * I2 - 3) + 1.5 * (J - 1)**2 #isihara
    elif material_model_name == "neohookean" :
        psi = 0.5 * (J**(-2/3) * Ic - 3) + 1.5 * (J - 1)**2 # neohookean
    elif material_model_name == "gentthomas" :
        psi = 0.5 * (J**(-2/3) *Ic - 3) + ufl.ln(J**(-4/3) * I2/3) + 1.5 * (J - 1)**2
    else :
        ValueError("material model not supported")
    # Hyper-elasticity

    P = ufl.diff(psi, F)

    metadata = {"quadrature_degree": 4}
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)

    # Define the residual of the equation (we want to find u such that residual(u) = 0)

    residual = (
        ufl.inner(grad_3D(v), P) * dx - ufl.inner(v, T_right) * ds(3) - ufl.inner(v, T_top) * ds(4)
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

    load_steps = loads
    u_steps = {}

    mesh_pos = domain.geometry.x
    fdim = domain.topology.dim - 1 
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(-1, 3)
    num_nodes = domain.geometry.x.shape[0]
    
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
    for i, n in enumerate(load_steps):
        T_right.value[0] = n[0]
        T_top.value[1] = n[1]

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


    return dict(mesh_pos = mesh_pos, cells = cells, u = u_steps, node_type = node_type_onehot, 
                rigth_xs = right_x,
                left_xs = left_x,
                top_ys = top_y,
                bottom_ys = bottom_y,
                reactions = reactions,
                loads = load_steps)


def plot_dataset_viz(data, material_model_name, disp_noise_level, load_noise_level, save_path) :
    # --- 1. Setup Dummy Data (Simulating FEM Output) ---
    # Create a simple 2x2 rectangular mesh with 4 nodes and 2 triangular elements
    # Node coordinates (Undeformed mesh_pos)
    mesh_pos = data["mesh_pos"]

    # Element connectivity (cells: indices of nodes forming each triangle)
    cells = data["cells"]

    # Displacement components (ux and uy) at each node
    # This simulates a simple shear/tensile deformation
    percent_noise = 0.000
    ux = data["u"][len(data["u"].keys()) - 1][:, 0]
    ux[(data["node_type"][:, 1] != 1)] += np.random.normal(0, percent_noise, ux.shape)[(data["node_type"][:, 1] != 1)]
    uy = data["u"][len(data["u"].keys()) - 1][:, 1]
    uy[(data["node_type"][:, 2] != 1)] += np.random.normal(0, percent_noise, uy.shape)[(data["node_type"][:, 2] != 1)]

    # Combine components into the full displacement vector u
    u = np.column_stack((ux, uy))
    # u[data["node_type"] == 0] = u[data["node_type"] == 0] + np.random.normal(0, 0.0001, u.shape)[data["node_type"] == 0]
    # Calculate the deformed coordinates (world_pos)
    world_pos = mesh_pos[:, :2] + u

    # --- 2. Initialize Triangulation Objects ---
    # We need the x and y coordinates from the undeformed mesh
    x = mesh_pos[:, 0]
    y = mesh_pos[:, 1]

    # Create the Matplotlib Triangulation object
    # This object stores the connectivity (cells) and coordinates (x, y)
    triangulation = tri.Triangulation(x, y, cells)

    # --- 3. Plotting ---

    # Set up the figure with 1 row and 3 columns for the plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Finite Element Visualization (Undeformed vs. Deformed)', fontsize=16)

    # --- Subplot 1: Plotting UX (Horizontal Displacement) ---
    ax1 = axes[0]
    # tripcolor uses the triangulation to color the triangles based on the nodal value
    # `facecolors` uses the average of the three nodal values per triangle for coloring
    tpc1 = ax1.tripcolor(triangulation, ux, cmap='viridis', edgecolors='k', linewidth=0.5)
    fig.colorbar(tpc1, ax=ax1, label='$u_x$ Displacement')
    # ax1.scatter(mesh_pos[data["node_type"] == 5, 0], mesh_pos[node_type == 5, 1])
    ax1.set_title('Color Plot: $u_x$ (Horizontal Displacement)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_aspect('equal')

    # --- Subplot 2: Plotting UY (Vertical Displacement) ---
    ax2 = axes[1]
    tpc2 = ax2.tripcolor(triangulation, uy, cmap='magma', edgecolors='k', linewidth=0.01)
    fig.colorbar(tpc2, ax=ax2, label='$u_y$ Displacement')

    ax2.set_title('Color Plot: $u_y$ (Vertical Displacement)')
    ax2.set_xlabel('X Position')
    ax2.set_aspect('equal')

    # --- Subplot 3: Plotting Deformed Domain ---
    ax3 = axes[2]

    # Plot the outline of the UNDEFORMED mesh for reference (dashed gray)
    ax3.triplot(triangulation, 'r-', alpha=0.5, linewidth=0.5, label='Undeformed Mesh')

    # Plot the DEFORMED mesh. We must manually create a new triangulation
    # object using the deformed coordinates (world_pos) but the SAME connectivity (cells).
    x_def = world_pos[:, 0]
    y_def = world_pos[:, 1]
    tri_def = tri.Triangulation(x_def, y_def, cells)

    # Plot the deformed mesh (solid blue lines)
    ax3.triplot(tri_def, 'b-', linewidth=0.5, label='Deformed Mesh')

    ax3.set_title('Deformed Domain')
    ax3.set_xlabel('X Position')
    ax3.legend()
    ax3.set_aspect('equal')

    # Adjust layout to prevent overlaps
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # create save_path dir if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
   
    plt.savefig(save_path / f"{material_model_name}_{disp_noise_level}_{load_noise_level}.png", dpi=300, bbox_inches='tight')
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="isihara")
    parser.add_argument('--disp_noise', type=float, default=0.0)
    parser.add_argument('--load_noise', type=float, default=0.01)
    parser.add_argument('--target_top', type=float, default=8.0)
    parser.add_argument('--asym', type=float, default=0.975)
    parser.add_argument('--n_steps', type=int, default=10)
    args = parser.parse_args()

    # Assign from args
    material_model_name = args.model
    disp_noise_level = args.disp_noise
    load_noise_level = args.load_noise
    asymetric_factor = args.asym
    n_loadsteps = args.n_steps
    target_load_top_true = args.target_top


    target_load_right_true = target_load_top_true * asymetric_factor
    target_load_true = np.array([target_load_right_true, target_load_top_true])
    load_noise_std = load_noise_level * target_load_true 
    target_load_noisy = np.random.normal(target_load_true, load_noise_std)

    noisy_top_load = np.linspace(0, target_load_noisy[1], n_loadsteps).reshape(-1,1)
    noisy_right_load = np.linspace(0, target_load_noisy[0], n_loadsteps).reshape(-1,1)
    noisy_load = np.concat([noisy_right_load, noisy_top_load], axis = 1)


    top_load = np.linspace(0, target_load_top_true, n_loadsteps).reshape(-1,1)
    right_load = np.linspace(0, target_load_right_true, n_loadsteps).reshape(-1,1)
    load_true = np.concat([right_load, top_load], axis = 1)

    data = FEM_solve(material_model_name, load_true)

    save_raw_dataset_dir = f"raw_dataset/{material_model_name}_{disp_noise_level}_{load_noise_level}_{target_load_top_true}_{asymetric_factor}"
    if not os.path.exists(save_raw_dataset_dir):
        os.makedirs(save_raw_dataset_dir)
    for step in data["u"].keys() :
        # data_ = dict(mesh_pos = data["mesh_pos"], cells = data["cells"], u = data["u"][step], node_type = data["node_type"], reaction = data["reactions"][step], load = data["loads"][step], load_noise_std = load_noise_std)
        data_ = dict(mesh_pos = data["mesh_pos"], cells = data["cells"], u = data["u"][step], node_type = data["node_type"], reaction = data["reactions"][step], load = noisy_load[step], load_noise_std = load_noise_std)
        
        np.savez_compressed(f"{save_raw_dataset_dir}/disp_{step}.npz", **data_)

    random_key = jax.random.PRNGKey(0)

    data_dir = Path(f"raw_dataset/{material_model_name}_{disp_noise_level}_{load_noise_level}_{target_load_top_true}_{asymetric_factor}")

    # find the first .npz file in that directory
    npz_files = list(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in {data_dir}")
    
    data = [dict(jnp.load(p)) for p in npz_files]
    F_all = []
    u_all = []
    load_all = []
    f_neu_all = []

    load_stat = jnp.array([d["load"] for d in data])
    mean_load = jnp.mean(load_stat)
    
    for d in data :
        random_key, subkey_disp, subkey_load = jax.random.split(random_key, 3)

        u = d["u"]
        # disp noise needed to be added here, so we can propagate noise from u to F
        u_noise = jax.random.normal(subkey_disp, u.shape) * disp_noise_level
        free_nodes = (d["node_type"][:, 1] != 1) & (d["node_type"][:, 2] != 1)
        u_noise = u_noise.at[free_nodes].set(0.0)
        u += u_noise

        mesh_pos = d["mesh_pos"][:, :2]
        cells = d["cells"]
        node_type = d["node_type"]
        load = d["load"]
        # check = load_noise * mean_load
        # load_noise_ = jax.random.normal(subkey_load, load.shape) * load_noise * mean_load
        # load += load_noise_

        m_cells = mesh_pos[cells]
        u_cells = u[cells]
        node_type_cells = node_type[cells]

        F, dNdX = deformation_gradient_element(m_cells, u_cells)
        dA = jnp.linalg.det(transformation_jacobian(m_cells)) / 2 
        f_neu_cells = jax.vmap(neumann_cell_force, in_axes=(0, 0, None, None))(m_cells, node_type_cells, load[0], load[1])
        f_neu = jnp.zeros((mesh_pos.shape[0], 2)).at[cells].add(f_neu_cells)

        F_all.append(F)
        u_all.append(u)
        load_all.append(load)
        f_neu_all.append(f_neu)

    u_array = jnp.stack(u_all)  
    F_array = jnp.stack(F_all)
    load_array = jnp.stack(load_all)
    f_neu_array = jnp.stack(f_neu_all)

    # save true psi/piola function to facilitate the plot

    # save all as npz in /precomputed_vfm/{material_model}_{disp_noise}_{load_noise}/
    precomputed_vfm = dict(mesh_pos = mesh_pos, cells = cells, node_type = d["node_type"], load = load_array, u = u_array, F = F_array, dNdX = dNdX, dA = dA, f_neu = f_neu_array, load_noise_std = load_noise_std)
    np.savez_compressed(f"precomputed_vfm/{material_model_name}_{disp_noise_level}_{load_noise_level}_{target_load_top_true}_{asymetric_factor}.npz", **precomputed_vfm)