

import jax
from dolfinx.io.gmsh import MeshData, read_from_msh
from mpi4py import MPI
import jax.numpy as jnp
import numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import rectangle_mesh, get_meshio_cell_type, Mesh
from core.material_models import get_material
from core.model import TensorBasisGPModel, SVTBGPModel
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from jax_fem.solver import solver, dynamic_relax_solve
def generate_deformation_plots(
    material_model,  
    u_gt, 
    u_pred, 
    node_coords,
    cells, 
    save_result_dir=""
):
    """
    Generates comparison plots for displacement field (Ground Truth vs. Prediction)
    and the squared error field for each load step.
    """
    
    # 1. Define the output directory
    output_dir = os.path.join(save_result_dir, material_model)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get static node information
    node_coords = node_coords
    cells = cells
    
    # Create a Triangulation object once (assuming fixed topology)
    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1], cells)

    # Calculate global color limits across ALL steps and ALL subplots 
    # for consistent visualization (Recommended practice for comparison)
    
    # Max value of the total displacement magnitude for color limit
    max_disp_mag = 0
    # Max value of the squared error for color limit
    max_error_mag = 0
        
        # Calculate displacement and error magnitudes
    disp_mag_gt = np.sqrt(u_gt[:, 0]**2 + u_gt[:, 1]**2)
    disp_mag_pred = np.sqrt(u_pred[:, 0]**2 + u_pred[:, 1]**2)
    
    # Error field: (u_x_pred - u_x_gt)**2 + (u_y_pred - u_y_gt)**2
    error_sq = np.sqrt((u_pred[:, 0] - u_gt[:, 0])**2 + (u_pred[:, 1] - u_gt[:, 1])**2)
    
    max_disp_mag = max(max_disp_mag, np.max(disp_mag_gt), np.max(disp_mag_pred))
    max_error_mag = max(max_error_mag, np.max(error_sq))

    # Define color limits
    disp_vmax = max_disp_mag
    error_vmax = max_error_mag

    # 2. Iterate and Plot for Each Load Step

    # Calculate deformed coordinates (x + u_x, y + u_y)
    x_gt_def = node_coords[:, 0] + u_gt[:, 0]
    y_gt_def = node_coords[:, 1] + u_gt[:, 1]
    
    x_pred_def = node_coords[:, 0] + u_pred[:, 0]
    y_pred_def = node_coords[:, 1] + u_pred[:, 1]

    # Recalculate magnitudes and error for plotting in this step
    disp_mag_gt = np.sqrt(u_gt[:, 0]**2 + u_gt[:, 1]**2)
    disp_mag_pred = np.sqrt(u_pred[:, 0]**2 + u_pred[:, 1]**2)
    error_sq = np.sqrt((u_pred[:, 0] - u_gt[:, 0])**2 + (u_pred[:, 1] - u_gt[:, 1])**2)
    
    # Create a new Triangulation for the deformed shapes
    triang_gt_def = tri.Triangulation(x_gt_def, y_gt_def, cells)
    triang_pred_def = tri.Triangulation(x_pred_def, y_pred_def, cells)

    # Initialize Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Material Model: {material_model}", fontsize=16)

    # --- Subplot 1: Deformed Domain (Ground Truth) ---
    ax = axes[0]
    tpc_gt = ax.tripcolor(
        triang_gt_def, 
        disp_mag_gt, 
        shading='flat', 
        cmap='viridis', 
        vmin=0, 
        vmax=disp_vmax
    )
    ax.set_title("Ground Truth $||\mathbf{u_{gt}}||_2$")
    ax.set_xlabel("X (-)")
    ax.set_xlabel("Y (-)")
    ax.set_aspect('equal')
    fig.colorbar(tpc_gt, ax=ax, orientation='vertical', label='$||\mathbf{u_{gt}}||_2$')
    ax.triplot(triang_gt_def, color='k', linewidth=0.5) # Add wire frame

    # --- Subplot 2: Deformed Domain (Prediction) ---
    ax = axes[1]
    tpc_pred = ax.tripcolor(
        triang_pred_def, 
        disp_mag_pred, 
        shading='flat', 
        cmap='viridis', 
        vmin=0, 
        vmax=disp_vmax
    )
    ax.set_title("SVTBGP Prediction $||\mathbf{u_{pred}}||_2$")
    ax.set_xlabel("X (-)")
    ax.set_xlabel("Y (-)")
    ax.set_aspect('equal')
    fig.colorbar(tpc_pred, ax=ax, orientation='vertical', label='$||\mathbf{u_{pred}}||_2$')
    ax.triplot(triang_pred_def, color='k', linewidth=0.5) # Add wire frame
    
    # --- Subplot 3: Error Field ---
    ax = axes[2]
    tpc_err = ax.tripcolor(
        triang, # Use original mesh topology, colored by error at nodes
        error_sq, 
        shading='flat', 
        cmap='plasma', # Use a different colormap for error
        vmin=0, 
        vmax=error_vmax 
    )
    ax.set_title("Error Field $||\mathbf{u_{pred}} - \mathbf{u_{gt}}||_2$")
    ax.set_xlabel("X (-)")
    ax.set_xlabel("Y (-)")
    ax.set_aspect('equal')
    fig.colorbar(tpc_err, ax=ax, orientation='vertical', label='$||\mathbf{u_{pred}} - \mathbf{u_{gt}}||_2$')
    ax.triplot(triang, color='k', linewidth=0.5) # Wire frame of original mesh

    # Save Figure
    filename = f"{material_model}_deformed_domain.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)

    print(f"Successfully generated plots for {material_model} in {output_dir}")

def plot_R2_PK(pk1_pred, pk1_gt, filename="pk_plot.png"):
    """
    pk1_pred, pk1_gt: arrays of shape (N, 2, 2) or (N, 4)
    Saves a PNG plot comparing prediction vs ground truth.
    Adds a 45° reference line (y = x).
    """
    # Flatten to (N, 4)
    pred = pk1_pred.reshape(len(pk1_pred), -1)
    gt   = pk1_gt.reshape(len(pk1_gt), -1)

    # R² function
    def r2(y_pred, y_true):
        ss_res = jnp.sum((y_true - y_pred)**2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true))**2)
        return 1 - ss_res / ss_tot

    r2_vals = [float(r2(pred[:, i], gt[:, i])) for i in range(4)]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.reshape(-1)

    component_labels = ["P11", "P12", "P21", "P22"]

    for i in range(4):
        ax = axes[i]

        # Scatter plot
        ax.scatter(gt[:, i], pred[:, i])

        # 45° reference line
        min_val = min(gt[:, i].min(), pred[:, i].min())
        max_val = max(gt[:, i].max(), pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], color = "k", linestyle = "-")

        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{component_labels[i]}  (R² = {r2_vals[i]:.3f})")

    fig.suptitle("First Piola–Kirchhoff Stress: Prediction vs Ground Truth")
    fig.tight_layout()

    # Save as PNG
    fig.savefig(filename, dpi=300)
    plt.close(fig)

    return r2_vals

def F2x2_to_F3x3(f):
    # f: (..., 2, 2)
    batch_shape = f.shape[:-2]

    # Start with identity matrices of the right batch shape
    out = jnp.broadcast_to(jnp.eye(3), batch_shape + (3, 3)).copy()

    # Insert the 2×2 blocks
    out = out.at[..., :2, :2].set(f)
    return out

class HyperElasticity(Problem):
    def __init__(self, model, **kwargs) :
        super().__init__(**kwargs)
        self.model = model # should be function outputing piola stress [2x2 matrix]
    def get_tensor_map(self):

        def first_PK_stress(u_grad):
            I = jnp.eye(self.dim)
            F = u_grad + I
            F = jnp.array([[F[0, 0], F[0, 1], 0.], 
                          [F[1, 0], F[1, 1], 0.],
                          [0., 0., 1.]])
            P = self.model(F)[:2, :2]
            return P
        return first_PK_stress

    def get_u_grad(self, sol):
        u_grads = self.fes[0].sol_to_grad(sol)
        return u_grads

class HyperElasticSolver :
    def __init__(self, node_coords, cells, boundary_conditions, material_model) :
        # here i should define the geometry, loadsteps, boundary conditions
        # assume the boundary of the geometry is the same for all geometry -> Square (1x1)
        self.node_coords = node_coords
        self.cells = cells
        self.boundary_conditions = boundary_conditions
        self.material_model = material_model
    
    def solve(self, disps) :
        """ Solve the FEM using the specified model"""
        # current_node_coords = np.copy(self.get_node_coords()[:, :2])
        mesh = Mesh(self.node_coords, self.cells)
        petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_monitor": True,
        "snes_damping": 0.8,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-5,
        "snes_stol": 1e-8,
        "ksp_type": "gmres",
        "pc_type": "hypre",   # or "gamg" if hypre not available
        "pc_factor_mat_solver_type": "mumps",
        "snes_npc_side": "right",
        "snes_npc_snes_type": "ngmres",}
        dirichlet_bc_info = self.boundary_conditions
        problem = HyperElasticity(mesh = mesh,
                            vec=self.node_coords.shape[1],
                            dim=self.node_coords.shape[1],
                            ele_type="TRI3",
                            dirichlet_bc_info=dirichlet_bc_info,
                            model = self.material_model)
        u_steps = []
        u_grads = []
        sol = solver(problem, solver_options = {'petsc_solver': petsc_options})[0]
        # sol = dynamic_relax_solve(problem, tol = 1e-6)[0]
        u_steps.append(sol.copy())
        # u_grad = problem.fes[0].sol_to_grad(sol.copy())
        # u_grads.append(u_grad)
        return u_steps, u_grads
def get_dirichlet_top(disp):
    def val_fn(point):
        return disp
    return val_fn
if __name__ == "__main__":# block to correctly implement incremental loading over the specified loadsteps.The two main issues fixed are:Lambda Closure & Constant Load: The lambda functions now correctly capture the incremental displacement (Total $\Delta u_y$) for each step using default arguments.Boundary Definition: The top boundary location is fixed at $y=1.0$ (initial position), while the applied displacement increases. The original attempt to change the boundary location in the top function was incorrect for applying load.Here is the adjusted code block:Pythonif __name__ == "__main__" :

    # --- Boundary Condition Definitions (Keep these) ---
    def left(point):
        return jnp.isclose(point[0], 0., atol=1e-6)
    def bottom(point):
        return jnp.isclose(point[1], 0., atol=1e-6)
    def right(point):
        return jnp.isclose(point[0], 1., atol=1e-6)
    def top(point):
        return jnp.isclose(point[1], 1.0, atol=1e-6)


    def load_displacement_yy(point):
        return 0.5
    def load_displacement_xy(point):
        return 0.0
    zero_dbc = lambda point : 0
    disps = np.linspace(0.5, 0.5, 5)
    dirichlet_bc_info = (
            # 1. Boundary Location Functions: [bottom, bottom, top, top]
            [bottom]*2 + [top] * 2, 
            
            # 2. Component Indices: [u_x, u_y, u_x, u_y]
            [0, 1] + [0, 1],
            
            # 3. Applied Values: [0, 0, 0, TOTAL_UY]
            [zero_dbc] * 2 + [load_displacement_xy, get_dirichlet_top(disps[0])]
        )
    boundary_conditions = dirichlet_bc_info

    # --- Solver Execution (Keep this) ---
    msh_data = read_from_msh("mesh_with_hole.msh", MPI.COMM_WORLD, 0, 2) 
    domain = msh_data.mesh

    node_coords = domain.geometry.x[:, :2]  # numpy array
    num_nodes, gdim = node_coords.shape
    # 3. Extract cell connectivity (triangles)
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    cells = domain.geometry.dofmap

    material_model_name = "isihara"
    # material_model = get_material(material_model_name, c1=0.5, c2=1.5)
    material_model = get_material(material_model_name)
    # GT Solve
    hes = HyperElasticSolver(node_coords, cells, boundary_conditions, material_model.P)
    u_list, ugrad_gt = hes.solve(disps)
    # GP Prediction Solve
    # model_date_time = "20251123T145059"
    # model_path = f"selected_model/neo-hookean/{model_date_time}" 
    model_date_time = "20251126T215746"
    model_path = f"saved_model/{model_date_time}"
    gp_model = SVTBGPModel()
    gp_model.load_model(model_path)

    def P_func(model, f):
        # Case 1: f is a single 2×2 matrix
        if f.ndim == 2:
            f3x3 = F2x2_to_F3x3(f)[None, ...]     # (1,3,3)
            P3x3 = model.predict_piola_stress(f3x3)[0]  # (1,3,3)
            return P3x3[0, :2, :2]                # (2,2)

        # Case 2: f is (N,1,2,2)
        elif f.ndim == 3 and f.shape[1:] == (2,2):
            N = f.shape[0]
            f2x2 = f[:, :, :]                  # (N,2,2)
            f3x3 = F2x2_to_F3x3(f2x2)       # (N,3,3)
            P3x3 = model.predict_piola_stress(f3x3)[0]  # (N,3,3)
            return P3x3[:, None, :2, :2]          # (N,1,2,2)

        else:
            raise ValueError(f"Unexpected f shape: {f.shape}")
    hes_gp = HyperElasticSolver(node_coords, cells, boundary_conditions, lambda f: gp_model.predict_piola_stress(f[None, ...])[0][0])
    u_pred_list, ugrad_pred = hes_gp.solve(disps)
    print("Complete FEM Solving")

    save_result_dir = f"results/model_dt_{model_date_time}/"
    output_dir = os.path.join(save_result_dir, material_model_name)
    os.makedirs(output_dir, exist_ok=True)
    np.save(save_result_dir + "/u_pred.npy", u_pred_list)
    np.save(save_result_dir + "/u_gt.npy", u_list)
    generate_deformation_plots(material_model_name, u_list[0], u_pred_list[0], node_coords, cells, save_result_dir)
    # plot_R2_PK(pk1_pred[:, :2, :2], pk1_gt[:, :2, :2], save_result_dir)