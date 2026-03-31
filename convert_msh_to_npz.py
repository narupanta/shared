import meshio
import numpy as np
from dolfinx.io.gmsh import read_from_msh
from mpi4py import MPI

def convert_msh_to_npz(msh_filename, npz_filename):
    # Read the mesh file
    # mesh = meshio.read(msh_filename)
    mesh_ = read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, 2)
    domain = mesh_.mesh
    mesh_pos = domain.geometry.x
    cells = domain.topology.connectivity(domain.topology.dim, 0).array.reshape(-1, 3)

    # Extract cell connectivity
    # Meshio stores cells by type (e.g., "triangle", "tetra")
    data_dict = {}
    data_dict[f"node_coords"] = mesh_pos
    data_dict[f"cells"] = cells

    # # Extract point/cell data if they exist (e.g., physical groups)
    # for key, val in mesh.point_data.items():
    #     data_dict[f"point_data_{key}"] = val
    
    # for key, val in mesh.cell_data.items():
    #     data_dict[f"cell_data_{key}"] = val

    # Save as compressed NumPy archive
    np.savez_compressed(npz_filename, **data_dict)
    print(f"Successfully converted {msh_filename} to {npz_filename}")

# Usage
convert_msh_to_npz("mesh.msh", "mesh/mesh.npz")