import meshio
import numpy as np

def convert_msh_to_npz(msh_path, npz_path):
    """
    Reads a Gmsh (.msh) mesh file using meshio, extracts node coordinates
    and elemental cell connectivity, and saves them into a compressed numpy (.npz) archive.
    """
    mesh = meshio.read(msh_path)
    node_coords = mesh.points
    
    # Prioritize 2D domain cells like triangles (TRI3) or quads, then 3D elements
    cells = None
    for cell_type in ["triangle", "quad", "tetra", "hexahedron"]:
        if cell_type in mesh.cells_dict:
            cells = mesh.cells_dict[cell_type]
            break
            
    if cells is None and len(mesh.cells) > 0:
        # Fallback to the primary cell data block
        cells = mesh.cells[-1].data
        
    np.savez_compressed(npz_path, node_coords=node_coords, cells=cells)
    return npz_path
