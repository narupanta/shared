# take data from dataset/raw and precompute all inputs neccessary for training and save as precomputed_vfm/{material_model_name} + {disp_noise} + {load_noise}/
import jax.numpy as jnp
import jax
from pathlib import Path
from core.utils import *
from core.loss_function import *
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == "__main__" :
    random_key = jax.random.PRNGKey(0)
    disp_noise = 0.000
    load_noise = 0.005
    material_model_name = "isihara"  # set this to the desired model name

    data_dir = Path("dataset") / material_model_name

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
        u_noise = jax.random.normal(subkey_disp, u.shape) * disp_noise
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
    precomputed_vfm = dict(mesh_pos = mesh_pos, cells = cells, node_type = d["node_type"], load = load_array, u = u_array, F = F_array, dNdX = dNdX, dA = dA, f_neu = f_neu_array)
    np.savez_compressed(f"precomputed_vfm/{material_model_name}_{disp_noise}_{load_noise}.npz", **precomputed_vfm)