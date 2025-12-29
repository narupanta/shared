{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9bd915f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:2025-12-25 10:53:34,055:jax._src.xla_bridge:854: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.\n"
     ]
    }
   ],
   "source": [
    "import jax \n",
    "import gpjax as gpx\n",
    "import jax.numpy as jnp\n",
    "from jax import config\n",
    "import jax.numpy as jnp\n",
    "import jax.random as jr\n",
    "from jaxtyping import install_import_hook\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "import optax\n",
    "from core.model import SparseHyperelasticityGP\n",
    "from core.material_models import get_material\n",
    "import jax\n",
    "import jax.numpy as jnp\n",
    "from core.utils import *\n",
    "import datetime\n",
    "import os\n",
    "\n",
    "from core.datasetclass import TractionDataset\n",
    "from core.loss_function import physical_loss, elbo_loss\n",
    "\n",
    "base_save_path = \"saved_model\"  # change as needed\n",
    "os.makedirs(base_save_path, exist_ok=True)\n",
    "\n",
    "# Subfolder with datetime\n",
    "timestamp = datetime.datetime.now().strftime(\"%Y%m%dT%H%M%S\")\n",
    "save_path = os.path.join(base_save_path, timestamp)\n",
    "os.makedirs(save_path, exist_ok=True)\n",
    "dataset = TractionDataset(\"dataset\",\"NH\")\n",
    "data = dataset[-1]\n",
    "coords = data[\"mesh_pos\"][:,:2]\n",
    "cells = data[\"cells\"]\n",
    "u = data[\"u\"]\n",
    "node_type = data[\"node_type\"]\n",
    "load_parameter = data[\"load_parameter\"]\n",
    "\n",
    "coord_cells = coords[cells]\n",
    "u_cells = u[cells]\n",
    "\n",
    "F, dNdx = deformation_gradient_element(coord_cells, u_cells)\n",
    "\n",
    "I_obs, _ = jax.vmap(invariants_and_derivatives)(F)\n",
    "\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "import os\n",
    "\n",
    "# Check for files in current directory\n",
    "files = os.listdir('.')\n",
    "print(f\"Files in directory: {files}\")\n",
    "\n",
    "# Generate synthetic data for a 3D scatter plot\n",
    "n = 100\n",
    "x = I_obs[:, 0] * I_obs[:, 2]**(-1/3)\n",
    "y = I_obs[:, 1] * I_obs[:, 2]**(-2/3)\n",
    "z = I_obs[:, 2]\n",
    "\n",
    "# Create the plot\n",
    "fig = plt.figure(figsize=(10, 7))\n",
    "ax = fig.add_subplot(111, projection='3d')\n",
    "\n",
    "scatter = ax.scatter(x, y, z, alpha=0.6, cmap='viridis')\n",
    "\n",
    "# Add labels\n",
    "ax.set_xlabel('X Axis')\n",
    "ax.set_ylabel('Y Axis')\n",
    "ax.set_zlabel('Z Axis')\n",
    "ax.set_title('3D Scatter Plot Example')\n",
    "\n",
    "# Add a color bar\n",
    "fig.colorbar(scatter, ax=ax, label='Intensity')\n",
    "\n",
    "# Save the plot\n",
    "plt.savefig('3d_scatter_plot.png')\n",
    "plt.close()\n",
    "\n",
    "print(\"Plot saved as 3d_scatter_plot.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "639848b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "base_save_path = \"saved_model\"  # change as needed\n",
    "os.makedirs(base_save_path, exist_ok=True)\n",
    "\n",
    "# Subfolder with datetime\n",
    "timestamp = datetime.datetime.now().strftime(\"%Y%m%dT%H%M%S\")\n",
    "save_path = os.path.join(base_save_path, timestamp)\n",
    "os.makedirs(save_path, exist_ok=True)\n",
    "dataset = TractionDataset(\"dataset\",\"NH\")\n",
    "data = dataset[-1]\n",
    "coords = data[\"mesh_pos\"][:,:2]\n",
    "cells = data[\"cells\"]\n",
    "u = data[\"u\"]\n",
    "node_type = data[\"node_type\"]\n",
    "load_parameter = data[\"load_parameter\"]\n",
    "\n",
    "coord_cells = coords[cells]\n",
    "u_cells = u[cells]\n",
    "\n",
    "F, dNdx = deformation_gradient_element(coord_cells, u_cells)\n",
    "\n",
    "I_obs, _ = jax.vmap(invariants_and_derivatives)(F)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "287d8df0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Array([[ 4.06913074,  5.22308739,  2.15395665],\n",
       "       [ 4.06284735,  5.2076505 ,  2.14480315],\n",
       "       [ 4.05985881,  5.19940945,  2.13955064],\n",
       "       ...,\n",
       "       [18.80382604, 21.71412358,  3.91029755],\n",
       "       [27.54941972, 31.02362808,  4.47420836],\n",
       "       [21.64875583, 23.76546326,  3.11670743]], dtype=float64)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "I_obs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4620a74d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files in directory: ['.devcontainer', '.git', '.gitignore', '.vscode', '3d_scatter_plot.png', 'core', 'create_geometry.py', 'dataset', 'data_analysis.ipynb', 'data_generator.ipynb', 'deviatoric_strain_energy_r2_plot.png', 'docker-compose.yaml', 'Dockerfile', 'figure.png', 'figures', 'function_with_noise.png', 'gp.py', 'gp_plot.pdf', 'gp_plot.png', 'gp_plot1.png', 'kl_eigenfunctions.png', 'kl_eigenvalues.png', 'main.py', 'mesh.msh', 'mesh_with_hole.msh', 'optimization_log.txt', 'plate_two_holes.msh', 'plotklgaussian.py', 'poc_archived', 'psi_data.csv', 'psi_regression_plot.png', 'ref_literature', 'results', 'run_model.py', 'run_model_eval.py', 'saved_model', 'selected_model', 'strain_energy_r2_plot.png', 'training_loss.png', 'training_set_analysis.ipynb', 'uniaxial_strain_energy_plot.png', 'vfm.ipynb', 'vigp.ipynb', 'vigp.py', '__pycache__']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_2669/4237202470.py:20: UserWarning: No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored\n",
      "  scatter = ax.scatter(x, y, z, alpha=0.6, cmap='viridis')\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Plot saved as 3d_scatter_plot.png\n"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "import os\n",
    "\n",
    "# Check for files in current directory\n",
    "files = os.listdir('.')\n",
    "print(f\"Files in directory: {files}\")\n",
    "\n",
    "# Generate synthetic data for a 3D scatter plot\n",
    "n = 100\n",
    "x = I_obs[:, 0] * I_obs[:, 2]**(-1/3)\n",
    "y = I_obs[:, 1] * I_obs[:, 2]**(-2/3)\n",
    "z = I_obs[:, 2]\n",
    "\n",
    "# Create the plot\n",
    "fig = plt.figure(figsize=(10, 7))\n",
    "ax = fig.add_subplot(111, projection='3d')\n",
    "\n",
    "scatter = ax.scatter(x, y, z, alpha=0.6, cmap='viridis')\n",
    "\n",
    "# Add labels\n",
    "ax.set_xlabel('X Axis')\n",
    "ax.set_ylabel('Y Axis')\n",
    "ax.set_zlabel('Z Axis')\n",
    "ax.set_title('3D Scatter Plot Example')\n",
    "\n",
    "# Add a color bar\n",
    "fig.colorbar(scatter, ax=ax, label='Intensity')\n",
    "\n",
    "# Save the plot\n",
    "plt.savefig('3d_scatter_plot.png')\n",
    "plt.close()\n",
    "\n",
    "print(\"Plot saved as 3d_scatter_plot.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ca5d1fb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dolfinx-env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
