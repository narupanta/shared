import streamlit as st
import plotly.express as px
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
from core.material_models import get_material
from core.utils import *
from core.datasetclass import TractionDataset
from core.model import SparseHyperelasticityGP
import os

# 1. Page Config
st.set_page_config(page_title="Hyperelastic Energy Landscape", layout="wide")
st.title("Interactive 3D Strain Energy Visualization")

# 2. Sidebar Controls
st.sidebar.header("Data Parameters")
# Added the noise slider: 0.0 to 0.01, 10 steps (0.001 increment)
noise_level = st.sidebar.select_slider(
    "Adjust Noise Level", 
    options=np.linspace(0.0, 0.01, 11),
    value=0.0
)

st.sidebar.header("Plot Settings")
marker_size = st.sidebar.slider("Marker Size", 1, 10, 3)
opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.7)

# 3. Data Processing (Re-runs when slider moves)
@st.cache_data
def load_base_dataset():
    return TractionDataset("dataset", "Isihara")

dataset = load_base_dataset()
all_F = []

# Process the dataset with the selected noise
for i in range(len(dataset)):
    data = dataset[i]
    coords = data["mesh_pos"][:, :2]
    cells = data["cells"]
    u = data["u"]
    
    # Apply noise from slider
    u_noisy = u + jax.random.normal(jr.key(0), u.shape) * noise_level * jnp.std(u)
    
    coord_cells = coords[cells]
    u_cells = u_noisy[cells]

    # Calculate deformation gradient
    F_el, _ = deformation_gradient_element(coord_cells, u_cells)
    
    # Only taking the last snapshot as per your original logic
    if (i == len(dataset) - 1):
        all_F.append(F_el)

F_total = jnp.concatenate(all_F, axis=0)

# 4. Invariant and Energy Calculation
I_obs, _ = jax.vmap(invariants_and_derivatives)(F_total)
f_mat = jax.vmap(fto3x3)(F_total)
psi_vals = jax.vmap(get_material("isihara").phi)(f_mat) - 1.5 * (jnp.sqrt(I_obs[:, 2]) - 1)**2


# Deviatoric scaling
x_inv = I_obs[:, 0] * I_obs[:, 2]**(-1/3)
y_inv = I_obs[:, 1] * I_obs[:, 2]**(-2/3)
n_grid = 20
# Create a grid for I1_bar and I2_bar
i1_bar_grid = jnp.linspace(3.0, 6, n_grid)
i2_bar_grid = jnp.linspace(3.0, 6, n_grid)


# Generate all combinations of I1_bar and I2_bar
I1_grid_mesh, I2_grid_mesh = jnp.meshgrid(i1_bar_grid, i2_bar_grid)


I_star = jnp.stack([I1_grid_mesh.flatten(), I2_grid_mesh.flatten(), jnp.ones_like(I1_grid_mesh.flatten())], axis=-1)

# Assuming I3_bar is 1 for the grid (incompressible, deviatoric part)
# We need to convert I1_bar and I2_bar back to I1, I2, I3 for the material model
# For incompressible materials, I3 = 1, so I1_bar = I1, I2_bar = I2
# This is a simplification for plotting the deviatoric part of the energy landscape

# Create dummy F tensors that would result in these I1_bar, I2_bar values
# This is complex, so for visualization, we can directly use the I_obs_dev values
# and assume I3=1 for the grid points to calculate psi_grid

# For plotting the grid, we can directly use the deviatoric invariants
# and assume a constant I3 (e.g., 1) for the purpose of calculating psi on the grid.
# This is a conceptual grid for the deviatoric part.

# For simplicity, let's assume I3_grid = 1 for the grid points to calculate psi_grid

# Uniaxial Tension for plotting
num_ut_points = 50
gamma_ut = jnp.linspace(1.0, 4.0, num_ut_points)
f_ut_3x3 = jnp.zeros((num_ut_points, 3, 3))
f_ut_3x3 = f_ut_3x3.at[:, 0, 0].set(gamma_ut)
f_ut_3x3 = f_ut_3x3.at[:, 1, 1].set(1.0)
f_ut_3x3 = f_ut_3x3.at[:, 2, 2].set(1.0)
I_ut, _ = jax.vmap(invariants_and_derivatives)(f_ut_3x3)
x_ut_inv = I_ut[:, 0] * I_ut[:, 2]**(-1/3)
y_ut_inv = I_ut[:, 1] * I_ut[:, 2]**(-2/3)
psi_ut_vals = jax.vmap(get_material("isihara").phi)(f_ut_3x3) - 1.5 * (jnp.sqrt(I_ut[:, 2]) - 1)**2



psi_grid_vals = 0.5 * (I1_grid_mesh - 3) + (I2_grid_mesh - 3) + (I1_grid_mesh - 3)**2 

# # 5. Create DataFrame
plot_df = pd.DataFrame({
    'I1_deviatoric': x_inv - 3,
    'I2_deviatoric': y_inv - 3,
    'Psi': psi_vals.flatten()
})

# 6. Create & Display Plotly Figure
# Add grid data to the DataFrame for plotting

model_path = "saved_model/20251228T141342" # Replace with the actual path to your saved model
with open(os.path.join(model_path, "best_params.npy"), "rb") as f:
    best_params = jnp.load(f, allow_pickle=True).item()

with open(os.path.join(model_path, "Z_stacked.npy"), "rb") as f:
    Z_stacked = jnp.load(f, allow_pickle=True)

learned_gp = SparseHyperelasticityGP(best_params["lengthscales"], best_params["log_scale_variance"], best_params["log_sigma_poly"], best_params["log_offset"], best_params["log_growth_constant"], best_params["poly_degree"], best_params["g_mean"], Z_stacked)
# deformation_gradient_ref = jnp.eye(deformation_gradient.shape[-1])
# E =  0.5 * (deformation_gradient.T @ deformation_gradient - jnp.eye(deformation_gradient.shape[-1]))

# H = jax.grad(self.psi_gp_f)(deformation_gradient_ref)
# stress_correction = jnp.sum(H * E)
# psi = self.psi_gp_f(deformation_gradient) - self.psi_gp_f(deformation_gradient_ref) - stress_correction

psi_dev_gp = jax.vmap(learned_gp.psi_dev_gp)(I_obs) - learned_gp.psi_dev_gp(jnp.array([3, 3, 1]))



# grid_df = pd.DataFrame({
#     'I1_deviatoric': I1_grid_mesh.flatten() - 3,
#     'I2_deviatoric': I2_grid_mesh.flatten() - 3,
#     'Psi': psi_dev_gp.flatten()
# })

grid_df = pd.DataFrame({
    'I1_deviatoric': x_inv- 3,
    'I2_deviatoric': y_inv - 3,
    'Psi': psi_dev_gp.flatten()
})

ut_df = pd.DataFrame({
    'I1_deviatoric': x_ut_inv - 3,
    'I2_deviatoric': y_ut_inv - 3,
    'Psi': psi_ut_vals.flatten()
})


fig = px.scatter_3d(
    plot_df, 
    x='I1_deviatoric', 
    y='I2_deviatoric', 
    z='Psi',
    color='Psi',
    color_continuous_scale='Viridis',
    opacity=opacity,
    labels={'I1_deviatoric': 'I1 - 3', 'I2_deviatoric': 'I2 - 3', 'Psi': 'Energy (Psi)'},
    title=f"Energy Landscape (Noise: {noise_level:.4f})"
)

# Add the grid as a surface plot or additional scatter points
fig.add_trace(
    px.scatter_3d(
        grid_df,
        x='I1_deviatoric',
        y='I2_deviatoric',
        z='Psi',
        color='Psi',
        color_continuous_scale='Viridis',
        opacity=0.5, # Make grid points slightly transparent
        symbol_sequence=['circle-open']).data[0]) # Use a different symbol for grid points

# Add Uniaxial Tension path
fig.add_trace(
    px.line_3d(
        ut_df,
        x='I1_deviatoric',
        y='I2_deviatoric',
        z='Psi',
        line_dash_sequence=['dash'],
        color_discrete_sequence=['red']).data[0]
)

# Add Z_stacked (inducing points) as red markers
Z_stacked_df = pd.DataFrame({
 'I1_deviatoric': Z_stacked[:, 0] - 3,
 'I2_deviatoric': Z_stacked[:, 1] - 3,
 'Psi': jax.vmap(learned_gp.psi_dev_gp)(Z_stacked) - learned_gp.psi_dev_gp(jnp.array([3, 3, 1]))
})
fig.add_trace(
 px.scatter_3d(
        Z_stacked_df,
 x='I1_deviatoric', y='I2_deviatoric', z='Psi',
 color_discrete_sequence=['red'], symbol_sequence=['x']).data[0]
)
fig.update_traces(marker=dict(size=marker_size))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

st.plotly_chart(fig, use_container_width=True)

st.write(f"Displaying **{len(plot_df)}** points. Noise applied to displacements: **{noise_level*100:.2f}%**")