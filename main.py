import jax 
import gpjax as gpx
import jax.numpy as jnp
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
from core.model import SparseHyperelasticityGP
from core.material_models import get_material
import jax
import jax.numpy as jnp
from core.utils import *
import datetime
import os

from core.datasetclass import TractionDataset
from core.loss_function import physical_loss, elbo_loss
# helper: per-element edge-based neumann traction contribution

@jax.vmap
def invariants_and_derivatives(F):
    f = fto3x3(F)
    C = f.T @ f
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C @ C))
    I3 = jnp.linalg.det(C)
    # derivatives wrt F (2x2)
    dI1_dF = 2*f
    dI2_dF = 2*(I1*f - f @ C)
    dI3_dF = 2*jnp.linalg.det(f)**2 * jnp.linalg.inv(f).T
    dI_dF = jnp.stack([dI1_dF, dI2_dF, dI3_dF])  # (3,2,2)
    return jnp.array([I1, I2, I3]), dI_dF
if __name__ == "__main__" :
    base_save_path = "saved_model"  # change as needed
    os.makedirs(base_save_path, exist_ok=True)

    # Subfolder with datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    save_path = os.path.join(base_save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)
    dataset = TractionDataset("dataset","NH")
    data = dataset[-1]
    coords = data["mesh_pos"][:,:2]
    cells = data["cells"]
    u = data["u"]
    node_type = data["node_type"]
    load_parameter = data["load_parameter"]

    coord_cells = coords[cells]
    u_cells = u[cells]

    F, dNdx = deformation_gradient_element(coord_cells, u_cells)
    
    I_obs, _ = invariants_and_derivatives(F)
    n_ip = 10
    # Z_i1 = jnp.linspace(I_obs[:, 0].min(), I_obs[:, 0].max(), n_ip)
    # Z_i2 = jnp.linspace(I_obs[:, 1].min(), I_obs[:, 1].max(), n_ip) # I2 is not used in NH model
    # Z_i3 = jnp.linspace(I_obs[:, 2].min(), I_obs[:, 2].max(), n_ip) # I3 is not used in NH model

    # Z_stacked = jnp.stack([Z_i1, Z_i2, Z_i3], axis = -1)

    from sklearn.cluster import KMeans

    # Find the 50 most representative points in your invariant data
    kmeans = KMeans(n_clusters=n_ip, random_state=0).fit(I_obs)
    Z_stacked = jnp.array(kmeans.cluster_centers_)
    g_params = jnp.ones((n_ip,)) * -1.0
    # g_params = jax.random.normal(jr.key(0), shape=(n_ip,)) * 0.1 - 1


    params = {
        "lengthscales": jnp.array([1.0, 1.0]),
        "variance": jnp.array(1.0),
        "log_growth_constant": jnp.array(1.0),
        "g_mean": g_params,
        "g_log_var": jnp.ones_like(g_params) * 1e-2,
        "log_sigma_n": 1e-3
    }



    # choose optimizer
    # opt = optax.lbfgs(1e-3)
    opt = optax.adam(1e-2)
    opt_state = opt.init(params)
    n_nodes = int(cells.max()) + 1
    # JIT the loss and gradients
    loss_and_grad = jax.jit(jax.value_and_grad( # type: ignore
        lambda p: physical_loss(p, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter)
    ))


    # loss_and_grad = jax.jit(jax.value_and_grad( # type: ignore
    #     lambda p: elbo_loss(p, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter, jr.key(0))
    # ))


    # for step in range(20000):
    #     loss, grads = loss_and_grad(params)
    #     updates, opt_state = opt.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)

    #     if step % 50 == 0:
    #         print(f"step {step:04d}  loss={loss:.6f}")
    
    # Plotting training progress
    # (You'll need to store losses in a list during training to plot them)
    # For example:
    losses = []
    best_loss = float('inf')
    best_params = None
    for step in range(50000):
        loss, grads = loss_and_grad(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss) # Store loss
        if loss < best_loss: # type: ignore
            best_loss = loss
            best_params = params

        if step % 50 == 0:
            print(f"step {step:04d}  loss={loss:.6f}")

    def H_func(g_vec_opt, params, *model_args) :
        params = params
        def loss_wrt_g_vec(g_vec):
            current_params = params.copy()
            current_params["g_mean"] = g_vec
        
            return physical_loss(current_params, *model_args)
        
        return jax.hessian(loss_wrt_g_vec)(g_vec_opt)
    H_matrix = H_func(params["g_mean"], params, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter)
    M = params["g_mean"].shape[0]
    jitter = 1e-6 * jnp.eye(M)
    S_recovered = jnp.linalg.inv(H_matrix + jitter)

    # Save the best parameters
    with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
        jnp.save(f, jax.tree_util.tree_map(lambda x: x, best_params)) # type: ignore
    #
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss over Steps")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "training_loss.png"))

    # R2 plot between true strain energy function and prediction
    # Assuming you have a true strain energy function (e.g., from material_models)
    # and you want to compare it with the learned SparseHyperelasticityGP.

    # 1. Instantiate the learned GP model
    learned_gp = SparseHyperelasticityGP(best_params["lengthscales"], best_params["variance"], best_params["log_growth_constant"], best_params["g_mean"], Z_stacked)

    f = jax.vmap(fto3x3)(F)
    psi_pred = jax.vmap(learned_gp.psi)(f)
    psi_dev_pred = psi_pred - jax.vmap(learned_gp.psi_vol_)(I_obs)
    psi_vol_pred = jax.vmap(learned_gp.psi_vol_)(I_obs)

    true_model = get_material("neohookean")
    psi_true = true_model.phi(f)
    psi_dev_true = psi_true - 1.5 * (jnp.sqrt(I_obs[:, 2]) - 1)**2
    psi_vol_true = 1.5 * (jnp.sqrt(I_obs[:, 2]) - 1)**2


    # Plot R2 between psi_pred and psi_true
    plt.figure(figsize=(8, 6))
    
    # Flatten the arrays for plotting and R2 calculation
    psi_pred_flat = psi_pred.flatten()
    psi_true_flat = psi_true.flatten()



    # Calculate R2 score
    r2 = jnp.corrcoef(psi_true_flat, psi_pred_flat)[0, 1]**2

    plt.scatter(psi_true_flat, psi_pred_flat, alpha=0.7)
    
    # Add a 45-degree line
    min_val = min(psi_true_flat.min(), psi_pred_flat.min())
    max_val = max(psi_true_flat.max(), psi_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')

    plt.xlabel("True Strain Energy (psi_true)")
    plt.ylabel("Predicted Strain Energy (psi_pred)")
    plt.title(f"Strain Energy Prediction vs. True (R² = {r2:.4f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "strain_energy_r2_plot.png"))

    # Plot R2 between psi_dev_pred and psi_dev_true
    plt.figure(figsize=(8, 6))

    # Flatten the arrays for plotting and R2 calculation
    psi_dev_pred_flat = psi_dev_pred.flatten()
    psi_dev_true_flat = psi_dev_true.flatten()

    # Calculate R2 score
    r2_dev = jnp.corrcoef(psi_dev_true_flat, psi_dev_pred_flat)[0, 1]**2

    plt.scatter(psi_dev_true_flat, psi_dev_pred_flat, alpha=0.7)

    # Add a 45-degree line
    min_val_dev = min(psi_dev_true_flat.min(), psi_dev_pred_flat.min())
    max_val_dev = max(psi_dev_true_flat.max(), psi_dev_pred_flat.max())
    plt.plot([min_val_dev, max_val_dev], [min_val_dev, max_val_dev], color='red', linestyle='--', label='y=x')

    plt.xlabel("True Deviatoric Strain Energy (psi_dev_true)")
    plt.ylabel("Predicted Deviatoric Strain Energy (psi_dev_pred)")
    plt.title(f"Deviatoric Strain Energy Prediction vs. True (R² = {r2_dev:.4f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "deviatoric_strain_energy_r2_plot.png"))

    # Plot R2 between psi_vol_pred and psi_vol_true
    plt.figure(figsize=(8, 6))

    # Flatten the arrays for plotting and R2 calculation
    psi_vol_pred_flat = psi_vol_pred.flatten()
    psi_vol_true_flat = psi_vol_true.flatten()

    # Calculate R2 score
    r2_vol = jnp.corrcoef(psi_vol_true_flat, psi_vol_pred_flat)[0, 1]**2

    plt.scatter(psi_vol_true_flat, psi_vol_pred_flat, alpha=0.7)

    # Add a 45-degree line
    min_val_vol = min(psi_vol_true_flat.min(), psi_vol_pred_flat.min())
    max_val_vol = max(psi_vol_true_flat.max(), psi_vol_pred_flat.max())
    plt.plot([min_val_vol, max_val_vol], [min_val_vol, max_val_vol], color='red', linestyle='--', label='y=x')

    plt.xlabel("True Volumetric Strain Energy (psi_vol_true)")
    plt.ylabel("Predicted Volumetric Strain Energy (psi_vol_pred)")
    plt.title(f"Volumetric Strain Energy Prediction vs. True (R² = {r2_vol:.4f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "volumetric_strain_energy_r2_plot.png"))

    # --- Deformation Modes Plots ---
    num_points = 100
    gamma_range = (1.0, 2.0)
    gamma = jnp.linspace(gamma_range[0], gamma_range[1], num_points)

    # Uniaxial Tension
    f_ut = jnp.zeros((num_points, 3, 3))
    f_ut = f_ut.at[:,0,0].set(gamma)
    f_ut = f_ut.at[:,1,1].set(1)
    f_ut = f_ut.at[:,2,2].set(1)

    # Equibiaxial Tension
    f_ebt = jnp.zeros((num_points, 3, 3))
    f_ebt = f_ebt.at[:,0,0].set(gamma)
    f_ebt = f_ebt.at[:,1,1].set(gamma)
    f_ebt = f_ebt.at[:,2,2].set(1)

    # Pure Shear
    f_ps = jnp.zeros((num_points, 3, 3))
    f_ps = f_ps.at[:,0,0].set(gamma)
    f_ps = f_ps.at[:,1,1].set(1/gamma)
    f_ps = f_ps.at[:,2,2].set(1)

    # Uniaxial Compression (using gamma < 1)
    gamma_comp = jnp.linspace(0.9, 1.0, num_points)
    f_uc = jnp.zeros((num_points, 3, 3))
    f_uc = f_uc.at[:,0,0].set(gamma_comp)
    f_uc = f_uc.at[:,1,1].set(1)
    f_uc = f_uc.at[:,2,2].set(1)

    psi_pred_ut = jax.vmap(learned_gp.psi)(f_ut)
    psi_dev_pred_var_ut = learned_gp.psi_dev_std(f_ut, S_recovered)
    psi_true_ut = true_model.phi(f_ut)
    y_min = jnp.min(psi_pred_ut)
    y_max = jnp.max(psi_pred_ut)
    margin = (y_max - y_min) * 0.1

    plt.figure(figsize=(8, 6))
    plt.plot(gamma, psi_true_ut, label="True")
    plt.plot(gamma, psi_pred_ut, label="Predicted")
    plt.fill_between(gamma, psi_pred_ut - 2 * psi_dev_pred_var_ut, psi_pred_ut + 2 * psi_dev_pred_var_ut, alpha=0.2, label="95% CI")
    plt.xlabel("gamma")
    plt.ylabel("psi")
    plt.title("Uniaxial Test: Strain Energy vs. Gamma")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend(["True", "Predicted"])
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "uniaxial_strain_energy_plot.png"))

    # Equibiaxial Tension Plot
    psi_dev_pred_var_ebt = learned_gp.psi_dev_std(f_ebt, S_recovered)
    psi_pred_ebt = jax.vmap(learned_gp.psi)(f_ebt)
    psi_true_ebt = true_model.phi(f_ebt)
    y_min = jnp.min(psi_pred_ebt)
    y_max = jnp.max(psi_pred_ebt)
    margin = (y_max - y_min) * 0.1
    plt.figure(figsize=(8, 6))
    plt.plot(gamma, psi_true_ebt, label="True")
    plt.plot(gamma, psi_pred_ebt, label="Predicted")
    plt.fill_between(gamma, psi_pred_ebt - 2 * psi_dev_pred_var_ebt, psi_pred_ebt + 2 * psi_dev_pred_var_ebt, alpha=0.2, label="95% CI")
    plt.xlabel("gamma")
    plt.ylabel("psi")
    plt.title("Equibiaxial Test: Strain Energy vs. Gamma")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend(["True", "Predicted"])
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "equibiaxial_strain_energy_plot.png"))

    # Pure Shear Plot
    psi_dev_pred_var_ps = learned_gp.psi_dev_std(f_ps, S_recovered)
    psi_pred_ps = jax.vmap(learned_gp.psi)(f_ps)
    psi_true_ps = true_model.phi(f_ps)
    y_min = jnp.min(psi_pred_ps)
    y_max = jnp.max(psi_pred_ps)
    margin = (y_max - y_min) * 0.1
    plt.figure(figsize=(8, 6))
    plt.plot(gamma, psi_true_ps, label="True")
    plt.plot(gamma, psi_pred_ps, label="Predicted")
    plt.fill_between(gamma, psi_pred_ps - 2 * psi_dev_pred_var_ps, psi_pred_ps + 2 * psi_dev_pred_var_ps, alpha=0.2, label="95% CI")
    plt.xlabel("gamma")
    plt.ylabel("psi")
    plt.title("Pure Shear Test: Strain Energy vs. Gamma")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend(["True", "Predicted"])
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "pure_shear_strain_energy_plot.png"))

    # Uniaxial Compression Plot
    psi_dev_pred_var_uc = learned_gp.psi_dev_std(f_uc, S_recovered)
    psi_pred_uc = jax.vmap(learned_gp.psi)(f_uc)
    psi_true_uc = true_model.phi(f_uc)
    y_min = jnp.min(psi_pred_uc)
    y_max = jnp.max(psi_pred_uc)
    margin = (y_max - y_min) * 0.1
    plt.figure(figsize=(8, 6))
    plt.plot(gamma_comp, psi_true_uc, label="True")
    plt.plot(gamma_comp, psi_pred_uc, label="Predicted")
    plt.fill_between(gamma_comp, psi_pred_uc - 2 * psi_dev_pred_var_uc, psi_pred_uc + 2 * psi_dev_pred_var_uc, alpha=0.2, label="95% CI")
    plt.xlabel("gamma")
    plt.ylabel("psi")
    plt.title("Uniaxial Compression Test: Strain Energy vs. Gamma")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend(["True", "Predicted"])
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "uniaxial_compression_strain_energy_plot.png"))