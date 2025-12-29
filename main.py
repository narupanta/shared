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
    dataset = TractionDataset("dataset","Isihara")
    data = dataset[-1]
    coords = data["mesh_pos"][:,:2]
    cells = data["cells"]
    u = data["u"]
    node_type = data["node_type"]
    u = u + jax.random.normal(jr.key(0), u.shape) * 0.0 * jnp.std(u)
    load_parameter = data["load_parameter"]

    coord_cells = coords[cells]
    u_cells = u[cells]

    F, dNdx = deformation_gradient_element(coord_cells, u_cells)
    
    I_obs, _ = invariants_and_derivatives(F)
    lscale_init = jnp.std(I_obs, axis = 0)
    x = I_obs[:, 0] * I_obs[:, 2]**(-1/3)
    y = I_obs[:, 1] * I_obs[:, 2]**(-2/3)
    z = (I_obs[:, 2] ** (1/2) - 1)**2
    I_obs_dev = jnp.stack([x, y, z], axis = -1)
    
    n_ip = 25
    # Z_i1 = jnp.linspace(I_obs[:, 0].min(), I_obs[:, 0].max(), n_ip)
    # Z_i2 = jnp.linspace(I_obs[:, 1].min(), I_obs[:, 1].max(), n_ip)
    # Z_i3 = jnp.linspace(I_obs[:, 2].min(), I_obs[:, 2].max(), n_ip)

    # Z_grid_i1, Z_grid_i2, Z_grid_i3 = jnp.meshgrid(Z_i1, Z_i2, Z_i3)
    # Z_stacked = jnp.stack([Z_grid_i1.flatten(), Z_grid_i2.flatten(), Z_grid_i3.flatten()], axis=-1)

    # Z_stacked = jnp.stack([Z_i1, Z_i2, Z_i3], axis = -1)

    from sklearn.cluster import KMeans

    # Find the 50 most representative points in your invariant data
    kmeans = KMeans(n_clusters=n_ip, random_state=0).fit(I_obs_dev)
    Z_stacked = jnp.array(kmeans.cluster_centers_)
    g_params = jnp.ones((n_ip,)) * 0.0
    # g_params = jax.random.normal(jr.key(0), shape=(n_ip,)) * 0.1 - 1


    params = {
        "lengthscales": lscale_init[:2],
        "log_scale_variance": jnp.array(-1.0),
        "log_sigma_poly": jnp.array(-5.0),
        "log_offset": jnp.array(-5.0),
        "log_growth_constant": jnp.array(1.0),
        "poly_degree": 2.0,
        "alpha": 1.0,
        "g_mean": g_params,
        "g_log_var": jnp.ones_like(g_params) * -5,
        "log_sigma_physic": -1.0
    }



    # choose optimizer
    # opt = optax.lbfgs(1e-3)
    opt = optax.adam(1e-2)
    opt_state = opt.init(params)
    n_nodes = int(cells.max()) + 1
    # JIT the loss and gradients
    loss_and_grad = jax.jit(jax.value_and_grad( # type: ignore
        lambda p: physical_loss(p, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter), 
                has_aux=True
    ))


    # loss_and_grad = jax.jit(jax.value_and_grad( # type: ignore
    #     lambda p: elbo_loss(p, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter, jr.key(0)),
    #     has_aux=True
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
    g_mean_history = []
    lengthscales_history = []
    log_scale_variance_history = []
    best_params = None
    for step in range(20000):
        (loss, (log_like_loss, kl_loss, phy_loss)), grads = loss_and_grad(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss) # Store loss
        if loss < best_loss: # type: ignore
            best_loss = loss
            best_params = params

        if step % 50 == 0:
            g_mean_history.append(params["g_mean"])
            lengthscales_history.append(params["lengthscales"])
            log_scale_variance_history.append(params["log_scale_variance"])
            print(f"step {step:04d}  loss={loss:.6f}, log_like_loss={log_like_loss:.6f}, kl_loss={kl_loss:.6f}, phy_loss = {phy_loss:.6f}")

    # def H_func(g_vec_opt, params, *model_args) :
    #     params = params
    #     def loss_wrt_g_vec(g_vec):
    #         current_params = params.copy()
    #         current_params["g_mean"] = g_vec
        
    #         return physical_loss(current_params, *model_args)
        
    #     return jax.hessian(loss_wrt_g_vec)(g_vec_opt)
    # H_matrix = H_func(params["g_mean"], params, Z_stacked, coord_cells, cells, u_cells, coords.shape[0], node_type, load_parameter)
    # M = params["g_mean"].shape[0]
    # jitter = 1e-6 * jnp.eye(M)
    # S_recovered = jnp.linalg.inv(H_matrix + jitter)
    # Save the best parameters
    with open(os.path.join(save_path, "best_params.npy"), "wb") as f:
        jnp.save(f, jax.tree_util.tree_map(lambda x: x, best_params)) # type: ignore
    jnp.save(os.path.join(save_path, "Z_stacked.npy"), Z_stacked)
    #
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss over Steps")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "training_loss.png"))

    # Plotting hyperparameter history
    plt.figure(figsize=(10, 6))
    lengthscales_history = jnp.array(lengthscales_history)
    for i in range(lengthscales_history.shape[1]):
        plt.plot(lengthscales_history[:, i], label=f"lengthscale_{i+1}")
    plt.xlabel("Training Step")
    plt.ylabel("Lengthscale Value")
    plt.title("Lengthscales over Training Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "lengthscales_history.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(jnp.array(log_scale_variance_history), label="log_scale_variance")
    plt.xlabel("Training Step")
    plt.ylabel("Log Scale Variance Value")
    plt.title("Log Scale Variance over Training Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "log_scale_variance_history.png"))

    plt.figure(figsize=(10, 6))
    plt.hist(best_params["g_mean"])
    plt.xlabel("Inducing Point Index")
    plt.ylabel("g_mean Value")
    plt.title("Final g_mean values for Inducing Points")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "g_mean_final.png"))

    # Plotting g_mean history
    plt.figure(figsize=(10, 6))
    g_mean_history = jnp.array(g_mean_history)
    for i in range(g_mean_history.shape[1]):
        plt.plot(g_mean_history[:, i], label=f"g_mean_{i+1}")
    plt.xlabel("Training Step")
    plt.ylabel("g_mean Value")
    plt.title("g_mean over Training Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "g_mean_history.png"))


    # R2 plot between true strain energy function and prediction
    # Assuming you have a true strain energy function (e.g., from material_models)
    # and you want to compare it with the learned SparseHyperelasticityGP.

    # 1. Instantiate the learned GP model

    learned_gp = SparseHyperelasticityGP(best_params["lengthscales"], best_params["log_scale_variance"], best_params["log_sigma_poly"], best_params["log_offset"], best_params["log_growth_constant"], best_params["poly_degree"],best_params["g_mean"], Z_stacked)

    f = jax.vmap(fto3x3)(F)
    psi_pred = jax.vmap(learned_gp.psi)(f)
    psi_dev_pred = psi_pred - jax.vmap(learned_gp.psi_vol_gp)(I_obs)
    psi_vol_pred = jax.vmap(learned_gp.psi_vol_gp)(I_obs)

    true_model = get_material("isihara")
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

    # # Plot R2 between psi_dev_pred and psi_dev_true
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

    # Equibiaxial Compression (using gamma < 1)
    gamma_ebc = jnp.linspace(0.9, 1.0, num_points)
    f_ebc = jnp.zeros((num_points, 3, 3))
    f_ebc = f_ebc.at[:,0,0].set(gamma_ebc)
    f_ebc = f_ebc.at[:,1,1].set(gamma_ebc)
    f_ebc = f_ebc.at[:,2,2].set(1)

    # Simple Shear
    gamma_ss = jnp.linspace(0.0, 1.0, num_points)
    f_ss = jnp.zeros((num_points, 3, 3))
    f_ss = f_ss.at[:,0,1].set(gamma_ss)
    f_ss = f_ss.at[:,0,0].set(1)
    f_ss = f_ss.at[:,1,1].set(1)
    f_ss = f_ss.at[:,2,2].set(1)

    psi_pred_ut = jax.vmap(learned_gp.psi)(f_ut)
    psi_dev_pred_var_ut = jnp.sqrt(jax.vmap(learned_gp.psi_dev_std, in_axes=(0, None))(f_ut, best_params))
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
    psi_dev_pred_var_ebt = jnp.sqrt(jax.vmap(learned_gp.psi_dev_std, in_axes=(0, None))(f_ebt, best_params))
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
    psi_dev_pred_var_ps = jnp.sqrt(jax.vmap(learned_gp.psi_dev_std, in_axes=(0, None))(f_ps, best_params))
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
    
    psi_pred_uc = jax.vmap(learned_gp.psi)(f_uc)
    psi_dev_pred_var_uc = jnp.sqrt(jax.vmap(learned_gp.psi_dev_std, in_axes=(0, None))(f_uc, best_params))
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

    # Equibiaxial Compression Plot
    psi_pred_ebc = jax.vmap(learned_gp.psi)(f_ebc)
    psi_dev_pred_var_ebc = jnp.sqrt(jax.vmap(learned_gp.psi_dev_std, in_axes=(0, None))(f_ebc, best_params))
    psi_true_ebc = true_model.phi(f_ebc)
    y_min = jnp.min(psi_pred_ebc)
    y_max = jnp.max(psi_pred_ebc)
    margin = (y_max - y_min) * 0.1
    plt.figure(figsize=(8, 6))
    plt.plot(gamma_ebc, psi_true_ebc, label="True")
    plt.plot(gamma_ebc, psi_pred_ebc, label="Predicted")
    plt.fill_between(gamma_ebc, psi_pred_ebc - 2 * psi_dev_pred_var_ebc, psi_pred_ebc + 2 * psi_dev_pred_var_ebc, alpha=0.2, label="95% CI")
    plt.xlabel("gamma")
    plt.ylabel("psi")
    plt.title("Equibiaxial Compression Test: Strain Energy vs. Gamma")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend(["True", "Predicted"])
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "equibiaxial_compression_strain_energy_plot.png"))

    # Simple Shear Plot
    psi_pred_ss = jax.vmap(learned_gp.psi)(f_ss)
    psi_dev_pred_var_ss = jnp.sqrt(jax.vmap(learned_gp.psi_dev_std, in_axes=(0, None))(f_ss, best_params))
    psi_true_ss = true_model.phi(f_ss)
    y_min = jnp.min(psi_pred_ss)
    y_max = jnp.max(psi_pred_ss)
    margin = (y_max - y_min) * 0.1
    plt.figure(figsize=(8, 6))
    plt.plot(gamma_ss, psi_true_ss, label="True")
    plt.plot(gamma_ss, psi_pred_ss, label="Predicted")
    plt.fill_between(gamma_ss, psi_pred_ss - 2 * psi_dev_pred_var_ss, psi_pred_ss + 2 * psi_dev_pred_var_ss, alpha=0.2, label="95% CI")
    plt.xlabel("gamma")
    plt.ylabel("psi")
    plt.title("Simple Shear Test: Strain Energy vs. Gamma")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend(["True", "Predicted"])
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "simple_shear_strain_energy_plot.png"))

    # Combined plot with 6 subplots (2x3)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Strain Energy vs. Gamma for Various Deformation Modes", fontsize=16)

    # Uniaxial Tension
    axs[0, 0].plot(gamma, psi_true_ut, label="True")
    axs[0, 0].plot(gamma, psi_pred_ut, label="Predicted")
    axs[0, 0].fill_between(gamma, psi_pred_ut - 2 * psi_dev_pred_var_ut, psi_pred_ut + 2 * psi_dev_pred_var_ut, alpha=0.2, label="95% CI")
    axs[0, 0].set_title("Uniaxial Tension")
    axs[0, 0].set_xlabel("Gamma")
    axs[0, 0].set_ylabel("Psi")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Equibiaxial Tension
    axs[0, 1].plot(gamma, psi_true_ebt, label="True")
    axs[0, 1].plot(gamma, psi_pred_ebt, label="Predicted")
    axs[0, 1].fill_between(gamma, psi_pred_ebt - 2 * psi_dev_pred_var_ebt, psi_pred_ebt + 2 * psi_dev_pred_var_ebt, alpha=0.2, label="95% CI")
    axs[0, 1].set_title("Equibiaxial Tension")
    axs[0, 1].set_xlabel("Gamma")
    axs[0, 1].set_ylabel("Psi")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Pure Shear
    axs[0, 2].plot(gamma, psi_true_ps, label="True")
    axs[0, 2].plot(gamma, psi_pred_ps, label="Predicted")
    axs[0, 2].fill_between(gamma, psi_pred_ps - 2 * psi_dev_pred_var_ps, psi_pred_ps + 2 * psi_dev_pred_var_ps, alpha=0.2, label="95% CI")
    axs[0, 2].set_title("Pure Shear")
    axs[0, 2].set_xlabel("Gamma")
    axs[0, 2].set_ylabel("Psi")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Uniaxial Compression
    axs[1, 0].plot(gamma_comp, psi_true_uc, label="True")
    axs[1, 0].plot(gamma_comp, psi_pred_uc, label="Predicted")
    axs[1, 0].fill_between(gamma_comp, psi_pred_uc - 2 * psi_dev_pred_var_uc, psi_pred_uc + 2 * psi_dev_pred_var_uc, alpha=0.2, label="95% CI")
    axs[1, 0].set_title("Uniaxial Compression")
    axs[1, 0].set_xlabel("Gamma")
    axs[1, 0].set_ylabel("Psi")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Equibiaxial Compression
    axs[1, 1].plot(gamma_ebc, psi_true_ebc, label="True")
    axs[1, 1].plot(gamma_ebc, psi_pred_ebc, label="Predicted")
    axs[1, 1].fill_between(gamma_ebc, psi_pred_ebc - 2 * psi_dev_pred_var_ebc, psi_pred_ebc + 2 * psi_dev_pred_var_ebc, alpha=0.2, label="95% CI")
    axs[1, 1].set_title("Equibiaxial Compression")
    axs[1, 1].set_xlabel("Gamma")
    axs[1, 1].set_ylabel("Psi")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Simple Shear
    axs[1, 2].plot(gamma_ss, psi_true_ss, label="True")
    axs[1, 2].plot(gamma_ss, psi_pred_ss, label="Predicted")
    axs[1, 2].fill_between(gamma_ss, psi_pred_ss - 2 * psi_dev_pred_var_ss, psi_pred_ss + 2 * psi_dev_pred_var_ss, alpha=0.2, label="95% CI")
    axs[1, 2].set_title("Simple Shear")
    axs[1, 2].set_xlabel("Gamma")
    axs[1, 2].set_ylabel("Psi")
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(save_path, "combined_strain_energy_plots.png"))
    plt.close(fig)